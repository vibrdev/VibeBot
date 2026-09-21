"""The 'System 2' layer: a small LLM that only gets called when Laya defers."""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from ..schema import Element, Observation

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the reasoning half of a browser agent working for a user.
A fast model (Laya) takes the obvious clicks; you are called for everything that
needs reading or judgement. You see one page at a time.

Each step you get:
- GOAL: what the user wants.
- NOTES: facts you saved on earlier pages. They are ALL you remember of pages you
  have left - anything you did not save is gone.
- PAGE: the current page as text, in reading order. Everything you can act on
  appears inline as [number] label. The site's header, menus and footer come last.
- A screenshot of the visible part of the page, when available.
- STEPS SO FAR, and why you were called.

Reply with ONE JSON object and nothing else:

{
  "seen": "<what on THIS page matters for the goal: names, prices, counts, which filters are on>",
  "plan": ["<the next steps AFTER this one, short, one action each - at least two unless the next step is answering>"],
  "op": "click|type|select|scroll|navigate|back|wait|extract|switch_tab|close_tab|ask_user|done|fail",
  "element_idx": <a [number] from PAGE, or null>,
  "text": "<text to type / option to select / url / tab number / question / final answer>",
  "notes": ["<any other fact worth keeping for later>"],
  "reason": "<one short sentence>",
  "confidence": <0.0-1.0>
}

Always fill in "seen" first - look before you act. It is saved to NOTES for you.
Then always fill in "plan": every step you plan is one you are not called for.

"plan" is handed to a fast helper that carries the steps out without asking
you, so write each step so it can be done by clicking or typing one visible
thing, and put the exact label in quotes:
  "type 'macbook pro' into the search box"
  "click 'Used'"
  "click 'Price: lowest first'"
  "click 'Next page'"
  "read the page and answer"
Make the last step "read the page and answer" when the answer should then be
visible. You are called back if a step cannot be done, fails, or when it is
time to read and answer.

How to work:
- Read PAGE first. The answer, or the control you need, is often already on it.
- Use the site's own tools. Search, filters and "sort by price" beat scrolling
  through everything.
- Search with two to four key words ("macbook pro"), then narrow with the
  site's filters. A whole sentence as a search usually finds nothing. If a
  search or filter leaves no results, loosen it rather than add more.
- Filters are often toggles: clicking an active one (shown in bold or
  "selected") turns it off again. Check what is already applied.
- Save what you find in "notes" as you find it, e.g.
  "Used MacBook Pro 16in M2 Max 96GB 2TB - 21,450 SEK (seller nordic_macs)".
  Leave "notes" empty when there is nothing new.
- For "cheapest", "most", "how many" or "all": be sure you have seen every
  candidate - every page of results, or a list sorted by what you compare.
- Check every condition in the goal against each candidate. Memory (RAM,
  unified memory) is not storage (SSD, TB). New, used, refurbished and "for
  parts" are different conditions.
- When you have the answer, reply "done" with the exact answer in "text" -
  numbers, prices and names as the page shows them.

Rules:
- Only use element_idx values that appear as [number] in PAGE.
- "type" needs both element_idx and text. It submits with Enter afterwards.
- Elements marked "(iframe N)" live inside an embedded frame. Act on them
  normally — the index is all you need.
- "switch_tab" / "close_tab" take the tab number in "text". The agent already
  follows tabs the page opens, so switch only to go back to an earlier one.
- "extract" means the goal was an information request: put the answer in "text".
- "done" means the goal is finished: put a short result summary in "text".
- "ask_user" is for missing information only you cannot invent (which account,
  which date, a confirmation). Put the question in "text". Use it sparingly.
- "fail" if the goal is impossible here; explain in "text".
- Never invent credentials. Never guess at payment or destructive steps —
  ask_user instead.
"""


@dataclass
class LLMAction:
    op: str = "ask_user"
    element_idx: int | None = None
    text: str = ""
    reason: str = ""
    confidence: float = 0.0
    raw: str = ""
    """The model's reply verbatim (truncated). Logged, so a bad step can be
    read back afterwards instead of guessed at."""
    prompt: str = ""
    """The prompt that produced it. Same reason."""
    notes: list[str] = field(default_factory=list)
    """Facts the model asked to keep. The agent carries them from page to page;
    without them it could not compare a listing on page 1 with one on page 3."""
    plan: list[str] = field(default_factory=list)
    """The steps after this one, for the fast decider to carry out. This is
    the system-1/system-2 split: the LLM decides what to do, rarely; Laya or
    Jev does it, one narrow "which element is this?" question at a time."""
    seen: str = ""
    """What the model says this page shows, written before it picks an action.

    Optional notes did not work with qwen3.5:4b: over two benchmark rounds it
    saved two notes in 67 steps. Small models reliably fill in whatever comes
    first in the reply format, so observation is now the first field and the
    agent files it into memory itself."""


class LLM(Protocol):
    name: str

    def available(self) -> bool: ...

    async def ping(self) -> tuple[bool, str]:
        """(usable, human-readable reason). `vibebot doctor` and Agent.start
        both call this, so it belongs in the contract."""
        ...

    async def decide(
        self,
        goal: str,
        obs: Observation,
        candidates: list[Element],
        history: list[str],
        note: str,
        screenshot_png: bytes | None,
    ) -> LLMAction: ...

    async def ask(self, prompt: str) -> str: ...

    async def close(self) -> None: ...


#: A page that says it found nothing.
NO_RESULTS = re.compile(
    r"\b(?:0|no|zero)\s+(?:results?|listings?|items?|matches|products?|hits)\b"
    r"|no listings match|did not match any|nothing (?:was )?found|no results found",
    re.I,
)


_SEARCH_BOX = re.compile(r"^\[(\d+)\] \(search box", re.M)
_QUERY_KEYS = ("q", "query", "search", "k", "keyword", "keywords", "_nkw", "term", "s")


def _no_results_alert(obs: Observation) -> str:
    """Say what to change, not just that nothing was found.

    A vaguer version ("broaden the search or remove a filter") backfired: the
    search "used MacBook Pro M-series" matched nothing because no title says
    "M-series", the model chose "remove a filter", still saw no results, and
    clicked the one filter it knew on and off eleven times. It never touched
    the search box. So name the words and the box.
    """
    from urllib.parse import parse_qs, urlsplit  # noqa: PLC0415

    params = parse_qs(urlsplit(obs.url or "").query)
    query = next((params[k][0] for k in _QUERY_KEYS if params.get(k) and params[k][0].strip()), "")
    box = _SEARCH_BOX.search(obs.page_text or "")
    where = f" in the search box [{box.group(1)}]" if box else ""
    if query:
        return (
            f'!! THIS PAGE SHOWS NO RESULTS. The search "{query}" matches nothing - the words '
            f"themselves are the problem, so no filter can fix it. Type a shorter search{where}: "
            "just the product (e.g. two words), then narrow with filters."
        )
    return (
        "!! THIS PAGE SHOWS NO RESULTS. Adding filters to an empty list cannot help. Remove a "
        f"filter, or search again with fewer words{where}."
    )


def render_prompt(
    goal: str,
    obs: Observation,
    candidates: list[Element],
    history: list[str],
    note: str,
) -> str:
    lines = [
        f"GOAL: {goal}",
        "",
        f"PAGE: {obs.title}",
        f"URL: {obs.url}",
    ]
    if len(obs.tabs) > 1:
        lines += ["", f"TABS ({len(obs.tabs)} open, * = current):"]
        lines += [f"  {tab.label()}" for tab in obs.tabs]
    lines += ["", "NOTES:"]
    lines += [f"  - {item}" for item in obs.memory] or ["  (nothing saved yet)"]
    if obs.plan:
        lines += ["", "YOUR PLAN (steps not done yet):"]
        lines += [f"  {i + 1}. {item}" for i, item in enumerate(obs.plan)]
    if obs.page_text and NO_RESULTS.search(obs.page_text):
        lines += ["", _no_results_alert(obs)]
    if obs.page_text:
        shown = len(obs.page_text)
        extent = (
            f"all {obs.page_text_total} characters"
            if obs.page_text_total <= shown
            else f"{shown} of {obs.page_text_total} characters"
        )
        lines += ["", f"PAGE ({extent}):", obs.page_text]
    else:
        # No reader (it failed, or was turned off): the old view, a short
        # element list and the top of the page.
        lines += ["", "ELEMENTS:"]
        lines += [f"  [{el.idx}] {el.label(110)}" for el in candidates]
        lines += ["", "VISIBLE TEXT (truncated):", " ".join(obs.text_digest.split())[:1200]]
    lines += ["", "STEPS SO FAR:"]
    lines += [f"  {i + 1}. {item}" for i, item in enumerate(history[-8:])] or ["  (none)"]
    if note:
        lines += ["", f"WHY YOU WERE CALLED: {note}"]
    lines += ["", "Respond with the JSON object only."]
    return "\n".join(lines)


def parse_action(content: str) -> LLMAction:
    """Small models wrap JSON in prose and code fences. Dig it out."""
    text = (content or "").strip()
    blob = _first_json_object(text)
    if blob is None:
        return LLMAction(op="ask_user", text="I could not parse my own plan — what should I do next?", raw=text)
    try:
        data: dict[str, Any] = json.loads(blob)
    except json.JSONDecodeError:
        return LLMAction(op="ask_user", text="I produced invalid JSON — what should I do next?", raw=text)

    idx = _as_element_idx(data.get("element_idx"))

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5

    return LLMAction(
        op=str(data.get("op") or data.get("operation") or "ask_user").strip().lower(),
        element_idx=idx,
        text=str(data.get("text") or ""),
        reason=str(data.get("reason") or ""),
        confidence=max(0.0, min(1.0, confidence)),
        raw=text[:2000],
        notes=_as_notes(data.get("notes", data.get("note", data.get("remember")))),
        seen=" ".join(str(data.get("seen") or data.get("observation") or "").split())[:400],
        plan=_as_notes(data.get("plan", data.get("next_steps"))),
    )


def _as_notes(value: Any) -> list[str]:
    """Whatever the model put under "notes", as a clean list of short strings.

    Small models are loose about shape: a list, one string, a dict of
    label -> fact, or null all turn up. None of them should cost a step.
    """
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, dict):
        items = [f"{k}: {v}" for k, v in value.items()]
    elif isinstance(value, (list, tuple)):
        items = [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v) for v in value]
    else:
        items = [str(value)]
    cleaned = []
    for item in items:
        item = " ".join(item.split())[:300]
        if item and item.lower() not in {"none", "null", "n/a", "-"}:
            cleaned.append(item)
    return cleaned[:8]


def _as_element_idx(value: Any) -> int | None:
    """The element number, however the model chose to spell it.

    Observed from qwen3.5:4b on a real run: it answered `"element_idx": [18]`,
    a one-item list, having correctly picked eBay's search box. The old parser
    accepted only int/str/float, so that became None and the agent fell back to
    scrolling — the model's one good decision of the run, discarded. The prompt
    numbers elements "e18", so a model echoing that spelling is just as likely.
    """
    if isinstance(value, bool):  # bool is an int subclass; not an index
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, (list, tuple)):
        # Only a single choice is meaningful; two elements is not an action.
        return _as_element_idx(value[0]) if len(value) == 1 else None
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        return int(match.group()) if match else None
    return None


def _first_json_object(text: str) -> str | None:
    """Pull one JSON object out of a model reply.

    Two things measured against the real models forced this to be more than a
    brace counter. A `}` inside a string value ("click the {x} button") used to
    close the object early and produce garbage, so the scan now tracks strings
    and escapes. And a reply cut off by the token budget never balances at all
    — see :func:`_repair`.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return _repair(text[start:], depth, in_string)


def _repair(blob: str, depth: int, in_string: bool) -> str | None:
    """Close a reply that the token budget cut off mid-object.

    Ollama reports this as done_reason "length" and hands back whatever it had
    written. Measured on qwen3.5:4b at num_predict=60, that is every field the
    agent needs and no closing brace — a complete, correct action thrown away
    over one character. So: close an open string, drop a dangling key or
    trailing comma, close the open braces, and if it still will not parse, shed
    the last field and try again.
    """
    if in_string:
        blob += '"'
    blob = re.sub(r',\s*"[^"]*"\s*:\s*$', "", blob)

    for _ in range(12):
        candidate = re.sub(r",\s*$", "", blob) + "}" * depth
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            cut = blob.rfind(",")
            if cut <= 0:
                return None
            blob = blob[:cut]
            continue
        return candidate
    return None


def b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


class LLMHTTPError(RuntimeError):
    """An HTTP error that still carries the server's own explanation."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def raise_for_status(response: httpx.Response) -> None:
    """Like `response.raise_for_status()`, but keeps the body.

    httpx renders a failed response as nothing but its status line and a link
    to MDN, so a model that fails to load reached the user as a bare "Server
    error '500 Internal Server Error'" with nothing to act on. Everything
    useful is in the body.

    Measured against Ollama 0.34.2, a load that runs out of memory answers:

        {"error": "llama-server startup failed after projector CPU offload
                   retry: ... cudaMalloc failed: out of memory ..."}

    OpenAI-compatible servers nest the same thing one level down, under
    {"error": {"message": ...}}, so both shapes are unwrapped here.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = _detail(response)
        host = response.request.url.host
        message = f"HTTP {response.status_code} from {host}"
        raise LLMHTTPError(f"{message}: {detail}" if detail else message, response.status_code) from exc


def _detail(response: httpx.Response, limit: int = 400) -> str:
    """The server's explanation, as one line short enough to show a user."""
    text = ""
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - error bodies are not always JSON
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if isinstance(error, dict):
            error = error.get("message") or error.get("error") or ""
        if isinstance(error, str):
            text = error
    if not text.strip():
        text = response.text or ""
    text = " ".join(text.split())
    return text[: limit - 1] + "…" if len(text) > limit else text
