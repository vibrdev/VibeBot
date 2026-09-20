"""The 'System 2' layer: a small LLM that only gets called when Laya defers."""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from ..schema import Element, Observation

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the reasoning half of a browser agent.

A fast local model (Laya) handles the obvious steps. You are called only when it
is unsure, so the step in front of you is genuinely ambiguous. Think, then act.

You get: the user's goal, the current page, a numbered list of interactive
elements, and (when available) a screenshot with those numbers drawn on it.

Reply with ONE JSON object and nothing else:

{
  "op": "click|type|select|scroll|navigate|back|wait|extract|switch_tab|close_tab|ask_user|done|fail",
  "element_idx": <number from the list, or null>,
  "text": "<text to type / option to select / url / tab number / question / final answer>",
  "reason": "<one short sentence>",
  "confidence": <0.0-1.0>
}

Rules:
- Only use element_idx values that appear in the list.
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
    lines += ["", "ELEMENTS:"]
    lines += [f"  [{el.idx}] {el.label(110)}" for el in candidates]
    lines += [
        "",
        "VISIBLE TEXT (truncated):",
        " ".join(obs.text_digest.split())[:1200],
        "",
        "STEPS SO FAR:",
    ]
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

    idx = data.get("element_idx")
    if isinstance(idx, str):
        idx = int(idx) if idx.strip().lstrip("-").isdigit() else None
    if isinstance(idx, float):
        idx = int(idx)

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5

    return LLMAction(
        op=str(data.get("op") or data.get("operation") or "ask_user").strip().lower(),
        element_idx=idx if isinstance(idx, int) else None,
        text=str(data.get("text") or ""),
        reason=str(data.get("reason") or ""),
        confidence=max(0.0, min(1.0, confidence)),
        raw=text[:2000],
    )


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
