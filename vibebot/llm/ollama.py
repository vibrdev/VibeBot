"""Local LLM via Ollama — the free default. Nothing leaves your machine."""

from __future__ import annotations

import logging

import httpx

from ..config import LLMConfig
from ..schema import Element, Observation
from .base import (
    LLMAction,
    LLMHTTPError,
    SYSTEM_PROMPT,
    b64,
    parse_action,
    raise_for_status,
    render_prompt,
)

log = logging.getLogger(__name__)

#: Used when `llm.base_url` is empty. Each backend brings its own default so
#: that changing `llm.backend` alone cannot leave you pointed at the other
#: backend's port.
DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaLLM:
    name = "ollama"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.base_url = (cfg.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=cfg.timeout_s)
        self._checked = False
        self._ok = False
        #: None until we learn whether this server accepts the `think` field.
        self._think_ok: bool | None = None

    async def ping(self) -> tuple[bool, str]:
        """Check the daemon is up and the model is pulled, with a useful message."""
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            raise_for_status(response)
        except Exception as exc:  # noqa: BLE001
            self._checked, self._ok = True, False
            return False, (
                f"Ollama not reachable at {self.base_url} ({exc}). "
                "Install from https://ollama.com and start it."
            )
        pulled = {_tagged(m.get("name", "")) for m in response.json().get("models", [])}
        wanted = self.cfg.model
        if _tagged(wanted) not in pulled:
            # Matching on the family name alone would pass ':70b' just because
            # ':8b' is present, and the run would then die mid-goal. Ollama
            # itself resolves tags exactly, so we do too.
            family = wanted.split(":")[0]
            near = sorted(n for n in pulled if n.split(":")[0] == family)
            hint = f" (you have: {', '.join(near)})" if near else ""
            self._checked, self._ok = True, False
            return False, f"Model '{wanted}' is not pulled{hint}. Run: ollama pull {wanted}"
        self._checked, self._ok = True, True
        return True, f"Ollama ready ({wanted})"

    def available(self) -> bool:
        return self._ok or not self._checked

    async def decide(
        self,
        goal: str,
        obs: Observation,
        candidates: list[Element],
        history: list[str],
        note: str,
        screenshot_png: bytes | None,
    ) -> LLMAction:
        message = {"role": "user", "content": render_prompt(goal, obs, candidates, history, note)}
        if self.cfg.vision and screenshot_png:
            message["images"] = [b64(screenshot_png)]

        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, message],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.cfg.temperature, "num_predict": self.cfg.max_tokens},
        }
        try:
            content, done_reason = await self._chat(payload)
        except Exception as exc:  # noqa: BLE001
            log.warning("ollama call failed: %s", exc)
            return LLMAction(
                op="ask_user",
                text=f"My local LLM failed ({exc}).{_next_step(str(exc))} What should I do next?",
            )
        return _finish(parse_action(content), done_reason, self.cfg)

    async def ask(self, prompt: str) -> str:
        """Free-form question — used for final summaries and done-verification."""
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
        }
        try:
            content, _ = await self._chat(payload)
            return content.strip()
        except Exception as exc:  # noqa: BLE001
            return f"(summary unavailable: {exc})"

    async def _chat(self, payload: dict) -> tuple[str, str]:
        """POST /api/chat and return (answer text, done_reason).

        done_reason matters: "length" means the token budget cut the reply off,
        which is not the same failure as a model that answered badly, and only
        one of the two is fixed by raising llm.max_tokens.

        Handles two thinking-model traps. First, a server that answers 400
        because it will not take the `think` field is retried once without it,
        and we remember — but only on a 400, so an unrelated failure cannot
        turn thinking back on behind our back. Second, a model that thinks
        anyway leaves `content` empty and puts everything in `thinking`, so we
        fall back to that rather than reporting a blank reply — parse_action
        digs the JSON out either way.
        """
        think = {"off": False, "on": True}.get(self.cfg.think)
        send_think = think is not None and self._think_ok is not False

        body = {**payload, "think": think} if send_think else payload
        try:
            response = await self._client.post("/api/chat", json=body)
            raise_for_status(response)
        except LLMHTTPError as exc:
            # Only 400 means "this server will not take `think`". Measured on
            # Ollama 0.34.2: `think: true` against a model without the
            # capability answers 400 '"gemma3:1b" does not support thinking',
            # while `think: false` answers 200. A 500 is the server failing to
            # serve at all - most often the model not fitting in memory - so
            # retrying without `think` cannot fix it, and remembering a
            # rejection that never happened would silently turn thinking back
            # on for the rest of the run and eat the token budget.
            if not send_think or exc.status_code != 400:
                raise
            log.info("server rejected the 'think' field; retrying without it")
            self._think_ok = False
            response = await self._client.post("/api/chat", json=payload)
            raise_for_status(response)
        else:
            if send_think:
                self._think_ok = True

        data = response.json()
        message = data.get("message", {}) or {}
        content = (message.get("content") or "").strip()
        return content or (message.get("thinking") or ""), str(data.get("done_reason") or "")

    async def close(self) -> None:
        await self._client.aclose()


def _finish(action: LLMAction, done_reason: str, cfg: LLMConfig) -> LLMAction:
    """Say so when the token budget, not the model, is what went wrong.

    Measured on qwen3.5:4b: at num_predict=60 the reply stops one character
    short of valid JSON (parse_action repairs that), and with think: on the
    reasoning can eat the whole budget so `content` is empty and only prose
    comes back (nothing to repair). Both used to surface as the same shrug.
    """
    if done_reason != "length":
        return action
    if action.op == "ask_user" and action.element_idx is None:
        action.text = (
            f"My reply hit the {cfg.max_tokens}-token limit (llm.max_tokens) before it was "
            "valid JSON, so I have no action to take. Raise llm.max_tokens"
            + (", or set llm.think: off so reasoning stops eating the budget" if cfg.think != "off" else "")
            + ". What should I do next?"
        )
    else:
        log.warning(
            "reply truncated at llm.max_tokens=%s; recovered op=%r. Raise it if this repeats.",
            cfg.max_tokens, action.op,
        )
    return action


def _next_step(message: str) -> str:
    """Name the fix when the server's reason is one we recognise.

    Measured on this repo's default setup - qwen3.5:4b, 4 GB RTX 3050 Ti, 16 GB
    RAM - with the browser window open and Laya resident on the CPU, Ollama had
    ~1 GB of host RAM free and could not pin the ~1.8 GB of model that does not
    fit in VRAM. /api/chat answered 500 for every step of the run; the same
    request succeeded once the browser was closed.
    """
    low = message.lower()
    if "out of memory" in low or "allocate" in low:
        return (
            " Ollama ran out of memory loading the model, so this is about the machine, not the"
            " page: close other apps (the browser window this run opened is usually the biggest),"
            " or set llm.vision: false, or drop to a smaller llm.model such as qwen3.5:2b."
        )
    return ""


def _tagged(name: str) -> str:
    """Ollama treats a bare name as ':latest'; compare like for like."""
    return name if ":" in name else f"{name}:latest"
