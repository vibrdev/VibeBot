"""Local LLM via Ollama — the free default. Nothing leaves your machine."""

from __future__ import annotations

import logging

import httpx

from ..config import LLMConfig
from ..schema import Element, Observation
from .base import LLMAction, SYSTEM_PROMPT, b64, parse_action, render_prompt

log = logging.getLogger(__name__)


class OllamaLLM:
    name = "ollama"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = httpx.AsyncClient(base_url=cfg.base_url.rstrip("/"), timeout=cfg.timeout_s)
        self._checked = False
        self._ok = False
        #: None until we learn whether this server accepts the `think` field.
        self._think_ok: bool | None = None

    async def ping(self) -> tuple[bool, str]:
        """Check the daemon is up and the model is pulled, with a useful message."""
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            self._checked, self._ok = True, False
            return False, (
                f"Ollama not reachable at {self.cfg.base_url} ({exc}). "
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
            content = await self._chat(payload)
        except Exception as exc:  # noqa: BLE001
            log.warning("ollama call failed: %s", exc)
            return LLMAction(op="ask_user", text=f"My local LLM failed ({exc}). What should I do next?")
        return parse_action(content)

    async def ask(self, prompt: str) -> str:
        """Free-form question — used for final summaries and done-verification."""
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.cfg.temperature},
        }
        try:
            return (await self._chat(payload)).strip()
        except Exception as exc:  # noqa: BLE001
            return f"(summary unavailable: {exc})"

    async def _chat(self, payload: dict) -> str:
        """POST /api/chat and return the model's answer text.

        Handles two thinking-model traps. First, servers that reject the
        `think` field are retried once without it, and we remember. Second, a
        model that thinks anyway leaves `content` empty and puts everything in
        `thinking`, so we fall back to that rather than reporting a blank reply
        — parse_action digs the JSON out either way.
        """
        think = {"off": False, "on": True}.get(self.cfg.think)
        send_think = think is not None and self._think_ok is not False

        body = {**payload, "think": think} if send_think else payload
        try:
            response = await self._client.post("/api/chat", json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError:
            if not send_think:
                raise
            log.info("server rejected the 'think' field; retrying without it")
            self._think_ok = False
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        else:
            if send_think:
                self._think_ok = True

        message = response.json().get("message", {}) or {}
        content = (message.get("content") or "").strip()
        return content or (message.get("thinking") or "")

    async def close(self) -> None:
        await self._client.aclose()


def _tagged(name: str) -> str:
    """Ollama treats a bare name as ':latest'; compare like for like."""
    return name if ":" in name else f"{name}:latest"
