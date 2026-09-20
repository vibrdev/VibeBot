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
        names = {m.get("name", "") for m in response.json().get("models", [])}
        base = {n.split(":")[0] for n in names}
        wanted = self.cfg.model
        if wanted not in names and wanted.split(":")[0] not in base:
            self._checked, self._ok = True, False
            return False, f"Model '{wanted}' is not pulled. Run: ollama pull {wanted}"
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
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
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
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "").strip()
        except Exception as exc:  # noqa: BLE001
            return f"(summary unavailable: {exc})"

    async def close(self) -> None:
        await self._client.aclose()
