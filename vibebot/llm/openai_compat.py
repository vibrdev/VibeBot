"""Any OpenAI-compatible chat endpoint.

Covers the paid path you said you might want later (OpenAI, OpenRouter,
Together, Groq…) and also local servers that speak the same protocol
(LM Studio, vLLM, llama.cpp's server). Costs money only if the URL does.
"""

from __future__ import annotations

import logging
import os

import httpx

from ..config import LLMConfig
from ..schema import Element, Observation
from .base import LLMAction, SYSTEM_PROMPT, b64, parse_action, render_prompt

log = logging.getLogger(__name__)


class OpenAICompatLLM:
    name = "openai"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        base = cfg.base_url.rstrip("/") or "https://api.openai.com"
        if not base.endswith("/v1"):
            base += "/v1"
        key = os.environ.get(cfg.api_key_env, "")
        headers = {"content-type": "application/json"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(base_url=base, headers=headers, timeout=cfg.timeout_s)
        self._key_present = bool(key)

    async def ping(self) -> tuple[bool, str]:
        if not self._key_present and "api.openai.com" in str(self._client.base_url):
            return False, f"No API key in ${self.cfg.api_key_env}"
        return True, f"OpenAI-compatible endpoint ({self.cfg.model})"

    def available(self) -> bool:
        return True

    async def decide(
        self,
        goal: str,
        obs: Observation,
        candidates: list[Element],
        history: list[str],
        note: str,
        screenshot_png: bytes | None,
    ) -> LLMAction:
        prompt = render_prompt(goal, obs, candidates, history, note)
        if self.cfg.vision and screenshot_png:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(screenshot_png)}"}},
            ]
        else:
            content = prompt

        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            log.warning("openai-compatible call failed: %s", exc)
            return LLMAction(op="ask_user", text=f"My LLM call failed ({exc}). What should I do next?")
        return parse_action(text)

    async def ask(self, prompt: str) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature,
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            return f"(summary unavailable: {exc})"

    async def close(self) -> None:
        await self._client.aclose()
