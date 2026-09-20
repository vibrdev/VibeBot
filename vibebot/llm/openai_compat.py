"""Any OpenAI-compatible chat endpoint.

Covers the paid path you said you might want later (OpenAI, OpenRouter,
Together, Groq...) and also local servers that speak the same protocol
(LM Studio, vLLM, llama.cpp's server). Costs money only if the URL does.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

import httpx

from ..config import LLMConfig
from ..schema import Element, Observation
from .base import LLMAction, SYSTEM_PROMPT, b64, parse_action, render_prompt

log = logging.getLogger(__name__)

#: Used when `llm.base_url` is empty. Each backend brings its own default, so
#: switching `llm.backend` without touching `base_url` lands somewhere sane
#: instead of on the other backend's port.
DEFAULT_BASE_URL = "https://api.openai.com"

LOOPBACK = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

#: Ollama's port. `backend: openai` pointed here is a half-finished switch,
#: not a deployment - Ollama does serve /v1/chat/completions, so the call
#: succeeds just long enough to be confusing.
OLLAMA_PORT = 11434


class OpenAICompatLLM:
    name = "openai"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        base = (cfg.base_url or DEFAULT_BASE_URL).rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.base_url = base
        parsed = urlparse(base)
        self.host = parsed.hostname or ""
        self.port = parsed.port
        self.is_local = self.host in LOOPBACK
        key = os.environ.get(cfg.api_key_env, "")
        headers = {"content-type": "application/json"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(base_url=base, headers=headers, timeout=cfg.timeout_s)
        self._key_present = bool(key)
        self._checked = False
        self._ok = False

    async def ping(self) -> tuple[bool, str]:
        """Check the endpoint is usable, the way the Ollama backend does.

        The old check only looked for a missing key on api.openai.com, so every
        other hosted provider - and the "I changed backend but not base_url"
        case - reported ready and then failed mid-goal.
        """
        if self.is_local and self.port == OLLAMA_PORT:
            return self._verdict(
                False,
                f"llm.backend is '{self.cfg.backend}' but llm.base_url points at Ollama "
                f"({self.base_url}). Set llm.base_url to your provider, or use llm.backend: ollama.",
            )
        if not self._key_present and not self.is_local:
            return self._verdict(
                False,
                f"No API key in ${self.cfg.api_key_env} - {self.host} will reject these calls.",
            )

        try:
            response = await self._client.get("/models", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403}:
                return self._verdict(False, f"{self.host} rejected the key in ${self.cfg.api_key_env} ({status}).")
            # Plenty of local servers implement /chat/completions and nothing
            # else. Not being able to list models is not being broken.
            return self._verdict(True, f"{self.host} reachable; model list unavailable ({status}), not verified")
        except Exception as exc:  # noqa: BLE001
            return self._verdict(False, f"{self.base_url} not reachable ({exc}).")

        try:
            served = {str(m.get("id", "")) for m in response.json().get("data", [])}
        except Exception:  # noqa: BLE001
            served = set()
        if served and self.cfg.model not in served:
            near = sorted(n for n in served if n.startswith(self.cfg.model.split("-")[0]))[:5]
            hint = f" (close matches: {', '.join(near)})" if near else ""
            return self._verdict(False, f"Model '{self.cfg.model}' is not served by {self.host}{hint}.")
        return self._verdict(True, f"{self.host} ready ({self.cfg.model})")

    def _verdict(self, ok: bool, message: str) -> tuple[bool, str]:
        self._checked, self._ok = True, ok
        return ok, message

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
