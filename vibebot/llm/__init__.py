"""Reasoning backends."""

from __future__ import annotations

from ..config import LLMConfig
from .base import LLM, LLMAction, parse_action
from .ollama import OllamaLLM
from .openai_compat import OpenAICompatLLM

__all__ = ["LLM", "LLMAction", "NullLLM", "OllamaLLM", "OpenAICompatLLM", "build_llm", "parse_action"]


class NullLLM:
    """No reasoning layer — every escalation goes straight to the human."""

    name = "none"

    def available(self) -> bool:
        return False

    async def ping(self) -> tuple[bool, str]:
        return False, "LLM disabled (llm.backend: none)"

    async def decide(self, goal, obs, candidates, history, note, screenshot_png) -> LLMAction:  # noqa: ANN001
        return LLMAction(op="ask_user", text=f"No LLM configured. {note or 'What should I do next?'}")

    async def ask(self, prompt: str) -> str:
        return ""

    async def close(self) -> None:
        return None


def build_llm(cfg: LLMConfig):
    backend = (cfg.backend or "ollama").lower()
    if backend in {"none", "off", "disabled"}:
        return NullLLM()
    if backend in {"openai", "openai_compat", "openrouter", "lmstudio", "vllm"}:
        return OpenAICompatLLM(cfg)
    return OllamaLLM(cfg)
