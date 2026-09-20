"""Reasoning backends."""

from __future__ import annotations

from ..config import LLM_BACKENDS, LLMConfig
from .base import LLM, LLMAction, parse_action
from .ollama import OllamaLLM
from .openai_compat import OpenAICompatLLM

__all__ = [
    "LLM",
    "LLMAction",
    "NullLLM",
    "OllamaLLM",
    "OpenAICompatLLM",
    "build_llm",
    "parse_action",
]

class NullLLM:
    """No reasoning layer - every escalation goes straight to the human."""

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
    """Build the backend named in `llm.backend`.

    Raises ValueError on an unknown name. The previous version treated
    anything unrecognised as "ollama", so `backend: anthropic` - or a plain
    typo - silently ran every escalation against localhost while the config
    said otherwise. A wrong route that still works is the expensive kind.
    """
    name = (cfg.backend or "ollama").strip().lower()
    kind = LLM_BACKENDS.get(name)
    if kind is None:
        valid = ", ".join(sorted(set(LLM_BACKENDS)))
        raise ValueError(f"llm.backend: unknown value {cfg.backend!r}. Valid values: {valid}")
    if kind == "none":
        return NullLLM()
    if kind == "openai":
        return OpenAICompatLLM(cfg)
    return OllamaLLM(cfg)
