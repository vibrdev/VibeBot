"""Fast decision backends."""

from __future__ import annotations

from ..config import DeciderConfig
from .base import ESCAPE_OPTIONS, Decider, Verdict
from .heuristic import HeuristicDecider
from .jev_decider import JevDecider
from .laya_decider import LayaDecider

__all__ = [
    "ESCAPE_OPTIONS",
    "Decider",
    "HeuristicDecider",
    "JevDecider",
    "LayaDecider",
    "NoDecider",
    "Verdict",
    "build_decider",
]


class NoDecider:
    """No fast decider: the LLM takes every step.

    Loads nothing, costs nothing, and always defers. See DeciderConfig.backend
    for the measurements behind offering it.
    """

    name = "off"

    def __init__(self, cfg: DeciderConfig):
        self.cfg = cfg

    def load(self) -> None:
        return None

    def available(self) -> bool:
        return True

    async def decide(self, goal, obs, candidates, history, text_options) -> Verdict:  # noqa: ANN001
        return Verdict(target="ask_llm", backend="off")


def build_decider(cfg: DeciderConfig):
    """Create the configured decider, falling back to the heuristic one if the
    real brain cannot be loaded (no torch, no network, bad repo id)."""
    backend = (cfg.backend or "laya").lower()
    if backend in {"off", "none", "llm"}:
        return NoDecider(cfg)
    if backend == "heuristic":
        decider = HeuristicDecider(cfg)
    elif backend == "jev":
        decider = JevDecider(cfg)
    else:
        decider = LayaDecider(cfg)

    decider.load()
    if not decider.available():
        reason = getattr(decider, "error", "unknown error")
        fallback = HeuristicDecider(cfg)
        fallback.load()
        fallback.degraded_from = (backend, reason)  # type: ignore[attr-defined]
        return fallback
    return decider
