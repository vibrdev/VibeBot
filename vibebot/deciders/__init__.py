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
    "Verdict",
    "build_decider",
]


def build_decider(cfg: DeciderConfig):
    """Create the configured decider, falling back to the heuristic one if the
    real brain cannot be loaded (no torch, no network, bad repo id)."""
    backend = (cfg.backend or "laya").lower()
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
