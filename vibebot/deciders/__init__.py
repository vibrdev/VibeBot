"""Fast decision backends."""

from __future__ import annotations

import re

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
    "LabelMatchDecider",
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


class LabelMatchDecider:
    """The simplest possible executor for decider.mode: plan - a control.

    It picks the element whose visible label is the text quoted in the plan
    step, or whose whole label appears in the step, if exactly one does, and
    otherwise defers. No model. It exists to answer one question honestly:
    does a learned fast decider carry out plan steps better than string
    matching does? If not, it is adding latency, not judgement.
    """

    name = "match"

    def __init__(self, cfg: DeciderConfig):
        self.cfg = cfg

    def load(self) -> None:
        return None

    def available(self) -> bool:
        return True

    async def decide(self, goal, obs, candidates, history, text_options) -> Verdict:  # noqa: ANN001
        def label(el) -> str:  # noqa: ANN001
            return " ".join((el.text or el.name or el.placeholder or "").lower().split())

        wanted = [" ".join(q.lower().split()) for q in text_options if q.strip()]
        step = " ".join((goal or "").lower().split())
        # The planner often names the element by its number, "click [18] 96";
        # a number that exists on the page is the least ambiguous name there is.
        numbered = [int(n) for n in re.findall(r"\[(\d+)\]", goal or "")]
        by_number = [el for el in (getattr(obs, "elements", None) or candidates) if el.idx in numbered]
        if len(by_number) == 1:
            el = by_number[0]
            return Verdict(target=f"e{el.idx}", probabilities={f"e{el.idx}": 1.0, "ask_llm": 0.0},
                           backend="match")
        hits = [el for el in candidates if label(el) and label(el) in wanted]
        if not hits:
            hits = [el for el in candidates if len(label(el)) >= 3 and label(el) in step]
        if len(hits) != 1:
            return Verdict(target="ask_llm", probabilities={"ask_llm": 1.0}, backend="match")
        return Verdict(
            target=f"e{hits[0].idx}", probabilities={f"e{hits[0].idx}": 1.0, "ask_llm": 0.0}, backend="match"
        )


def build_decider(cfg: DeciderConfig):
    """Create the configured decider, falling back to the heuristic one if the
    real brain cannot be loaded (no torch, no network, bad repo id)."""
    backend = (cfg.backend or "laya").lower()
    if backend in {"off", "none", "llm"}:
        return NoDecider(cfg)
    if backend == "match":
        return LabelMatchDecider(cfg)
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
