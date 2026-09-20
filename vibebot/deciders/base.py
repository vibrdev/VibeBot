"""Decider interface — the fast 'System 1' layer that picks the next action."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..schema import Element, Observation

#: Pseudo-options that always sit alongside the real page elements. This is the
#: "…or let the LLM handle this step" escape hatch, exposed to Laya as just
#: another choice it can take.
ESCAPE_OPTIONS: dict[str, str] = {
    "ask_llm": "the page is confusing, none of the elements clearly fit, or this "
               "step needs reasoning — hand it to the LLM",
    "scroll": "the answer is probably further down the page",
    "back": "this page was a dead end, go back",
    "done": "the goal has already been achieved on this page",
}


@dataclass
class Verdict:
    """What the fast decider thinks, with calibrated probabilities.

    Measured against the real model: Laya's reported ``confidence`` is
    ``1 - normalised entropy`` of the distribution, *not* the top option's
    probability. It therefore sags as you add options — a correct pick at
    p=0.66 out of four reports confidence 0.29. Gate on :attr:`p_top` and
    :attr:`margin` instead, and treat ``confidence`` as a display value.
    """

    target: str = "ask_llm"
    """An element key ("e7") or one of ESCAPE_OPTIONS."""
    target_confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    operation: str = "click"
    operation_confidence: float = 0.0
    text: str = ""
    text_confidence: float = 0.0
    done_p: float = 0.0
    risky_p: float = 0.0
    latency_ms: int = 0
    backend: str = "none"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_llm(self) -> bool:
        return self.target == "ask_llm"

    @property
    def p_top(self) -> float:
        """Probability of the chosen option (falls back to confidence)."""
        if not self.probabilities:
            return self.target_confidence
        return float(self.probabilities.get(self.target, max(self.probabilities.values(), default=0.0)))

    @property
    def margin(self) -> float:
        """Lead over the runner-up. A close race means "no idea"."""
        if len(self.probabilities) < 2:
            return self.p_top
        ordered = sorted(self.probabilities.values(), reverse=True)
        return float(ordered[0] - ordered[1])

    def is_confident(self, min_probability: float, min_margin: float) -> bool:
        return self.p_top >= min_probability and self.margin >= min_margin

    def to_json(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "target_confidence": round(self.target_confidence, 4),
            "p_top": round(self.p_top, 4),
            "margin": round(self.margin, 4),
            "operation": self.operation,
            "operation_confidence": round(self.operation_confidence, 4),
            "text": self.text,
            "done_p": round(self.done_p, 4),
            "risky_p": round(self.risky_p, 4),
            "latency_ms": self.latency_ms,
            "backend": self.backend,
        }


class Decider(Protocol):
    name: str

    def available(self) -> bool: ...

    async def decide(
        self,
        goal: str,
        obs: Observation,
        candidates: list[Element],
        history: list[str],
        text_options: list[str],
    ) -> Verdict: ...


def build_state(goal: str, obs: Observation, history: list[str], text_budget: int = 600) -> dict[str, str]:
    """The 'state' blob Laya conditions on. Deliberately short — context is money."""
    recent = history[-4:]
    state = {
        "goal": goal,
        "page_title": obs.title[:120],
        "url": obs.url[:180],
        "steps_taken": "; ".join(recent) if recent else "none yet",
        "page_text": " ".join(obs.text_digest.split())[:text_budget],
    }
    if len(obs.tabs) > 1:
        state["open_tabs"] = " | ".join(tab.label(40) for tab in obs.tabs[:6])
    return state


def element_criteria(candidates: list[Element], tab_count: int = 1) -> dict[str, str]:
    """Element list in Laya's `choice` criteria format, plus the escape hatches."""
    criteria = {f"e{el.idx}": el.label() for el in candidates}
    criteria.update(ESCAPE_OPTIONS)
    if tab_count > 1:
        criteria["switch_tab"] = "what we need is in one of the other open tabs"
    return criteria


def pick(answers: dict[str, Any], key: str, *names: str, default: Any = None) -> Any:
    """Laya's answer payloads are dicts keyed by primitive ("choice"/"score"/"noul").
    Pull the first field that exists so a minor library change does not break us."""
    answer = answers.get(key)
    if not isinstance(answer, dict):
        return default
    for name in names:
        if name in answer and answer[name] is not None:
            return answer[name]
    return default
