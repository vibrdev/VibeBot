"""Zero-dependency fallback decider.

Used when Laya is not installed (no torch) or failed to load. It is dumb on
purpose: it trusts the lexical ranker for the obvious cases and escalates
everything else to the LLM, so the agent still works end to end.
"""

from __future__ import annotations

from ..config import DeciderConfig
from ..ranking import score_element, tokens
from ..schema import Element, Observation
from .base import Verdict


class HeuristicDecider:
    name = "heuristic"

    def __init__(self, cfg: DeciderConfig):
        self.cfg = cfg

    def available(self) -> bool:
        return True

    def load(self) -> None:  # parity with LayaDecider
        return None

    @property
    def error(self) -> str | None:
        return None

    async def decide(
        self,
        goal: str,
        obs: Observation,
        candidates: list[Element],
        history: list[str],
        text_options: list[str],
    ) -> Verdict:
        if not candidates:
            return Verdict(target="scroll", target_confidence=0.5, backend="heuristic")

        goal_tokens = tokens(goal)
        history_tokens = tokens(" ".join(history))
        ranked = sorted(
            ((score_element(el, goal_tokens, history_tokens), el) for el in candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
        best_score, best = ranked[0]

        # Softmax the lexical scores so the agent's probability gate works the
        # same way here as it does with a real model behind it.
        probabilities = _softmax({f"e{el.idx}": score for score, el in ranked})
        confidence = probabilities[f"e{best.idx}"]
        if best_score < 1.0:
            # Nothing actually matched the goal — say so instead of bluffing.
            probabilities = {key: value * 0.4 for key, value in probabilities.items()}
            probabilities["ask_llm"] = 0.6
            confidence = probabilities[f"e{best.idx}"]

        is_field = best.tag in {"input", "textarea"} or best.role in {"searchbox", "textbox", "combobox"}
        return Verdict(
            target=f"e{best.idx}",
            target_confidence=confidence,
            probabilities=probabilities,
            operation="type" if is_field else "click",
            operation_confidence=confidence,
            text=text_options[0] if (is_field and text_options) else "",
            done_p=0.0,
            risky_p=_risk(best),
            backend="heuristic",
        )


def _softmax(scores: dict[str, float], temperature: float = 0.8) -> dict[str, float]:
    import math

    if not scores:
        return {}
    top = max(scores.values())
    exponentiated = {key: math.exp((value - top) / temperature) for key, value in scores.items()}
    total = sum(exponentiated.values()) or 1.0
    return {key: value / total for key, value in exponentiated.items()}


def _risk(el: Element) -> float:
    blob = " ".join([el.text, el.name, el.value, el.href]).lower()
    hot = ("buy", "pay", "checkout", "order", "delete", "send", "publish", "confirm", "subscribe")
    return 0.8 if any(word in blob for word in hot) else 0.05
