"""Laya: local, free, calibrated probabilities, no tokens generated.

Measured, not quoted from the README: one batched step costs ~0.3s on a laptop
CPU with a short page digest and ~1.5-2.5s with the default 600-character one.
The advertised 33ms is a single question on a T4 GPU. Either way it is an order
of magnitude cheaper than waking a 7B vision model.

We ask it four typed questions per step in a single forward pass:

  target     choice  which element (or escape hatch) to act on
  operation  choice  click / type / select
  type_text  choice  what to type, when the target is a text field
  done       noul    is the goal already satisfied?
  risky      noul    would this step spend money / send / delete / log in?

The `ask_llm` option inside `target` is the whole point of the design: Laya
itself decides when a step is above its pay grade.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..config import DeciderConfig
from ..schema import Element, Observation
from .base import Verdict, build_state, element_criteria, pick

log = logging.getLogger(__name__)

_OPERATIONS = {
    "click": "press, tap, follow or toggle the element",
    "type": "enter text into a field, then submit",
    "select": "choose an option from a dropdown",
}


class LayaDecider:
    name = "laya"

    def __init__(self, cfg: DeciderConfig):
        self.cfg = cfg
        self._agent: Any = None
        self._error: str | None = None

    # ------------------------------------------------------------------ setup

    def available(self) -> bool:
        return self._error is None

    def load(self) -> None:
        """Import and warm the model. Slow (seconds), so do it once up front."""
        try:
            import laya  # noqa: PLC0415 - optional heavy dependency
        except ImportError as exc:
            self._error = f"laya is not installed ({exc}). Run: pip install laya"
            log.warning(self._error)
            return

        try:
            if self.cfg.model in ("auto", "router", ""):
                self._agent = laya.Router(preload=True, device=self.cfg.device)
            else:
                self._agent = laya.load(self.cfg.model, device=self.cfg.device)
            log.info("Laya ready (%s, device=%s)", self.cfg.model, self.cfg.device)
        except Exception as exc:  # noqa: BLE001 - model load can fail many ways
            self._error = f"could not load Laya: {exc}"
            log.warning(self._error)

    @property
    def error(self) -> str | None:
        return self._error

    # --------------------------------------------------------------- decision

    async def decide(
        self,
        goal: str,
        obs: Observation,
        candidates: list[Element],
        history: list[str],
        text_options: list[str],
    ) -> Verdict:
        if self._agent is None:
            return Verdict(target="ask_llm", backend="laya-unavailable")

        state = build_state(goal, obs, history, self.cfg.page_text_chars)
        questions = self._questions(candidates, text_options, len(obs.tabs))

        started = time.perf_counter()
        try:
            # torch is blocking; keep the event loop (and the live UI) responsive.
            result = await asyncio.to_thread(self._agent.predict, state, questions)
        except Exception as exc:  # noqa: BLE001 - never let the brain kill the run
            log.warning("Laya predict failed: %s", exc)
            return Verdict(target="ask_llm", backend="laya-error", raw={"error": str(exc)})
        latency = int((time.perf_counter() - started) * 1000)

        answers = result.get("answers", result) if isinstance(result, dict) else {}
        target = str(pick(answers, "target", "choice", "label", default="ask_llm"))
        operation = str(pick(answers, "operation", "choice", "label", default="click"))
        text = str(pick(answers, "type_text", "choice", "label", default="") or "")

        verdict = Verdict(
            target=target,
            target_confidence=_as_float(pick(answers, "target", "confidence", default=0.0)),
            probabilities=_probabilities(answers, "target"),
            operation=operation if operation in _OPERATIONS else "click",
            operation_confidence=_as_float(pick(answers, "operation", "confidence", default=0.0)),
            text=text,
            text_confidence=_as_float(pick(answers, "type_text", "confidence", default=0.0)),
            done_p=_as_float(pick(answers, "done", "noul", "probability", "p_true", default=0.0)),
            risky_p=_as_float(pick(answers, "risky", "noul", "probability", "p_true", default=0.0)),
            latency_ms=latency,
            backend="laya",
            raw=_trim_raw(result),
        )
        log.debug("laya -> %s (%.2f) op=%s", verdict.target, verdict.target_confidence, verdict.operation)
        return verdict

    # ---------------------------------------------------------------- helpers

    def _questions(
        self, candidates: list[Element], text_options: list[str], tab_count: int = 1
    ) -> dict[str, Any]:
        questions: dict[str, Any] = {
            "target": {
                "type": "choice",
                "instructions": (
                    "Given the goal and the current page, which single element should be "
                    "acted on next? Pick 'ask_llm' if the choice needs real reasoning."
                ),
                "criteria": element_criteria(candidates, tab_count),
            },
            "operation": {
                "type": "choice",
                "instructions": "What should be done to that element?",
                "criteria": dict(_OPERATIONS),
            },
            "done": {
                "type": "noul",
                "instructions": "Has the goal already been fully achieved on this page?",
            },
            "risky": {
                "type": "noul",
                "instructions": (
                    "Would the next step spend money, place an order, send or publish a "
                    "message, delete data, or submit credentials?"
                ),
            },
        }
        if text_options:
            questions["type_text"] = {
                "type": "choice",
                "instructions": "If text must be entered, which of these is the right text?",
                "criteria": {opt: f"enter: {opt}" for opt in text_options},
            }
        return questions


def _probabilities(answers: Any, key: str) -> dict[str, float]:
    raw = pick(answers, key, "probabilities", "probs", default=None)
    if not isinstance(raw, dict):
        return {}
    return {str(k): _as_float(v) for k, v in raw.items()}


def _as_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _trim_raw(result: Any) -> dict[str, Any]:
    """Keep the trace readable: probabilities only, no tensors."""
    if not isinstance(result, dict):
        return {}
    answers = result.get("answers", {})
    out: dict[str, Any] = {"routing": result.get("routing", {})}
    for key, answer in answers.items() if isinstance(answers, dict) else []:
        if isinstance(answer, dict):
            probs = answer.get("probabilities") or {}
            top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5] if isinstance(probs, dict) else []
            out[key] = {k: round(float(v), 4) for k, v in top}
    return out
