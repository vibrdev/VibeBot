"""Laya: local, free, calibrated probabilities, no tokens generated.

Measured on an i7-11370H, CPU only, four typed questions in one batch, with
the weights already warm. The earlier figures in this docstring (~0.3s short,
1.5-2.5s at the default digest) were optimistic by roughly a factor of three:

    page_text_chars    0    3.70s
                     100    4.10s
                     200    4.48s
                     400    5.03s
                     600    5.84s   <- the default
                    1000    7.10s

So a step costs about 3.7s before it reads a single character of the page, and
another 3.4s per thousand characters after that. The candidate list is nearly
free by comparison: 4 options 5.11s, 20 options 5.84s, which is why the
pre-ranking limit can be generous and the text budget cannot.

The advertised 33ms is a single question on a T4 GPU. Even at 5.8s this is
still several times cheaper than waking the vision model, which takes 20-30s.

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
import threading
import time
from typing import Any

from ..config import DeciderConfig
from ..schema import Element, Observation
from ..sysmem import snapshot
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
        #: The weights load on the first predict when preload is off, so the
        #: memory guard has to hold until that has actually happened.
        self._warm = False
        self._failures = 0
        #: predict() is a torch forward pass; one at a time. The background
        #: warm-up and the first real step must not race into it together.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup

    def available(self) -> bool:
        return self._error is None

    def load(self) -> None:
        """Import the library and prepare the model. Slow (seconds) either way."""
        try:
            import laya  # noqa: PLC0415 - optional heavy dependency
        except ImportError as exc:
            self._error = f"laya is not installed ({exc}). Run: pip install laya"
            log.warning(self._error)
            return

        shortfall = self._memory_shortfall()
        if shortfall:
            # Degrade on purpose rather than die by accident: torch treats a
            # declined commit charge as fatal, so the process would segfault
            # here with no traceback to explain it.
            self._error = shortfall
            log.warning(self._error)
            return

        try:
            if self.cfg.model in ("auto", "router", ""):
                self._agent = laya.Router(preload=self.cfg.preload, device=self.cfg.device)
            else:
                self._agent = laya.load(self.cfg.model, device=self.cfg.device)
            self._warm = self.cfg.preload or self.cfg.model not in ("auto", "router", "")
            log.info(
                "Laya ready (%s, device=%s, %s)",
                self.cfg.model,
                self.cfg.device,
                "preloaded" if self._warm else "weights load on first step",
            )
        except Exception as exc:  # noqa: BLE001 - model load can fail many ways
            self._error = f"could not load Laya: {exc}"
            log.warning(self._error)

    def _memory_shortfall(self) -> str | None:
        """Why we should not touch the weights right now, if we should not."""
        memory = snapshot()
        if memory is None or self.cfg.min_headroom_mb <= 0:
            return None
        if memory.headroom_mb >= self.cfg.min_headroom_mb:
            return None
        return (
            f"only {memory.headroom_mb:,.0f} MB of memory can still be committed "
            f"({memory.committed_mb:,.0f} of {memory.limit_mb:,.0f} MB in use; "
            f"{memory.available_mb:,.0f} MB RAM free), below the "
            f"{self.cfg.min_headroom_mb:,} MB decider.min_headroom_mb floor. "
            "Free memory (an idle Ollama model holds ~5.7 GB; a WSL/Docker VM is "
            "often several GB more) or lower the floor to try anyway."
        )

    def warm(self) -> None:
        """Start loading the weights now, in the background.

        Lazy loading moved a ~35s load off startup and onto the first step,
        where it is the only thing happening and you watch all of it. Chromium
        starting and the first LLM call together take longer than the load, so
        run it alongside them instead and the wait disappears.
        """
        if self._agent is None or self._warm or self._error:
            return

        def load() -> None:
            with self._lock:
                if self._warm:
                    return
                if self._memory_shortfall():
                    return  # decide() will report it properly on the first step
                started = time.perf_counter()
                try:
                    self._agent.predict(
                        {"goal": "warm up", "page_text": ""},
                        {"warm": {"type": "noul", "instructions": "ready?"}},
                    )
                except Exception as exc:  # noqa: BLE001 - a failed warm is not fatal
                    log.debug("Laya warm-up failed, first step will pay for it: %s", exc)
                    return
                self._warm = True
                log.info("Laya warm after %.1fs", time.perf_counter() - started)

        threading.Thread(target=load, name="laya-warm", daemon=True).start()

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

        if not self._warm:
            # First real step with lazy weights: the same commit charge that
            # kills the process at load time is charged here instead.
            shortfall = self._memory_shortfall()
            if shortfall:
                self._failures += 1
                log.warning("Laya skipped, every step is going to the LLM: %s", shortfall)
                return Verdict(
                    target="ask_llm",
                    backend="laya-low-memory",
                    raw={"error": shortfall, "consecutive_failures": self._failures},
                )

        state = build_state(goal, obs, history, self.cfg.page_text_chars)
        questions = self._questions(candidates, text_options, len(obs.tabs))

        def predict() -> Any:
            with self._lock:  # never two forward passes at once
                return self._agent.predict(state, questions)

        started = time.perf_counter()
        try:
            # torch is blocking; keep the event loop (and the live UI) responsive.
            result = await asyncio.to_thread(predict)
        except Exception as exc:  # noqa: BLE001 - never let the brain kill the run
            self._failures += 1
            memory = snapshot()
            # A silent fall through to the LLM turns a broken decider into a
            # slow, expensive run that still looks like it is working. Say it.
            log.warning(
                "Laya predict failed (%s in a row), so this step goes to the LLM: %s%s",
                self._failures,
                exc,
                f" [{memory.short()}]" if memory else "",
            )
            return Verdict(
                target="ask_llm",
                backend="laya-error",
                raw={
                    "error": str(exc),
                    "consecutive_failures": self._failures,
                    "memory": memory.short() if memory else None,
                },
            )
        latency = int((time.perf_counter() - started) * 1000)
        self._warm = True
        self._failures = 0

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
