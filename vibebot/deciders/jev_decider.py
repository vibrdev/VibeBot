"""Hosted decision-engine backend (Jev or any compatible service).

You asked for this to stay switchable. It is: set `decider.backend: jev` and a
base URL, and the rest of the agent does not care.

Honest caveat — I could not verify Jev's wire format against live docs, so this
speaks the same shape Laya uses locally:

    POST {base_url}/predict
    {"state": {...}, "questions": {...}}
    -> {"answers": {"<name>": {"choice": "e3", "confidence": 0.91, ...}}}

If the real endpoint differs, `_request` and `_parse` are the only two places
that need touching.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from ..config import DeciderConfig
from ..schema import Element, Observation
from .base import Verdict, build_state, element_criteria, pick
from .laya_decider import _OPERATIONS, _as_float, _probabilities

log = logging.getLogger(__name__)


class JevDecider:
    name = "jev"

    def __init__(self, cfg: DeciderConfig):
        self.cfg = cfg
        self._client: httpx.AsyncClient | None = None
        self._error: str | None = None
        if not cfg.jev_base_url:
            self._error = "decider.jev_base_url is not set"

    def available(self) -> bool:
        return self._error is None

    def load(self) -> None:
        if self._error:
            log.warning("Jev backend disabled: %s", self._error)
            return
        headers = {"content-type": "application/json"}
        key = os.environ.get(self.cfg.jev_api_key_env, "")
        if key:
            headers["authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(
            base_url=self.cfg.jev_base_url.rstrip("/"),
            headers=headers,
            timeout=30.0,
        )

    @property
    def error(self) -> str | None:
        return self._error

    async def decide(
        self,
        goal: str,
        obs: Observation,
        candidates: list[Element],
        history: list[str],
        text_options: list[str],
    ) -> Verdict:
        if self._client is None:
            return Verdict(target="ask_llm", backend="jev-unavailable")

        payload = {
            "state": build_state(goal, obs, history, self.cfg.page_text_chars),
            "questions": {
                "target": {
                    "type": "choice",
                    "instructions": "Which element should be acted on next?",
                    "criteria": element_criteria(candidates),
                },
                "operation": {
                    "type": "choice",
                    "instructions": "What should be done to that element?",
                    "criteria": dict(_OPERATIONS),
                },
                "done": {"type": "noul", "instructions": "Has the goal been achieved?"},
                "risky": {
                    "type": "noul",
                    "instructions": "Would this step spend money, send, publish, delete or log in?",
                },
                **(
                    {
                        "type_text": {
                            "type": "choice",
                            "instructions": "Which text should be entered?",
                            "criteria": {opt: f"enter: {opt}" for opt in text_options},
                        }
                    }
                    if text_options
                    else {}
                ),
            },
        }

        started = time.perf_counter()
        try:
            result = await self._request(payload)
        except Exception as exc:  # noqa: BLE001 - a remote brain must never be fatal
            log.warning("Jev request failed: %s", exc)
            return Verdict(target="ask_llm", backend="jev-error", raw={"error": str(exc)})

        verdict = self._parse(result)
        verdict.latency_ms = int((time.perf_counter() - started) * 1000)
        return verdict

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self._client is not None
        response = await self._client.post("/predict", json=payload)
        response.raise_for_status()
        return response.json()

    def _parse(self, result: dict[str, Any]) -> Verdict:
        answers = result.get("answers", result)
        operation = str(pick(answers, "operation", "choice", "label", default="click"))
        return Verdict(
            target=str(pick(answers, "target", "choice", "label", default="ask_llm")),
            target_confidence=_as_float(pick(answers, "target", "confidence", default=0.0)),
            probabilities=_probabilities(answers, "target"),
            operation=operation if operation in _OPERATIONS else "click",
            operation_confidence=_as_float(pick(answers, "operation", "confidence", default=0.0)),
            text=str(pick(answers, "type_text", "choice", "label", default="") or ""),
            done_p=_as_float(pick(answers, "done", "noul", "probability", "p_true", default=0.0)),
            risky_p=_as_float(pick(answers, "risky", "noul", "probability", "p_true", default=0.0)),
            backend="jev",
        )
