"""Shared data types passed between perception, deciders and the browser."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Op = Literal[
    "click", "type", "select", "scroll", "navigate", "back",
    "wait", "extract", "ask_user", "done", "fail",
]

#: Ops that do not need a target element.
GLOBAL_OPS = {"scroll", "navigate", "back", "wait", "extract", "ask_user", "done", "fail"}

#: Ops that end the run.
TERMINAL_OPS = {"done", "extract", "fail"}


@dataclass
class Element:
    """One interactive thing on the page."""

    idx: int
    tag: str
    role: str = ""
    text: str = ""
    name: str = ""          # aria-label / title / associated label
    placeholder: str = ""
    value: str = ""
    href: str = ""
    input_type: str = ""
    rect: tuple[float, float, float, float] = (0, 0, 0, 0)
    in_viewport: bool = True

    def label(self, limit: int = 90) -> str:
        """Short human/model readable description. Kept tight on purpose: Laya has
        a 512-1024 token context and pays for every wasted word."""
        parts = [self.role or self.tag]
        body = self.text or self.name or self.placeholder or self.value or self.href
        if body:
            parts.append(f'"{_squash(body, limit)}"')
        if self.input_type and self.input_type not in {"text", "submit"}:
            parts.append(f"[{self.input_type}]")
        if not self.in_viewport:
            parts.append("(offscreen)")
        return " ".join(parts)


@dataclass
class Observation:
    """A snapshot of the page the agent is looking at."""

    url: str
    title: str
    elements: list[Element]
    text_digest: str = ""
    screenshot_png: bytes | None = None
    annotated_png: bytes | None = None
    step: int = 0

    def fingerprint(self) -> str:
        import hashlib

        blob = self.url + "|" + "|".join(e.label(40) for e in self.elements[:40])
        return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:16]


@dataclass
class Action:
    op: Op
    element_idx: int | None = None
    text: str = ""
    """Text to type, option to select, URL to navigate to, question to ask."""
    reason: str = ""
    source: str = "laya"
    """laya | llm | user | policy"""
    confidence: float = 0.0
    risky: float = 0.0

    def describe(self, obs: Observation | None = None) -> str:
        target = ""
        if self.element_idx is not None and obs is not None:
            match = next((e for e in obs.elements if e.idx == self.element_idx), None)
            target = f" -> [{self.element_idx}] {match.label(60)}" if match else f" -> [{self.element_idx}]"
        extra = f' "{_squash(self.text, 60)}"' if self.text else ""
        return f"{self.op}{target}{extra}"


@dataclass
class StepRecord:
    """One row of the trace log — also the training data for fine-tuning Laya later."""

    step: int
    url: str
    title: str
    goal: str
    candidates: list[str]
    laya: dict[str, Any] | None
    action: dict[str, Any]
    outcome: str
    escalated_to: str | None = None
    duration_ms: int = 0
    ts: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _squash(text: str, limit: int) -> str:
    clean = " ".join((text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"
