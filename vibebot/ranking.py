"""Candidate pre-filter.

Laya is a 421M encoder with a 512-1024 token window, and its own docs warn that
choices with >50 options need token-budget tuning. Handing it 150 raw DOM nodes
would be malpractice. So we cheaply rank elements first and only let it decide
between the top handful.

Everything here is deterministic string matching — no model, no latency.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .schema import Element

_WORD = re.compile(r"[a-z0-9]+")

_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "for", "on", "in", "at", "my",
    "me", "please", "go", "find", "then", "with", "from", "it", "is", "be",
}

#: Things that are almost always worth having on the table.
_ALWAYS_USEFUL = (
    "search", "submit", "next", "continue", "accept", "login", "sign in",
    "menu", "close", "ok", "yes", "no", "more",
)


def tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 1}


def score_element(el: Element, goal_tokens: set[str], history_tokens: set[str]) -> float:
    """Higher is more likely to be the thing we want to touch next."""
    blob = " ".join([el.text, el.name, el.placeholder, el.value, el.href, el.role, el.tag])
    el_tokens = tokens(blob)
    if not el_tokens:
        return 0.05

    overlap = len(goal_tokens & el_tokens)
    score = 2.4 * overlap / max(1, len(goal_tokens) ** 0.5)

    lowered = blob.lower()
    if any(hint in lowered for hint in _ALWAYS_USEFUL):
        score += 0.55
    if el.tag in {"input", "textarea"} or el.role in {"searchbox", "textbox", "combobox"}:
        score += 0.7  # text fields are how goals usually start
    if el.tag == "button" or el.role == "button":
        score += 0.35
    if el.in_viewport:
        score += 0.6
    if el.rect[1] < 900:
        score += 0.2  # near the top of the document
    # Mild penalty for things we have already poked at, to break loops.
    if el_tokens & history_tokens:
        score -= 0.45
    if el.tag == "a" and not (el.text or el.name):
        score -= 0.5
    return score


_COMMAND_PREFIX = re.compile(
    r"^(?:please\s+)?(?:go\s+to|open|visit|search\s+(?:for|on)?|look\s+up|find(?:\s+me)?|"
    r"get(?:\s+me)?|show\s+me|buy|order|book|check)\s+",
    re.I,
)
# Only strips a trailing site reference ("… on prisjakt.nu"), never a location
# ("… in Malmö"), so the domain has to actually look like a domain.
_SITE_SUFFIX = re.compile(r"\s+(?:on|at|from|using|via)\s+(?:https?://)?[\w-]+(?:\.[a-z]{2,})+/?\S*$", re.I)


def text_candidates(goal: str, limit: int = 3) -> list[str]:
    """Plausible strings to type, so Laya can *pick* text instead of generating it.

    Laya is non-autoregressive — it cannot write. But typing is usually "put the
    subject of the goal in the box", so we precompute two or three options and
    let it choose. Anything cleverer falls through to the LLM.
    """
    goal = " ".join((goal or "").split())
    options: list[str] = []

    quoted = re.findall(r'"([^"]{2,80})"|“([^”]{2,80})”', goal)
    for pair in quoted:
        value = next((part for part in pair if part), "")
        if value:
            options.append(value)

    stripped = _COMMAND_PREFIX.sub("", goal)
    stripped = _SITE_SUFFIX.sub("", stripped).strip(" .?!").strip('"“”')
    if stripped and stripped.lower() != goal.lower():
        options.append(stripped)
    options.append(goal)

    seen: set[str] = set()
    unique = []
    for option in options:
        key = option.lower()
        if option and key not in seen:
            seen.add(key)
            unique.append(option[:120])
    return unique[:limit]


def rank_candidates(
    elements: Iterable[Element],
    goal: str,
    history: Iterable[str] = (),
    limit: int = 12,
) -> list[Element]:
    goal_tokens = tokens(goal)
    history_tokens = tokens(" ".join(history))
    scored = sorted(
        elements,
        key=lambda el: score_element(el, goal_tokens, history_tokens),
        reverse=True,
    )
    return scored[:limit]
