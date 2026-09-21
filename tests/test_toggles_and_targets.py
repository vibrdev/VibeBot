"""Laya undoing filters, and the LLM misreading an index on a crowded line.

Both observed on the benchmark shop, with the page reader in place: the LLM
sorted and filtered sensibly, and Laya clicked the same filter again straight
after, which on most sites turns it off.
"""

from __future__ import annotations

import asyncio

from vibebot.agent import Agent, _click_signature, _named_target
from vibebot.config import Config
from vibebot.deciders.base import Verdict
from vibebot.schema import Action, Element, Observation

FACETS = [
    Element(idx=20, tag="a", text="M1", href="/search?chip=M1"),
    Element(idx=21, tag="a", text="M2", href="/search?chip=M2"),
    Element(idx=22, tag="a", text="M3", href="/search?chip=M3"),
    Element(idx=23, tag="a", text="M4", href="/search?chip=M4"),
]


def test_the_llm_is_taken_at_its_word_when_the_number_is_off_by_one():
    """It clicked [23] with text "M3" - [23] is M4."""
    action = Action(op="click", element_idx=23, text="M3")
    assert _named_target(action, FACETS) == 22


def test_a_matching_number_is_left_alone():
    assert _named_target(Action(op="click", element_idx=22, text="M3"), FACETS) is None


def test_no_correction_without_one_clear_match():
    assert _named_target(Action(op="click", element_idx=23, text="an M-series chip"), FACETS) is None
    doubled = FACETS + [Element(idx=40, tag="a", text="M3")]
    assert _named_target(Action(op="click", element_idx=23, text="M3"), doubled) is None


def test_typed_text_is_never_read_as_a_target():
    """For "type", text is what goes in the box, not which box."""
    assert _named_target(Action(op="type", element_idx=23, text="M3"), FACETS) is None


def test_the_same_filter_on_the_filtered_page_has_the_same_signature():
    """Filter on and filter off differ in query string and element numbers,
    so neither can be how a toggle is recognised."""
    used = Element(idx=9, tag="a", text="Refurbished", href="/search?cond=Refurbished")
    used_later = Element(idx=10, tag="a", text="Refurbished", href="/search")
    a = Observation(url="http://shop/search?q=macbook", title="", elements=[used])
    b = Observation(url="http://shop/search?q=macbook&cond=Refurbished", title="", elements=[used_later])
    assert _click_signature(Action(op="click", element_idx=9), a) == _click_signature(
        Action(op="click", element_idx=10), b
    )


def test_laya_does_not_click_the_filter_it_just_clicked():
    refurb = Element(idx=10, tag="a", text="Refurbished", href="/search?cond=Refurbished")
    obs = Observation(url="http://shop/search?q=macbook&cond=Refurbished", title="", elements=[refurb])

    agent = Agent.__new__(Agent)
    agent.cfg = Config()
    agent.goal = "find the cheapest refurbished macbook"
    agent._drifted = ""
    agent._just_navigated = False
    agent._not_done, agent._tried, agent._scrolls = set(), set(), {}
    agent._last_click = ("/search", "click", refurb.label(60).lower())
    notes: list[str] = []

    async def capture(o, c, note):
        notes.append(note)
        return None, "llm"

    agent._llm_step = capture
    verdict = Verdict(target="e10", probabilities={"e10": 1.0, "ask_llm": 0.0}, operation="click")
    action, _ = asyncio.run(agent._resolve(obs, [refurb], verdict, stalls=0))
    assert action is None and notes and "undoes it" in notes[0]


# ------------------------------------------------ giving up too early

def test_a_nothing_found_answer_is_recognised():
    from vibebot.agent import _sounds_like_nothing
    from vibebot.llm.base import LLMAction

    for text in (
        "No listings match for refurbished MacBook M3 Max",
        "No listings match.",
        "I could not find any results for that.",
        "There are 0 results.",
        "That model does not exist on this site.",
    ):
        assert _sounds_like_nothing(LLMAction(op="done", text=text)), text
    assert _sounds_like_nothing(LLMAction(op="fail", text="impossible"))


def test_real_answers_are_not_challenged():
    from vibebot.agent import _sounds_like_nothing
    from vibebot.llm.base import LLMAction

    for text in ("21,450 SEK", "3", "greenbyte_refurb", "30 days", "1907",
                 "The cheapest used one is 21,450 SEK, not the new one"):
        assert not _sounds_like_nothing(LLMAction(op="done", text=text)), text
    assert not _sounds_like_nothing(LLMAction(op="click", element_idx=3, text="no results"))


def test_giving_up_is_challenged_once_then_believed():
    from vibebot.llm.base import LLMAction

    agent = Agent.__new__(Agent)
    agent._challenged = False
    agent._pages_checked = False
    agent._incomplete_checked = False
    agent.goal = "find a refurbished M3 Max"
    notes: list[str] = []

    async def escalate(o, c, note):
        notes.append(note)
        return LLMAction(op="done", text="No listings match.")

    agent._escalate = escalate
    first = LLMAction(op="done", text="No listings match.")
    obs = Observation(url="u", title="t", elements=[])
    again = asyncio.run(agent._second_look(first, obs, []))
    assert len(notes) == 1 and "Loosen the search" in notes[0]
    assert asyncio.run(agent._second_look(again, obs, [])) is again  # not asked twice
    assert len(notes) == 1
