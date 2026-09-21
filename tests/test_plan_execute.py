"""decider.mode: plan - the LLM plans, the fast decider carries steps out.

The point is to give the fast decider the question it is suited to - "which
element is this?" - instead of "what should happen next, for this goal?",
which it answered with confident wrong clicks in gate mode.
"""

from __future__ import annotations

import asyncio

from vibebot.agent import Agent, _intent_kind, _intent_operation, _quoted
from vibebot.config import Config
from vibebot.deciders.base import Verdict
from vibebot.llm.base import LLMAction, parse_action
from vibebot.schema import Action, Element, Observation

SEARCH = Element(idx=1, tag="input", input_type="search", name="Search MacMarket")
USED = Element(idx=9, tag="a", text="Used", href="/search?cond=Used")
SORT = Element(idx=26, tag="a", text="Price: lowest first", href="/search?sort=price_asc")
NEXT = Element(idx=38, tag="a", text="Next page", href="/search?page=2")
DROPDOWN = Element(idx=5, tag="select", name="Sort by")
PAGE = [SEARCH, USED, SORT, NEXT, DROPDOWN]


# ---------------------------------------------------------------- parsing

def test_the_plan_comes_back_from_the_reply():
    reply = parse_action(
        '{"seen": "home page", "op": "type", "element_idx": 1, "text": "macbook pro", '
        '"plan": ["click \'Used\'", "click \'Price: lowest first\'", "read the page and answer"]}'
    )
    assert reply.plan == ["click 'Used'", "click 'Price: lowest first'", "read the page and answer"]


def test_step_kinds():
    assert _intent_kind("click 'Price: lowest first'") == "element"
    assert _intent_kind("type 'macbook pro' into the search box") == "element"
    assert _intent_kind("read the page and answer") == "answer"
    assert _intent_kind("Report the price of the cheapest one") == "answer"
    assert _intent_kind("go to http://127.0.0.1:8777/search?q=macbook") == "navigate"
    assert _intent_kind("scroll down to see more") == "scroll"
    assert _intent_kind("go back to the results") == "back"
    assert _intent_kind("go to page 2") == "element"  # no URL: a link to find


def test_quotes_survive_apostrophes():
    assert _quoted("click the item's 'Buy It Now'") == ["Buy It Now"]
    assert _quoted('type "macbook pro" into the search box') == ["macbook pro"]
    assert _quoted("click Next page") == []


def test_the_verb_comes_from_the_element_not_the_model():
    assert _intent_operation("click 'Used'", USED, ["Used"]) == ("click", "")
    assert _intent_operation("type 'macbook pro' into the search box", SEARCH, ["macbook pro"]) == (
        "type", "macbook pro")
    assert _intent_operation("search for macbook pro", SEARCH, []) == ("type", "macbook pro")
    assert _intent_operation("type 'macbook' into 'Search MacMarket'", SEARCH, ["macbook", "Search MacMarket"]) == (
        "type", "macbook")
    assert _intent_operation("select 'Price: lowest first'", DROPDOWN, ["Price: lowest first"]) == (
        "select", "Price: lowest first")
    # "select" on a link is a click - the mistake Laya made in gate mode.
    assert _intent_operation("select 'Used'", USED, ["Used"]) == ("click", "")


# ------------------------------------------------------------- the step

class FakeDecider:
    """Picks whatever it is told to, at a chosen confidence, and records what
    it was asked - so the test can check the question, not just the answer."""

    name = "laya"

    def __init__(self, target: str, p: float):
        self.target, self.p, self.asked = target, p, []

    async def decide(self, goal, obs, candidates, history, text_options):
        self.asked.append({"goal": goal, "candidates": [c.idx for c in candidates], "text": obs.text_digest})
        others = {"ask_llm": round(1 - self.p, 3)}
        return Verdict(target=self.target, probabilities={self.target: self.p, **others}, backend="laya")


def _agent(decider, plan):
    agent = Agent.__new__(Agent)
    agent.cfg = Config()
    agent.cfg.policy.autonomy = "yolo"
    agent.decider = decider
    agent.goal = "find the cheapest used macbook"
    agent._plan = list(plan)
    agent._drifted = ""
    agent._tried = set()
    agent._last_click = None
    agent._last_typed = ""
    agent._history = []
    llm_notes: list[str] = []

    async def llm_step(obs, candidates, note):
        llm_notes.append(note)
        return Action(op="wait", text="0", source="llm"), "llm"

    async def emit(kind, **payload):
        return None

    agent._llm_step = llm_step
    agent.emit = emit
    return agent, llm_notes


def _obs(text="30 results for macbook · page 1 of 3"):
    return Observation(url="http://shop/search?q=macbook", title="t", elements=PAGE,
                       text_digest="lots of page text", page_text=text)


def _run(agent, obs):
    return asyncio.run(agent._plan_step(obs, [], stalls=0, step=1))


def test_a_confident_match_is_executed_without_the_llm():
    decider = FakeDecider("e26", 0.95)
    agent, llm = _agent(decider, ["click 'Price: lowest first'", "read the page and answer"])
    action, escalated, _ = _run(agent, _obs())
    assert (action.op, action.element_idx, action.source) == ("click", 26, "laya")
    assert escalated is None and not llm
    assert agent._plan == ["read the page and answer"]
    # It was asked about the step, not the whole goal, and not given page text.
    assert decider.asked[0]["goal"] == "click 'Price: lowest first'"
    assert decider.asked[0]["text"] == ""
    assert 26 in decider.asked[0]["candidates"]


def test_an_unsure_match_goes_back_to_the_llm_and_drops_the_plan():
    agent, llm = _agent(FakeDecider("e9", 0.40), ["click 'Price: lowest first'", "read the page and answer"])
    action, escalated, _ = _run(agent, _obs())
    assert escalated == "llm" and "could not find" in llm[0]
    assert agent._plan == []


def test_reading_and_answering_is_the_llm_s_job():
    decider = FakeDecider("e26", 0.99)
    agent, llm = _agent(decider, ["read the page and answer"])
    _run(agent, _obs())
    assert llm and "read the page and answer" in llm[0]
    assert not decider.asked


def test_no_results_stops_the_plan():
    decider = FakeDecider("e9", 0.99)
    agent, llm = _agent(decider, ["click 'Used'"])
    _run(agent, _obs(text="0 results for used MacBook Pro M-series"))
    assert llm and "no results" in llm[0] and not decider.asked


def test_a_step_that_would_undo_the_last_one_goes_to_the_llm():
    """Worded differently from the last step, so not skipped as a repeat -
    but Laya matches it to the filter just switched on, which would undo it."""
    agent, llm = _agent(FakeDecider("e9", 0.99), ["filter to second-hand condition"])
    agent._last_click = ("/search", "click", USED.label(60).lower())
    _run(agent, _obs())
    assert llm and "undo" in llm[0]


def test_typing_takes_the_text_from_the_step():
    agent, _ = _agent(FakeDecider("e1", 0.97), ["type 'macbook pro' into the search box"])
    action, _, _ = _run(agent, _obs())
    assert (action.op, action.element_idx, action.text) == ("type", 1, "macbook pro")


def test_a_url_step_needs_no_model():
    decider = FakeDecider("e1", 0.99)
    agent, llm = _agent(decider, ["go to http://127.0.0.1:8777/search?q=macbook"])
    action, _, _ = _run(agent, _obs())
    assert (action.op, action.text, action.source) == ("navigate", "http://127.0.0.1:8777/search?q=macbook", "plan")
    assert not decider.asked and not llm


def test_plan_mode_is_the_default_and_validated():
    import pytest
    import yaml

    from vibebot.config import _merge

    assert Config().decider.backend == "match" and Config().decider.mode == "plan"
    cfg = Config()
    _merge(cfg, yaml.safe_load("decider:\n  mode: planning\n"))
    with pytest.raises(ValueError, match="decider.mode"):
        cfg._normalize()


def test_a_plan_that_starts_with_the_step_just_taken_skips_it():
    """It sorted by price, then planned "click 'Price: lowest first'" first.
    Laya found it, the undo guard refused it, and the LLM was called again."""
    decider = FakeDecider("e38", 0.97)
    agent, llm = _agent(decider, ["click 'Price: lowest first' to sort results", "click 'Next page'"])
    agent._last_click = ("/search", "click", SORT.label(60).lower())
    action, escalated, _ = _run(agent, _obs())
    assert (action.element_idx, action.source) == (38, "laya") and not llm
    assert decider.asked[0]["goal"] == "click 'Next page'"


def test_a_plan_of_nothing_but_the_last_step_goes_to_the_llm():
    agent, llm = _agent(FakeDecider("e26", 0.97), ["click 'Price: lowest first'"])
    agent._last_click = ("/search", "click", SORT.label(60).lower())
    _run(agent, _obs())
    assert llm and "Choose the next step" in llm[0]


def test_retyping_the_same_search_is_skipped():
    agent, llm = _agent(FakeDecider("e26", 0.97), ["type 'macbook pro' into the search box", "click 'Price: lowest first'"])
    agent._last_typed = "macbook pro"
    action, _, _ = _run(agent, _obs())
    assert action.element_idx == 26 and not llm


# ------------------------------------------------------ the string baseline

def test_the_label_matcher_picks_a_unique_quoted_label():
    from vibebot.deciders import LabelMatchDecider

    matcher = LabelMatchDecider(Config().decider)
    verdict = asyncio.run(matcher.decide("click 'Price: lowest first'", _obs(), PAGE, [], ["Price: lowest first"]))
    assert verdict.target == "e26" and verdict.p_top == 1.0
    unquoted = asyncio.run(matcher.decide("click Next page", _obs(), PAGE, [], []))
    assert unquoted.target == "e38"


def test_the_label_matcher_defers_when_it_cannot_be_sure():
    from vibebot.deciders import LabelMatchDecider

    matcher = LabelMatchDecider(Config().decider)
    assert asyncio.run(matcher.decide("sort cheapest first", _obs(), PAGE, [], [])).target == "ask_llm"
    twice = PAGE + [Element(idx=40, tag="a", text="Used")]
    assert asyncio.run(matcher.decide("click 'Used'", _obs(), twice, [], ["Used"])).target == "ask_llm"


def test_the_label_matcher_takes_a_page_number_the_planner_named():
    """The planner wrote "click [18] 96 to set memory filter"; Laya clicked
    "Used". A number that exists on the page cannot be misread."""
    from vibebot.deciders import LabelMatchDecider

    memory = Element(idx=18, tag="a", text="96", href="/search?mem=96")
    obs = Observation(url="http://shop/search", title="t", elements=PAGE + [memory], page_text="x")
    matcher = LabelMatchDecider(Config().decider)
    verdict = asyncio.run(matcher.decide("click [18] 96 to set memory filter", obs, PAGE, [], []))
    assert verdict.target == "e18"
