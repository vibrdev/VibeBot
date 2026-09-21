"""Checks the agent makes on its own answers, and what it tells the model.

Each one comes from a benchmark failure with the LLM deciding every step:
counting page 1 of 3, stacking filters on a search that matched nothing, and
clicking an active filter "to verify it" - which turned it off.
"""

from __future__ import annotations

import asyncio

from vibebot.agent import Agent, _more_pages
from vibebot.llm.base import NO_RESULTS, LLMAction, parse_action, render_prompt
from vibebot.schema import Element, Observation


def test_seen_comes_back_from_the_reply():
    reply = parse_action('{"seen": "30 results, page 1 of 3, Used filter on", "op": "click", "element_idx": 38}')
    assert reply.seen == "30 results, page 1 of 3, Used filter on"
    assert reply.element_idx == 38


def test_zero_result_pages_are_recognised():
    for text in ("0 results for macbook", "No listings match.", "Your search did not match any items",
                 "No results found"):
        assert NO_RESULTS.search(text), text
    for text in ("30 results for macbook", "10 results", "Results 1-10", "no-nonsense returns"):
        assert not NO_RESULTS.search(text), text


def test_the_prompt_shouts_when_there_are_no_results():
    empty = Observation(url="u", title="t", elements=[], page_text="0 results for used MacBook M-series",
                        page_text_total=40)
    full = Observation(url="u", title="t", elements=[], page_text="30 results for macbook", page_text_total=22)
    assert "NO RESULTS" in render_prompt("g", empty, [], [], "")
    assert "NO RESULTS" not in render_prompt("g", full, [], [], "")


def test_more_pages_is_detected_from_the_text_or_a_next_link():
    assert _more_pages(Observation(url="u", title="t", elements=[], page_text="22 results · page 1 of 3"))
    assert not _more_pages(Observation(url="u", title="t", elements=[], page_text="3 results · page 1 of 1"))
    assert _more_pages(Observation(url="u", title="t", elements=[Element(idx=1, tag="a", text="Next page")]))


def _agent(goal):
    agent = Agent.__new__(Agent)
    agent.goal = goal
    agent._challenged = False
    agent._pages_checked = False
    agent._incomplete_checked = False
    asked: list[str] = []

    async def escalate(o, c, note):
        asked.append(note)
        return LLMAction(op="done", text="3")

    agent._escalate = escalate
    return agent, asked


def test_a_count_from_page_one_of_three_is_questioned_once():
    agent, asked = _agent("find out how many used listings have 128GB")
    obs = Observation(url="u", title="t", elements=[], page_text="22 results · page 1 of 3")
    first = asyncio.run(agent._second_look(LLMAction(op="done", text="2"), obs, []))
    assert len(asked) == 1 and "more pages" in asked[0]
    asyncio.run(agent._second_look(first, obs, []))
    assert len(asked) == 1  # once per run


def test_a_single_page_answer_or_a_non_comparison_is_left_alone():
    # Known and accepted: "how many days" reads as a count, so a help page that
    # also showed "page 1 of 3" would get one extra question. Help pages do not.
    agent, asked = _agent("find what year the Halvardsen Tower was completed")
    obs = Observation(url="u", title="t", elements=[], page_text="22 results · page 1 of 3")
    asyncio.run(agent._second_look(LLMAction(op="done", text="1907"), obs, []))
    assert not asked

    agent, asked = _agent("find the cheapest used one")
    single = Observation(url="u", title="t", elements=[], page_text="3 results · page 1 of 1")
    asyncio.run(agent._second_look(LLMAction(op="done", text="21,450 SEK"), single, []))
    assert not asked


def test_an_answer_that_admits_a_missing_part_is_sent_back_once():
    """It found the right listing and answered "16,400 SEK ... (seller not
    specified in visible text)" - twice. The seller was one click away."""
    agent, asked = _agent("find who sells the cheapest refurbished M3 Max")
    agent._incomplete_checked = False
    obs = Observation(url="u", title="t", elements=[], page_text="3 results · page 1 of 1")
    answer = LLMAction(op="done", text="MacBook Pro 14in M3 Max - 16,400 SEK (seller not specified in visible text)")
    asyncio.run(agent._second_look(answer, obs, []))
    assert len(asked) == 1 and "item's own page" in asked[0]


def test_complete_answers_are_not_sent_back():
    agent, asked = _agent("find who sells the cheapest refurbished M3 Max")
    agent._incomplete_checked = False
    obs = Observation(url="u", title="t", elements=[], page_text="Sold by greenbyte_refurb")
    asyncio.run(agent._second_look(LLMAction(op="done", text="greenbyte_refurb"), obs, []))
    assert not asked


# ------------------------------------------ what "no results" should say

def test_no_results_names_the_search_words_and_the_box():
    """A vague "broaden the search or remove a filter" made the model toggle
    the one filter it knew eleven times; it never touched the search box."""
    from vibebot.llm.base import _no_results_alert

    obs = Observation(
        url="http://shop/search?q=used+MacBook+Pro+M-series&sort=price_asc",
        title="t", elements=[],
        page_text='[1] (search box "Search MacMarket", contains "used MacBook Pro M-series")\n0 results',
    )
    alert = _no_results_alert(obs)
    assert '"used MacBook Pro M-series" matches nothing' in alert
    assert "[1]" in alert and "no filter can fix it" in alert


def test_no_results_without_a_query_suggests_removing_a_filter():
    from vibebot.llm.base import _no_results_alert

    obs = Observation(url="http://shop/search?cond=Used&mem=96", title="t", elements=[], page_text="0 results")
    assert "Remove a" in _no_results_alert(obs)


def test_the_llm_is_told_when_it_is_about_to_undo_its_last_click():
    """With Laya off, the LLM clicked "Used" on and off eleven times."""
    from vibebot.config import Config

    used = Element(idx=9, tag="a", text="Used", href="/search?q=x&cond=Used")
    other = Element(idx=1, tag="input", input_type="search", name="Search")
    obs = Observation(url="http://shop/search?q=x&sort=price_asc", title="t", elements=[used, other])

    agent = Agent.__new__(Agent)
    agent.cfg = Config()
    agent.cfg.policy.autonomy = "yolo"
    agent._tried = set()
    agent._last_click = ("/search", "click", used.label(60).lower())
    asked: list[str] = []

    async def escalate(o, c, note):
        asked.append(note)
        return LLMAction(op="type", element_idx=1, text="macbook pro")

    agent._escalate = escalate
    agent._second_look = lambda d, o, c: asyncio.sleep(0, result=d)
    action, _ = asyncio.run(agent._use_llm_decision(LLMAction(op="click", element_idx=9), obs, [used, other]))
    assert asked and "undoes that" in asked[0]
    assert action.op == "type" and action.element_idx == 1
