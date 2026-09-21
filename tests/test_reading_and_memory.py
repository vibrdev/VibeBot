"""The LLM has to be able to read the page and remember what it found.

Before this, the LLM got the first 1,200 characters of body.innerText and twelve
pre-ranked elements. On a shop that is the header and a promo banner: the
listings and prices a goal was about never reached the model.
"""

from __future__ import annotations

import asyncio

import pytest

from vibebot.llm.base import _as_notes, parse_action, render_prompt
from vibebot.schema import Element, Observation


# ------------------------------------------------------------------ notes

def test_notes_parse_from_the_shapes_small_models_actually_send():
    assert _as_notes(["a", "b"]) == ["a", "b"]
    assert _as_notes("one fact") == ["one fact"]
    assert _as_notes({"cheapest": "21,450 SEK"}) == ["cheapest: 21,450 SEK"]
    assert _as_notes(None) == []
    assert _as_notes(["", "none", "  real  fact "]) == ["real fact"]


def test_notes_ride_along_in_a_normal_reply():
    reply = parse_action(
        '{"op": "click", "element_idx": 25, "notes": ["M2 Max 96GB used: 21,450 SEK"], '
        '"reason": "sort by price"}'
    )
    assert reply.op == "click" and reply.element_idx == 25
    assert reply.notes == ["M2 Max 96GB used: 21,450 SEK"]


def test_a_reply_without_notes_still_parses():
    assert parse_action('{"op": "scroll"}').notes == []


def _agent_with_memory():
    from vibebot.agent import Agent

    agent = Agent.__new__(Agent)
    agent._memory = []
    return agent


def test_memory_deduplicates_and_is_bounded():
    from vibebot.agent import MEMORY_CHARS, MEMORY_ITEMS

    agent = _agent_with_memory()
    assert agent._remember(["A costs 10", "a  costs 10"]) == ["A costs 10"]
    assert agent._remember(["A costs 10"]) == []
    for n in range(MEMORY_ITEMS * 2):
        agent._remember([f"listing {n} costs {n * 1000} SEK"])
    assert len(agent._memory) <= MEMORY_ITEMS
    assert sum(map(len, agent._memory)) <= MEMORY_CHARS
    assert agent._memory[-1].startswith(f"listing {MEMORY_ITEMS * 2 - 1}")  # newest kept


# ----------------------------------------------------------------- prompt

def test_prompt_shows_the_page_and_the_notes():
    page = "[27] MacBook Pro M2 Max 96GB\n21,450 SEK"
    obs = Observation(url="http://shop/search", title="Results", elements=[],
                      page_text=page, page_text_total=len(page))
    obs.memory = ["page 1 cheapest used 96GB: 22,300 SEK"]
    prompt = render_prompt("find the cheapest", obs, [], [], "")
    assert "21,450 SEK" in prompt
    assert "22,300 SEK" in prompt
    assert f"PAGE (all {len(page)} characters)" in prompt


def test_prompt_says_when_the_page_was_cut_short():
    obs = Observation(url="u", title="t", elements=[], page_text="x" * 100, page_text_total=5000)
    assert "100 of 5000 characters" in render_prompt("g", obs, [], [], "")


def test_prompt_falls_back_when_the_reader_did_not_run():
    obs = Observation(url="u", title="t", elements=[], text_digest="top of page")
    prompt = render_prompt("g", obs, [Element(idx=3, tag="a", text="Deals")], [], "")
    assert "[3]" in prompt and "top of page" in prompt


# --------------------------------------------------- the reader, for real

def _browser_or_skip():
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError:
        pytest.skip("playwright is not installed")


def test_the_reader_puts_listings_and_prices_in_front_of_the_model(tmp_path):
    """On the benchmark shop's results page the old input was the header and a
    promo. The new one has to carry every listing with its price and number."""
    _browser_or_skip()
    from vibebot.benchsite import start
    from vibebot.browser import BrowserSession
    from vibebot.config import Config

    async def run():
        site = start()
        cfg = Config()
        cfg.browser.headless = True
        cfg.browser.user_data_dir = str(tmp_path / "profile")
        browser = BrowserSession(cfg.browser)
        try:
            await browser.start()
            await browser.goto(site.base_url + "/search?q=macbook&page=3")
            obs = await browser.observe(screenshot=False, read_chars=6000)
            await browser.goto(site.base_url + "/wiki/halvardsen-tower")
            wiki = await browser.observe(screenshot=False, read_chars=6000)
            await browser.goto(site.base_url + "/search?q=macbook&cond=Used&sort=price_asc")
            filtered = await browser.observe(screenshot=False, read_chars=6000)
            return obs, wiki, filtered
        finally:
            await browser.close()
            site.stop()

    try:
        obs, wiki, filtered = asyncio.run(run())
    except Exception as exc:  # noqa: BLE001 - no chromium on this machine
        pytest.skip(f"browser unavailable: {exc}")

    text = obs.page_text
    # The right answer to the headline goal, with its number next to it.
    assert "M2 Max 96GB 2TB Space Grey" in text and "21,450 SEK" in text
    title_line = next(line for line in text.splitlines() if "M2 Max 96GB 2TB Space Grey" in line)
    assert title_line.startswith("["), title_line
    # Content comes before the site's menus.
    assert text.index("21,450 SEK") < text.index("Gift cards")
    # Controls a shopper needs are clickable, not just readable.
    assert "Price: lowest first" in text and "[" in text.split("Price: lowest first")[0][-8:]
    # The wiki answer is thousands of characters in; the old view stopped at 1,200.
    assert "completed in 1907" in wiki.page_text
    assert "completed in 1907" not in " ".join(wiki.text_digest.split())[:1200]
    # One element per line: "[22] M3 [23] M4" on one line was misread twice.
    assert not any(line.count("[") > 1 for line in text.splitlines() if line.startswith("[")), text
    # Active filters and sort say so, and warn that clicking again turns them off.
    used_line = next(line for line in filtered.page_text.splitlines() if "Used" in line and line.startswith("["))
    assert "(selected" in used_line, used_line
    sort_line = next(line for line in filtered.page_text.splitlines() if "lowest first" in line)
    assert "(selected" in sort_line, sort_line
