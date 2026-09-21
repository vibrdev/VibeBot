"""Regressions for the failures that actually happened on a real run.

Each test here is a bug that was observed in a trace, not a hypothetical.
Run with: python -m pytest tests -q
"""

from __future__ import annotations

from vibebot.agent import _coerce_operation
from vibebot.llm.base import LLMHTTPError, parse_action
from vibebot.schema import Element
from vibebot.sysmem import snapshot

LINK = Element(idx=3, tag="a", text="Deals")
SEARCH = Element(idx=18, tag="input", role="combobox", name="Search for anything")
DROPDOWN = Element(idx=5, tag="select", role="listbox", name="Sort")
BUTTON = Element(idx=9, tag="button", text="Search")


# --------------------------------------------------------------- element_idx

def test_element_idx_accepts_a_single_item_list():
    """qwen3.5:4b answered `"element_idx": [18]` having correctly picked eBay's
    search box. That became None, and the agent scrolled instead."""
    assert parse_action('{"op":"click","element_idx":[18],"text":"MacBook"}').element_idx == 18


def test_element_idx_accepts_the_prompt_s_own_spelling():
    """The element list is rendered as "e18", so a model may echo it back."""
    assert parse_action('{"op":"click","element_idx":"e18"}').element_idx == 18
    assert parse_action('{"op":"click","element_idx":["e18"]}').element_idx == 18


def test_element_idx_plain_forms():
    assert parse_action('{"op":"click","element_idx":18}').element_idx == 18
    assert parse_action('{"op":"click","element_idx":"18"}').element_idx == 18
    assert parse_action('{"op":"click","element_idx":18.0}').element_idx == 18


def test_element_idx_rejects_what_is_not_an_index():
    for blob in (
        '{"op":"click","element_idx":null}',
        '{"op":"click","element_idx":[3,7]}',      # two choices is not an action
        '{"op":"click","element_idx":true}',       # bool is an int subclass
        '{"op":"click","element_idx":"none"}',
        '{"op":"click","element_idx":{}}',
    ):
        assert parse_action(blob).element_idx is None, blob


def test_parse_action_keeps_the_raw_reply():
    action = parse_action('{"op":"click","element_idx":1,"reason":"because"}')
    assert "element_idx" in action.raw


# ----------------------------------------------------------------- operation

def test_select_on_a_link_becomes_click():
    """Laya answers "which element" and "what to do" separately, so it paired
    eBay's "Deals" link with `select`. Playwright raised, the step failed, and
    Laya picked the identical pair for six steps running."""
    assert _coerce_operation("select", LINK, "anything")[0] == "click"
    assert _coerce_operation("select", LINK, "")[0] == "click"


def test_select_survives_on_a_real_select():
    assert _coerce_operation("select", DROPDOWN, "Price + shipping")[0] == "select"


def test_type_on_a_link_or_button_becomes_click():
    assert _coerce_operation("type", LINK, "macbook")[0] == "click"
    assert _coerce_operation("type", BUTTON, "macbook")[0] == "click"


def test_type_into_a_field_with_no_text_asks_the_llm():
    op, note = _coerce_operation("type", SEARCH, "")
    assert op == "type"
    assert note  # escalates rather than typing nothing


def test_llm_click_with_text_on_a_field_becomes_type():
    """The LLM asked to click the search box and supplied the query. Clicking
    only focuses it and the text is dropped."""
    assert _coerce_operation("click", SEARCH, "MacBook Pro M4", text_is_deliberate=True)[0] == "type"


def test_laya_click_with_text_is_left_alone():
    """Laya always carries a text candidate, so it must not trigger the same
    coercion — it would type the whole goal into any box it clicked."""
    assert _coerce_operation("click", SEARCH, "the entire goal string")[0] == "click"


def test_llm_click_without_text_stays_a_click():
    assert _coerce_operation("click", SEARCH, "", text_is_deliberate=True)[0] == "click"


# -------------------------------------------------------------------- memory

def test_snapshot_reports_headroom_not_just_free_ram():
    """Free RAM was the wrong number: torch died with 7 GB free because the
    commit limit had 2.8 GB left."""
    memory = snapshot()
    if memory is None:
        return  # unsupported platform; nothing to assert
    assert memory.total_mb > 0
    assert memory.headroom_mb >= 0
    assert memory.limit_mb >= memory.committed_mb
    assert "headroom" in memory.short()


def test_llm_http_error_carries_the_status():
    error = LLMHTTPError("HTTP 500 from 127.0.0.1: out of memory", 500)
    assert error.status_code == 500
    assert "out of memory" in str(error)


# --------------------------------------------------------------- plausibility

from vibebot.ranking import goal_terms, is_plausible  # noqa: E402

WIKI_GOAL = "go to en.wikipedia.org and find what year the Eiffel Tower was completed"
WIKI_URL = "https://en.wikipedia.org/wiki/Main_Page"
EBAY_GOAL = "go to ebay.com and find the cheapest used MacBook Pro M4 listing; report the price"
EBAY_URL = "https://www.ebay.com/"


def test_goal_terms_drop_the_site_you_are_already_on():
    """"go to en.wikipedia.org ..." otherwise makes every "Wikipedia" link look
    topical, and a run walked front page -> English Wikipedia -> back."""
    terms = goal_terms(WIKI_GOAL, WIKI_URL)
    assert "eiffel" in terms and "tower" in terms
    assert "wikipedia" not in terms and "org" not in terms


def test_off_topic_picks_are_not_plausible():
    """Every one of these was chosen by Laya at p>=0.74 on a real run."""
    terms = goal_terms(WIKI_GOAL, WIKI_URL)
    for text in ("Wikipedia", "English Wikipedia", "Main Page", "Misti as seen from Arequipa"):
        assert not is_plausible(Element(idx=1, tag="a", text=text), terms), text

    ebay = goal_terms(EBAY_GOAL, EBAY_URL)
    for text in ("Deals", "My eBay", "Brand Outlet"):
        assert not is_plausible(Element(idx=1, tag="a", text=text), ebay), text


def test_on_topic_and_navigational_picks_are_plausible():
    terms = goal_terms(WIKI_GOAL, WIKI_URL)
    assert is_plausible(Element(idx=1, tag="a", text="Eiffel Tower"), terms)
    assert is_plausible(Element(idx=2, tag="input", role="combobox", name="Search Wikipedia"), terms)

    ebay = goal_terms(EBAY_GOAL, EBAY_URL)
    assert is_plausible(Element(idx=3, tag="a", text="Apple MacBook Pro M4 14-inch"), ebay)
    assert is_plausible(Element(idx=4, tag="button", text="Next"), ebay)


# ------------------------------------------------------------ start / stop

import os  # noqa: E402
from pathlib import Path  # noqa: E402

from vibebot.config import Config  # noqa: E402
from vibebot.server import _pid_file  # noqa: E402


def test_pid_file_written_while_running_and_removed_after(tmp_path):
    """`Stop VibeBot` finds the server through this file."""
    target = tmp_path / "nested" / "server.pid"
    with _pid_file(str(target)):
        assert target.read_text(encoding="ascii") == str(os.getpid())
    assert not target.exists()


def test_pid_file_survives_an_unwritable_path(tmp_path):
    """A pid file that cannot be written is not a reason to refuse to start."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="ascii")
    with _pid_file(str(blocker / "server.pid")):
        pass  # must not raise


def test_pid_file_removed_even_if_the_server_raises(tmp_path):
    target = tmp_path / "server.pid"
    try:
        with _pid_file(str(target)):
            raise RuntimeError("server crashed")
    except RuntimeError:
        pass
    assert not target.exists()


def test_launcher_scripts_are_double_clickable():
    """Batch files need CRLF, and must not have picked up stray control
    characters - an earlier pass turned \taskkill into a literal tab."""
    root = Path(__file__).resolve().parent.parent
    for name in ("VibeBot.cmd", "Stop VibeBot.cmd"):
        raw = (root / name).read_bytes()
        assert raw, f"{name} is empty"
        assert b"\r\n" in raw, f"{name} needs CRLF line endings"
        assert b"\t" not in raw, f"{name} contains a tab"
        assert b"\f" not in raw, f"{name} contains a form feed"
        raw.decode("ascii")  # non-ASCII breaks on other code pages


def test_server_defaults_are_one_double_click():
    cfg = Config()
    assert cfg.server.open_browser is True
    assert cfg.server.pid_file.endswith("server.pid")


# ------------------------------------------------------------------- the UI

import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402

UI_HTML = Path(__file__).resolve().parent.parent / "vibebot" / "ui" / "index.html"


def _script_source() -> str:
    html = UI_HTML.read_text(encoding="utf-8")
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    assert blocks, "the UI has no script block"
    return "\n".join(blocks)


def test_ui_javascript_parses():
    """A syntax error in the page is invisible from Python and total from the
    user's side: the socket never opens, the status pill sits on
    "connecting...", and every button does nothing. That shipped once, when a
    confirm() message picked up real newlines inside a string literal.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed; cannot parse-check the UI")
    with tempfile.TemporaryDirectory() as folder:
        js = Path(folder) / "ui.js"
        js.write_text(_script_source(), encoding="utf-8")
        result = subprocess.run(
            [node, "--check", str(js)], capture_output=True, text=True, timeout=60
        )
    assert result.returncode == 0, f"UI JavaScript does not parse:\n{result.stderr}"


def test_ui_string_literals_stay_on_one_line():
    """The specific failure above, catchable without node.

    Walks the script tracking quote state. A ' or " literal left open at the
    end of a line is the bug that shipped; backticks may legitimately span
    lines, so they carry over.
    """
    backslash = chr(92)
    in_template = False
    for number, line in enumerate(_script_source().splitlines(), 1):
        quote = "`" if in_template else ""
        index = 0
        while index < len(line):
            char = line[index]
            if quote:
                if char == backslash:
                    index += 2
                    continue
                if char == quote:
                    quote = ""
            elif char in "'" + chr(34) + "`":
                quote = char
            elif char == "/" and line[index : index + 2] == "//":
                break
            index += 1
        assert quote in ("", "`"), f"unterminated {quote} string on script line {number}: {line!r}"
        in_template = quote == "`"


def test_ui_still_wires_up_its_controls():
    source = _script_source()
    for handler in ("$('go').onclick", "$('stop').onclick", "$('quit').onclick", "function connect("):
        assert handler in source, f"{handler} is missing from the UI"


# --------------------------------------------------- how often Laya may act

from vibebot.agent import _offtopic_and_unsure  # noqa: E402

SHOPPING = goal_terms(
    "find, dont buy, FIND the cheapest second hand macbook pro m4 on the market",
    "https://www.google.com/search",
)


def test_a_confident_pick_acts_even_when_the_words_do_not_match():
    """The regression the user hit: Laya wanted "For Sale" at p=0.91 on a page
    of shopping results, the gate escalated because the label shares no word
    with the goal, and the LLM clicked "Store" instead. Three runs in a row
    reached 0% Laya."""
    for_sale = Element(idx=32, tag="link", text="For Sale")
    assert not is_plausible(for_sale, SHOPPING), "precondition: the words do not match"
    assert not _offtopic_and_unsure(for_sale, SHOPPING, p_top=0.91, trust=0.85)


def test_a_vague_offtopic_pick_still_escalates():
    vague = Element(idx=9, tag="a", text="Help & Contact")
    assert _offtopic_and_unsure(vague, SHOPPING, p_top=0.60, trust=0.85)


def test_an_ontopic_pick_acts_at_any_confidence():
    listing = Element(idx=5, tag="a", text="Apple MacBook Pro M4 refurbished")
    assert not _offtopic_and_unsure(listing, SHOPPING, p_top=0.60, trust=0.85)


def test_trust_threshold_is_configurable_and_sane():
    from vibebot.config import DeciderConfig

    assert 0.5 < DeciderConfig().trust_confidence < 1.0


# ----------------------------------------------------------- consent / auth

def test_consent_and_signin_buttons_count_as_risky():
    """A run clicked "Accept All" on a cookie banner and then "Continue with
    Google", landing on account creation. Neither label contains "sign in"."""
    from vibebot.config import PolicyConfig

    words = PolicyConfig().risky_keywords
    for label in ("Accept All", "Continue with Google", "Agree", "Sign up", "Create account"):
        assert any(word in label.lower() for word in words), label


def test_ordinary_buttons_are_not_risky():
    from vibebot.config import PolicyConfig

    words = PolicyConfig().risky_keywords
    for label in ("Search", "Next page", "Used (387) Items", "Price + Shipping: lowest first"):
        assert not any(word in label.lower() for word in words), label


# -------------------------------------------------- what the log must show

def test_search_engines_are_a_waypoint_not_the_destination():
    """The drift guard fired on a click from Google to eBay - "which is not
    what the goal is about" - when following a search result to a shop is the
    entire point of searching."""
    from vibebot.agent import _is_search_host

    for host in ("www.google.com", "google.se", "duckduckgo.com", "www.bing.com"):
        assert _is_search_host(host), host
    for host in ("www.ebay.com", "en.wikipedia.org", "backmarket.se", "creativecommons.org"):
        assert not _is_search_host(host), host


def test_trace_writes_down_what_you_were_asked(tmp_path):
    """A run you approved must read back as a run you approved."""
    from vibebot.trace import Trace

    trace = Trace(str(tmp_path), "run")
    record = {
        "step": 3,
        "url": "https://example.com",
        "title": "t",
        "candidates": [],
        "laya": {"backend": "laya", "target": "e1"},
        "action": {"op": "click", "source": "laya"},
        "outcome": "clicked element 1",
        "asked_you": [{"question": "About to: click Buy Now", "answer": "ok"}],
    }
    trace.step_note(record, [])
    log = (tmp_path / "run" / "run.log").read_text(encoding="utf-8")
    assert "ASKED YOU" in log and "click Buy Now" in log
    assert "YOU SAID" in log and "ok" in log


# ------------------------------------------------------- one key per page

def test_the_same_page_spelled_two_ways_is_one_key():
    """Wikipedia served title=Special:Search and title=Special%3ASearch for the
    same results page two steps apart. The loop guard saw two pages and let
    Laya bounce between that page and a red link for the rest of the run."""
    from vibebot.agent import _page_key

    a = "https://en.wikipedia.org/w/index.php?search=Eiffel+Tower&title=Special:Search&fulltext=1"
    b = "https://en.wikipedia.org/w/index.php?search=Eiffel+Tower&title=Special%3ASearch&fulltext=1"
    assert _page_key(a) == _page_key(b)


def test_page_key_ignores_the_fragment_and_case_of_the_host():
    from vibebot.agent import _page_key

    assert _page_key("https://EN.wikipedia.org/wiki/X#Licensing") == _page_key("https://en.wikipedia.org/wiki/X")


def test_page_key_keeps_genuinely_different_pages_apart():
    from vibebot.agent import _page_key

    assert _page_key("https://ebay.com/a") != _page_key("https://ebay.com/b")
    assert _page_key("https://ebay.com/s?q=one") != _page_key("https://ebay.com/s?q=two")


# ------------------------------------------------------ when to interrupt

from vibebot.agent import _is_plain_navigation, _is_search_typing  # noqa: E402


def test_a_plain_link_click_is_reversible():
    """Laya reported risky_p=1.00 for eBay's "Deals" link and 0.98 for
    "Laptops & Netbooks". Both are category links; Back undoes them."""
    link = Element(idx=31, tag="a", text="Laptops & Netbooks", href="/b/Laptops/175672")
    assert _is_plain_navigation(link, "click")


def test_a_link_that_is_really_a_button_is_not():
    """No href, "#" or javascript: means it does something rather than going
    somewhere, and that something need not be undoable."""
    for href in ("", "#", "javascript:void(0)"):
        fake = Element(idx=1, tag="a", text="Delete", href=href)
        assert not _is_plain_navigation(fake, "click"), href


def test_only_clicks_count_as_navigation():
    link = Element(idx=1, tag="a", text="Next", href="/page/2")
    assert not _is_plain_navigation(link, "type")


def test_typing_a_query_into_search_is_reversible():
    for element in (
        Element(idx=1, tag="input", role="searchbox", name="Search"),
        Element(idx=2, tag="input", input_type="search"),
        Element(idx=3, tag="input", role="combobox", name="Search for anything"),
    ):
        assert _is_search_typing(element, "type")


def test_typing_into_an_ordinary_field_is_not():
    field = Element(idx=9, tag="input", role="textbox", name="Card number")
    assert not _is_search_typing(field, "type")


def test_risky_words_still_win_over_reversibility():
    """A link labelled "Buy It Now" has an href like any other, so the
    exemption must not swallow it. The gate checks the keyword first."""
    from vibebot.config import PolicyConfig

    words = PolicyConfig().risky_keywords
    for label in ("Buy It Now", "Buy", "Continue with Google", "Accept All"):
        assert any(w in label.lower() for w in words), label


def test_risk_gate_interrupts_for_consequences_not_for_confidence():
    """The contract, end to end. Laya's risky_p must not be able to stop a
    reversible action on its own, and must not be needed to stop an
    irreversible one."""
    import asyncio

    from vibebot.agent import Agent
    from vibebot.config import Config
    from vibebot.schema import Action, Observation

    class FakeBrowser:
        async def element_is_password(self, idx):
            return False

    agent = Agent.__new__(Agent)
    agent.cfg = Config()
    agent.on_event = lambda event: asyncio.sleep(0)
    agent._answer = None
    agent._step_asks = []
    agent._history = []
    agent.browser = FakeBrowser()
    asked: list[str] = []

    async def fake_ask(question):
        asked.append(question)
        return "ok"

    agent.ask_user = fake_ask

    def interrupts(element, op, risky):
        asked.clear()
        obs = Observation(url="https://www.ebay.com/", title="t", elements=[element])
        action = Action(op=op, element_idx=element.idx, text="q" if op == "type" else "",
                        reason="laya", source="laya", risky=risky)
        asyncio.run(agent._risk_gate(action, obs))
        return bool(asked)

    # Harmless, however sure Laya is that it is not.
    assert not interrupts(Element(idx=31, tag="a", text="Laptops & Netbooks", href="/b/x"), "click", 0.98)
    assert not interrupts(Element(idx=3, tag="a", text="Deals", href="/deals"), "click", 1.00)
    assert not interrupts(Element(idx=20, tag="input", role="combobox", name="Search for anything"), "type", 1.00)

    # Consequential, however sure Laya is that it is not.
    assert interrupts(Element(idx=61, tag="a", text="Buy It Now", href="/itm/1"), "click", 0.10)
    assert interrupts(Element(idx=6, tag="button", text="Accept All"), "click", 0.10)
    assert interrupts(Element(idx=7, tag="a", text="Continue with Google", href="/oauth"), "click", 0.10)

    # Not a plain navigation, so risky_p still gets to speak.
    assert interrupts(Element(idx=1, tag="input", input_type="checkbox"), "click", 0.87)
    assert interrupts(Element(idx=9, tag="input", role="textbox", name="Card number"), "type", 0.87)
    assert interrupts(Element(idx=4, tag="a", text="Remove", href="javascript:void(0)"), "click", 0.80)


# --------------------------------------------------------------- benchmark

def test_benchmark_scores_the_answer_not_the_status():
    from vibebot.bench import Task

    task = Task("eiffel", "…", r"\b1889\b")
    assert task.passed("The Eiffel Tower was completed in 1889.")
    assert not task.passed("I found a page about the Eiffel Tower.")
    assert not task.passed("")


def test_benchmark_matching_ignores_case_and_surrounding_words():
    from vibebot.bench import Task

    task = Task("python", "…", r"van rossum")
    assert task.passed("It was created by Guido van Rossum in 1991.")
    assert task.passed("GUIDO VAN ROSSUM")


def test_benchmark_summary_reports_what_changed_between_runs():
    from vibebot.bench import Report, Result

    report = Report([
        Result("a", True, "done", 2, 70.9, 0.0, 2, "x", ""),
        Result("b", False, "max_steps", 10, 190.0, 0.5, 8, "y", ""),
    ])
    summary = report.summary()
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["median_steps"] == 6
    assert summary["llm_calls"] == 10
    assert summary["laya_share"] == 0.25


def test_every_benchmark_goal_has_something_to_check():
    """A suite entry with no expectation always passes, which is worse than
    having no benchmark at all."""
    from vibebot.bench import SUITE

    assert SUITE
    for task in SUITE:
        assert task.expect.strip(), task.name
        assert task.goal.strip(), task.name
        assert task.max_steps > 0, task.name


# ------------------------------------------- what the LLM is told, exactly

def _resolve_note(verdict, url, not_done=()):
    """Run the real _resolve and capture the note it would send the LLM."""
    import asyncio

    from vibebot.agent import Agent, _page_key
    from vibebot.config import Config
    from vibebot.schema import Observation

    agent = Agent.__new__(Agent)
    agent.cfg = Config()
    agent.goal = "go to example.com and tell me which organisation the domain is reserved by"
    agent._drifted = ""
    agent._just_navigated = False
    agent._not_done = {_page_key(u) for u in not_done}
    agent._tried = set()
    agent._scrolls = {}
    seen: list[str] = []

    async def capture(obs, candidates, note):
        seen.append(note)
        return None, "llm"

    agent._llm_step = capture
    obs = Observation(url=url, title="Example Domain", elements=[])
    asyncio.run(agent._resolve(obs, [], verdict, stalls=0))
    return seen[0] if seen else ""


def test_a_done_lean_on_a_fresh_page_invites_the_answer():
    """Told "you have already been asked about this page" on its first visit,
    the LLM declined to answer on example.com - whose text holds the answer -
    and wandered off to iana.org until the step budget ran out."""
    from vibebot.deciders.base import Verdict

    verdict = Verdict(target="done", probabilities={"done": 0.36, "e0": 0.2}, done_p=0.7987)
    note = _resolve_note(verdict, "https://example.com/")
    assert "already been asked" not in note
    assert "reply 'done'" in note


def test_a_done_lean_on_a_page_already_refused_says_so():
    from vibebot.deciders.base import Verdict

    verdict = Verdict(target="done", probabilities={"done": 0.98, "e0": 0.01}, done_p=0.98)
    note = _resolve_note(verdict, "https://example.com/", not_done=["https://example.com/"])
    assert "already been asked" in note


# ------------------------------------------------------------------ ranking

def test_the_site_you_are_on_does_not_outrank_the_search_box():
    """Measured on Wikipedia's front page: every link's href contains
    en.wikipedia.org, the goal said "go to en.wikipedia.org", and the search
    box ranked below twelfth - so it was never offered at all."""
    from vibebot.ranking import rank_candidates

    url = "https://en.wikipedia.org/wiki/Main_Page"
    goal = "go to en.wikipedia.org and find what year the Eiffel Tower was completed"
    elements = [
        Element(idx=i, tag="a", text=t, href=f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}")
        for i, t in enumerate(
            ["The Arnolfini Portrait", "Northern Renaissance", "iconography", "Noemvriana",
             "Lebanon war", "More current events", "Nominate an article", "Donate",
             "Wikipedia", "free", "anyone can edit", "Cold spots", "Main Page"], start=2)
    ]
    elements.append(Element(idx=1, tag="input", name="Search Wikipedia", input_type="search"))
    ranked = rank_candidates(elements, goal, [], limit=12, url=url)
    assert ranked[0].idx == 1, [e.label() for e in ranked[:3]]


def test_href_scoring_uses_the_path_not_the_host():
    from vibebot.ranking import _href_path

    assert _href_path("https://en.wikipedia.org/wiki/Eiffel_Tower") == "/wiki/Eiffel_Tower"
    assert _href_path("/wiki/Eiffel_Tower") == "/wiki/Eiffel_Tower"
    assert _href_path("") == ""


def test_the_llm_is_held_to_the_repeat_rule_once():
    """Told Laya's repeat had got nowhere, the LLM repeated it itself."""
    import asyncio

    from vibebot.agent import Agent, _page_key
    from vibebot.config import Config
    from vibebot.llm.base import LLMAction
    from vibebot.schema import Observation

    url = "https://en.wikipedia.org/wiki/Main_Page"
    wiki = Element(idx=12, tag="a", text="Wikipedia", href="/wiki/Wikipedia")
    search = Element(idx=1, tag="input", name="Search Wikipedia", input_type="search")
    obs = Observation(url=url, title="t", elements=[wiki, search])

    agent = Agent.__new__(Agent)
    agent.cfg = Config()
    agent.cfg.policy.autonomy = "yolo"
    agent._tried = {(_page_key(url), "click", 12)}
    agent._challenged = False
    agent._pages_checked = False
    agent._incomplete_checked = False
    agent._last_click = None
    agent.goal = "find the year"
    asked: list[str] = []

    async def escalate(o, c, note):
        asked.append(note)
        return LLMAction(op="type", element_idx=1, text="Eiffel Tower")

    agent._escalate = escalate
    action, _ = asyncio.run(
        agent._use_llm_decision(LLMAction(op="click", element_idx=12), obs, [wiki, search])
    )
    assert asked and "already been done" in asked[0]
    assert action.element_idx == 1 and action.op == "type"

    # and it does not loop forever if the model insists
    async def insist(o, c, note):
        asked.append(note)
        return LLMAction(op="click", element_idx=12)

    agent._escalate = insist
    asked.clear()
    action, _ = asyncio.run(
        agent._use_llm_decision(LLMAction(op="click", element_idx=12), obs, [wiki, search])
    )
    assert len(asked) == 1
    assert action.element_idx == 12


def test_laya_does_not_switch_into_a_blank_tab():
    """A run began with two about:blank tabs and Laya spent two steps
    switching between them at p=0.72 and p=0.88."""
    from vibebot.deciders.base import Verdict
    from vibebot.schema import TabInfo

    import asyncio

    from vibebot.agent import Agent
    from vibebot.config import Config
    from vibebot.schema import Observation

    def resolve(other_url):
        agent = Agent.__new__(Agent)
        agent.cfg = Config()
        agent.goal = "find something"
        agent._drifted = ""
        agent._just_navigated = False
        agent._not_done, agent._tried, agent._scrolls = set(), set(), {}
        notes: list[str] = []

        async def capture(obs, candidates, note):
            notes.append(note)
            return None, "llm"

        agent._llm_step = capture
        obs = Observation(url="https://example.com/", title="t", elements=[])
        obs.tabs = [TabInfo(index=0, url="https://example.com/", title="t", active=True),
                    TabInfo(index=1, url=other_url, title="", active=False)]
        obs.active_tab = 0
        verdict = Verdict(target="switch_tab", probabilities={"switch_tab": 0.9, "back": 0.05})
        action, _ = asyncio.run(agent._resolve(obs, [], verdict, stalls=0))
        return action, notes

    action, notes = resolve("about:blank")
    assert action is None and notes  # went to the LLM instead

    action, notes = resolve("https://www.iana.org/domains")
    assert action.op == "switch_tab" and action.text == "1"


# ------------------------------------------------------ YAML's bare "off"

def test_decider_backend_off_survives_yaml():
    """`backend: off` parses as False under YAML 1.1. The factory read
    `False or "laya"`, and a benchmark labelled "off" ran with Laya deciding
    55% of its steps."""
    import yaml

    from vibebot.config import Config, _merge
    from vibebot.deciders import build_decider

    cfg = Config()
    _merge(cfg, yaml.safe_load("decider:\n  backend: off\n"))
    cfg._normalize()
    assert cfg.decider.backend == "off"
    assert build_decider(cfg.decider).name == "off"


def test_llm_backend_off_does_not_become_ollama():
    import yaml

    from vibebot.config import Config, _merge

    cfg = Config()
    _merge(cfg, yaml.safe_load("llm:\n  backend: off\n"))
    cfg._normalize()
    assert cfg.llm.backend == "off"


def test_an_unknown_decider_is_an_error_not_a_quiet_default():
    import pytest
    import yaml

    from vibebot.config import Config, _merge

    cfg = Config()
    _merge(cfg, yaml.safe_load("decider:\n  backend: lyaa\n"))
    with pytest.raises(ValueError, match="decider.backend"):
        cfg._normalize()
