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
