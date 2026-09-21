"""Configuration: YAML file + environment overrides, with local-only defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BrowserConfig:
    headless: bool = False
    """Headless is forced on when no display is available (servers)."""
    width: int = 1440
    height: int = 900
    user_data_dir: str = "./.vibebot/profile"
    """Persistent profile so logins survive between runs."""
    start_url: str = "about:blank"
    action_timeout_ms: int = 15000
    nav_timeout_ms: int = 30000
    locale: str = "en-US"
    max_frames: int = 12
    """Frames scanned per page (main document + iframes). Ad-heavy pages can
    have dozens; each one costs a round trip."""
    follow_new_tabs: bool = True
    """Switch to a tab the page opens, the way a person would."""
    screenshot_max_width: int = 1024
    """Downscale screenshots to this width before anything sees them (0 = off).

    Vision models charge by image area. Measured on qwen3.5:4b with the same
    eBay screenshot, prompt tokens scale almost linearly with width:

        1440x900 -> 1280 tokens, 20.9s
        1024x640 ->  660 tokens, 17.2s
         768x480 ->  380 tokens, 16.4s
         512x320 ->  180 tokens, 16.5s

    At 1440 the image alone eats more than the whole llm.max_tokens budget,
    which is how a reply ends up truncated before it is valid JSON. It does
    *not* change how much memory Ollama commits (~5.7 GB at every size above),
    so this is a token and latency fix, not a memory one."""


@dataclass
class DeciderConfig:
    backend: str = "match"
    """match | laya | jev | heuristic | off - the fast "system 1".

    In plan mode its one job is "which element is this plan step?". Measured
    with `vibebot bench --executor` (20 steps on the benchmark shop, half in
    the planner's wording, half paraphrased):

                         planner wording        paraphrased          per step
                         right  wrong  defer    right  wrong  defer
        laya             4      4      2        0      8      2         3.2 s
        match            10     0      0        1      0      9           0 ms

    Zero-shot Laya is wrong more often than right, at high confidence, and the
    order of the options changes its answer; a check with the options
    reversed and shuffled ruled out position bias (it picked the first option
    3/36 times) - it is drawn to "Search" whatever the step says. Label
    matching is never wrong: it acts when a label matches and defers when not.
    So it is the default, and `laya` / `jev` are one line away; run the exam
    on anything you want to put in this slot before trusting it."""
    mode: str = "plan"
    """plan | gate - how the fast decider and the LLM share the work.

    gate (the original design): the fast decider answers "which of these
    elements should I act on next, for this goal?" every step, and the LLM
    takes over when it is unsure. Measured on the benchmark shop, two rounds
    each, against the LLM taking every step:

                      passed   median per goal   total    LLM calls
        gate          7/10     144s              1469s    41
        off           7/10      57s               922s    47

    Same accuracy, 2.5x slower: "what next, for this goal" is a planning
    question, and a zero-shot classifier with no view of page state answered
    it with confident wrong clicks (p=1.00 on "Deals", on the Eiffel Tower
    logo, on the filter it had just switched on) that then had to be undone.

    plan: the LLM, when called, acts and writes the next few steps as short
    intents ("click 'Price: lowest first'"). The fast decider carries each one
    out by answering a much narrower question - "which element is this?" -
    over candidates ranked against the intent. The LLM is called again only
    when a step cannot be matched, fails, or it is time to read and answer."""
    model: str = "auto"
    """auto (Laya Router), or an explicit repo id such as convaiinnovations/laya."""
    device: str = "cpu"
    """cpu | cuda"""
    preload: bool = False
    """Load every routed model up front, instead of the one actually used.

    Laya's Router(preload=True) pulls the English *and* multilingual weight
    sets. Measured here, that is ~2.5 GB of commit charged before the first
    page is even open, for models an English run never touches. Lazy loading
    gives identical answers (e0 at p=0.9855 on the same question either way)
    and pays the load cost once, on the first step that needs it."""
    trust_confidence: float = 0.85
    """Above this, Laya acts even on an element unrelated to the goal's wording.

    The relatedness check earns its keep on vague picks, but it reads only the
    words: "For Sale" on a page of shopping results shares nothing with "find
    the cheapest second hand macbook pro m4", and Laya wanted it at p=0.91.
    Escalating that traded a good decision for a worse one - the LLM clicked
    "Store" instead. Off-topic *and* hesitant is the combination worth a call."""
    min_headroom_mb: int = 3000
    """Refuse to load the model below this much grantable memory.

    Torch does not raise when the OS declines a commit charge - it dies. A run
    here segfaulted with no traceback while 7 GB of RAM sat free, because the
    commit limit had only 2.8 GB left. Better to run degraded and say so."""
    max_candidates: int = 12
    """How many page elements Laya is allowed to choose between. Laya's context is
    512-1024 tokens and its own docs warn about high-cardinality choices, so we
    pre-rank and truncate rather than dumping the whole DOM at it."""
    page_text_chars: int = 200
    """How much visible page text Laya gets, and the main thing you can trade.

    Measured on a laptop CPU with the weights warm, this is very close to
    linear: 0 chars 3.70s, 200 4.48s, 600 5.84s, 1000 7.10s. About 3.7s of a
    step is fixed and the rest is this number. The candidate list barely
    registers next to it (4 options 5.11s, 20 options 5.84s).

    The default was 600. Benchmarked against 200 with Laya genuinely loaded,
    same code, same four goals: both passed the same two and failed the same
    two, Laya decided 32% of steps either way, and 200 finished in 410s against
    442s. One round each, so that is evidence it costs nothing rather than
    proof it helps - `vibebot bench --page-text-chars N --repeat 3` to check."""
    accept_probability: float = 0.55
    """Minimum probability on the winning option before we act without the LLM.

    Gate on probability, not on Laya's `confidence` field: that field is
    1 - normalised entropy, so it sinks as the option list grows and would
    escalate perfectly good picks."""
    accept_margin: float = 0.15
    """Minimum lead over the runner-up. A photo finish means "ask the LLM"."""
    done_confidence: float = 0.80
    risk_threshold: float = 0.60
    """P(action is consequential) above which we ask you first.

    Measured over 208 recorded steps, this score is weak on its own: it clears
    0.60 on 39% of them, averages 0.71 where the label really is risky against
    0.46 where it is not, and came back 1.00 for eBay's "Deals" link. So it
    only decides actions that could do something Back cannot undo. Following a
    link, or typing a query into a search box, no longer counts however high it
    reads; policy.risky_keywords, which is exact, still stops the rest."""
    jev_base_url: str = ""
    jev_api_key_env: str = "JEV_API_KEY"


#: Every accepted `decider.backend` value -> the decider it selects. As with
#: the LLM table below, anything else is a startup error, never a quiet default.
DECIDER_BACKENDS: dict[str, str] = {
    "off": "off",
    "none": "off",
    "llm": "off",
    "laya": "laya",
    "jev": "jev",
    "heuristic": "heuristic",
    "match": "match",  # plan-mode baseline: exact label matching, no model
}

#: Every accepted `llm.backend` value -> the implementation it selects.
#: Spelled out on purpose: an unknown name is a configuration error, not a
#: reason to quietly pick something. `vibebot.llm.build_llm` routes on this.
LLM_BACKENDS: dict[str, str] = {
    "ollama": "ollama",
    "openai": "openai",
    "openai_compat": "openai",
    "openai-compat": "openai",
    "openrouter": "openai",
    "lmstudio": "openai",
    "lm_studio": "openai",
    "vllm": "openai",
    "llamacpp": "openai",
    "llama_cpp": "openai",
    "groq": "openai",
    "together": "openai",
    "none": "none",
    "off": "none",
    "disabled": "none",
}


@dataclass
class LLMConfig:
    backend: str = "ollama"
    """ollama | openai | none"""
    model: str = "qwen3.5:4b"
    """Any vision model Ollama serves. qwen3.5 is natively multimodal at every
    size, so 4b (3.4 GB) buys vision on a modest card; go up to :9b if you have
    the VRAM, down to :2b or :0.8b if you do not. Not benchmarked in this repo —
    a sane starting point, not a measured winner."""
    think: str = "off"
    """off | on | auto. Thinking is ON by default in Ollama for models that
    support it, and a thinking model puts its answer in `message.thinking`
    while `content` comes back empty — which reads as a broken model. We send
    `think: false` unless told otherwise. `auto` omits the field entirely."""
    base_url: str = ""
    """Empty means "whatever this backend's default is" - http://127.0.0.1:11434
    for ollama, https://api.openai.com for openai. Do not hard-code one backend's
    URL here: `backend: openai` inheriting Ollama's port used to report ready and
    then talk to localhost, because Ollama answers /v1/chat/completions too."""
    api_key_env: str = "OPENAI_API_KEY"
    vision: bool = True
    """Send annotated screenshots. Turn off for text-only models."""
    temperature: float = 0.1
    timeout_s: float = 180.0
    max_tokens: int = 1024
    page_chars: int = 6000
    """How much of the page the LLM reads, in characters, per step.

    It used to be the first 1,200 characters of the page's raw text, which on
    a shop is the header, the sign-in link and a promo banner. It is now the
    page in reading order with the menus moved to the end (browser._READ_JS),
    and this is its budget. A results page on the benchmark shop is about
    1,900 characters; real listing pages run to tens of thousands, where this
    is what decides how far down the model can see."""
    num_ctx: int = 8192
    """Ollama's context window, in tokens. Ignored by the openai backend.

    Ollama's own default is 4096, and that is now too small: measured, the
    system prompt plus 6,000 characters of page is 3,620 tokens before the
    ~660-token screenshot and the reply, so 4096 would silently cut the prompt.
    Measured on qwen3.5:4b, cost of the window on this machine:

        4096   model 3,555 MB   commit +5,621 MB
        8192   model 3,702 MB   commit +5,818 MB   (+197 MB)
       16384   model 3,997 MB   commit +6,083 MB   (+462 MB)

    Cheap, because the model's hybrid attention keeps the KV cache small.
    Latency barely moves (9.5s / 11.9s / 11.1s per call). Raise it with
    page_chars; do not change it mid-run, which forces an 8.3s reload."""
    keep_alive: str = "5m"
    """How long Ollama keeps the model resident between calls.

    Measured: qwen3.5:4b holds ~5.7 GB of commit while loaded, and Ollama's own
    default keeps it there for 5 minutes after the last call - including the
    whole time Laya wants memory, and long after a run has finished. VibeBot
    releases it explicitly on shutdown regardless of this value. Shorten it
    ("30s", or "0" to unload immediately) to trade reload latency for memory."""


@dataclass
class PolicyConfig:
    autonomy: str = "normal"
    """guided (confirm everything) | normal (confirm risky) | yolo (never confirm)"""
    max_steps: int = 40
    max_llm_calls: int = 25
    stall_limit: int = 3
    """Repeated no-op or identical actions before we escalate."""
    allow_domains: list[str] = field(default_factory=list)
    """Empty means "anywhere". Entries are matched as hostname suffixes."""
    block_domains: list[str] = field(default_factory=list)
    risky_keywords: list[str] = field(
        default_factory=lambda: [
            "buy", "purchase", "checkout", "pay", "order", "subscribe",
            "delete", "remove", "cancel", "send", "post", "publish", "tweet",
            "transfer", "withdraw", "confirm", "sign in", "log in", "password",
            # Consent and identity, added after a run clicked "Accept All" on a
            # cookie banner and then "Continue with Google", which put it two
            # steps into creating a Google account. Agreeing to terms and
            # handing over an identity are decisions with consequences outside
            # the page, and neither one contains the word "sign in".
            "accept all", "accept cookies", "agree", "consent",
            "continue with", "sign up", "register", "create account",
        ]
    )
    never_type_into_password: bool = True
    user_reply_timeout_s: float = 900.0


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    token: str = ""
    """Set this (or VIBEBOT_TOKEN) before binding to 0.0.0.0 on a server."""
    live_fps: float = 2.0
    """Frame rate of the optional live view. Off until a viewer asks for it;
    each frame is a full PNG screenshot, so this is not free."""
    open_browser: bool = True
    """Open the UI in your default browser once the server is listening, so
    starting VibeBot is one double-click and no copying URLs out of a console."""
    pid_file: str = "./.vibebot/server.pid"
    """Where the running server records its PID, so `Stop VibeBot` can find it
    without you hunting through Task Manager. Removed on a clean exit."""


@dataclass
class Config:
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    decider: DeciderConfig = field(default_factory=DeciderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    trace_dir: str = "./.vibebot/traces"

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Config":
        cfg = cls()
        candidates = [Path(path)] if path else [Path("config.yaml"), Path("config.example.yaml")]
        for candidate in candidates:
            if candidate.is_file():
                raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                _merge(cfg, raw)
                break
        cfg._normalize()   # before env, so _coerce sees a str and not a bool
        cfg._apply_env()
        cfg._normalize()
        return cfg

    def _normalize(self) -> None:
        """Repair values YAML mangles.

        `think: off` parses as the boolean False under YAML 1.1 (the same rule
        that turns Norway's `no` into False), which would silently skip the
        `think` field and leave thinking switched on. Accept every spelling.

        Also rejects an unroutable `llm.backend` here rather than at the first
        escalation, so a typo is a startup error instead of a silent reroute."""
        self.llm.think = _tristate(self.llm.think)
        # Same YAML rule, and it bit: `decider.backend: off` arrived as False,
        # the decider factory read `False or "laya"`, and a benchmark labelled
        # "off" was quietly run with Laya deciding 55% of its steps.
        # `llm.backend: off` would likewise have become "ollama".
        if self.decider.backend is False:
            self.decider.backend = "off"
        if self.llm.backend is False:
            self.llm.backend = "off"
        self._validate_backends()

    def _validate_backends(self) -> None:
        decider = str(self.decider.backend).strip().lower() if self.decider.backend else ""
        if decider not in DECIDER_BACKENDS:
            valid = ", ".join(sorted(DECIDER_BACKENDS))
            raise ValueError(
                f"decider.backend: unknown value {self.decider.backend!r}. Valid values: {valid}"
            )
        self.decider.backend = DECIDER_BACKENDS[decider]
        mode = str(self.decider.mode or "plan").strip().lower()
        if mode not in {"plan", "gate"}:
            raise ValueError(f"decider.mode: unknown value {self.decider.mode!r}. Valid values: gate, plan")
        self.decider.mode = mode

        name = str(self.llm.backend or "ollama").strip().lower()
        if name not in LLM_BACKENDS:
            valid = ", ".join(sorted(set(LLM_BACKENDS)))
            raise ValueError(f"llm.backend: unknown value {self.llm.backend!r}. Valid values: {valid}")
        self.llm.backend = name

    def _apply_env(self) -> None:
        """VIBEBOT_<SECTION>_<FIELD> overrides anything from the YAML file."""
        for section in fields(self):
            value = getattr(self, section.name)
            if not is_dataclass(value):
                env = os.environ.get(f"VIBEBOT_{section.name.upper()}")
                if env is not None:
                    setattr(self, section.name, env)
                continue
            for sub in fields(value):
                env = os.environ.get(f"VIBEBOT_{section.name.upper()}_{sub.name.upper()}")
                if env is not None:
                    setattr(value, sub.name, _coerce(env, getattr(value, sub.name)))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = {s.name: getattr(value, s.name) for s in fields(value)} if is_dataclass(value) else value
        return out


def _merge(target: Any, raw: dict[str, Any]) -> None:
    for key, value in raw.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif value is not None:
            setattr(target, key, value)


def _tristate(value: Any) -> str:
    """Anything that means off/on/auto -> "off" | "on" | "auto"."""
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in {"off", "false", "no", "0", "none", ""}:
        return "off"
    if text in {"on", "true", "yes", "1"}:
        return "on"
    return "auto"


def _coerce(text: str, like: Any) -> Any:
    if isinstance(like, bool):
        return text.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(like, int):
        return int(text)
    if isinstance(like, float):
        return float(text)
    if isinstance(like, list):
        return [part.strip() for part in text.split(",") if part.strip()]
    return text
