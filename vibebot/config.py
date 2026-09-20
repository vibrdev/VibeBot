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
    backend: str = "laya"
    """laya | jev | heuristic"""
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
    min_headroom_mb: int = 3000
    """Refuse to load the model below this much grantable memory.

    Torch does not raise when the OS declines a commit charge - it dies. A run
    here segfaulted with no traceback while 7 GB of RAM sat free, because the
    commit limit had only 2.8 GB left. Better to run degraded and say so."""
    max_candidates: int = 12
    """How many page elements Laya is allowed to choose between. Laya's context is
    512-1024 tokens and its own docs warn about high-cardinality choices, so we
    pre-rank and truncate rather than dumping the whole DOM at it."""
    page_text_chars: int = 600
    """How much visible page text Laya gets. Every character is a token it pays
    for: 600 costs roughly 1-2s per step on a laptop CPU, 200 is noticeably
    snappier and usually enough."""
    accept_probability: float = 0.55
    """Minimum probability on the winning option before we act without the LLM.

    Gate on probability, not on Laya's `confidence` field: that field is
    1 - normalised entropy, so it sinks as the option list grows and would
    escalate perfectly good picks."""
    accept_margin: float = 0.15
    """Minimum lead over the runner-up. A photo finish means "ask the LLM"."""
    done_confidence: float = 0.80
    risk_threshold: float = 0.60
    """P(action is consequential) above which we ask you first. Laya reads the
    whole page for this, so it runs a little hot — raise it if it nags."""
    jev_base_url: str = ""
    jev_api_key_env: str = "JEV_API_KEY"


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
        self._validate_backends()

    def _validate_backends(self) -> None:
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
