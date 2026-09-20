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


@dataclass
class DeciderConfig:
    backend: str = "laya"
    """laya | jev | heuristic"""
    model: str = "auto"
    """auto (Laya Router), or an explicit repo id such as convaiinnovations/laya."""
    device: str = "cpu"
    """cpu | cuda"""
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


@dataclass
class LLMConfig:
    backend: str = "ollama"
    """ollama | openai | none"""
    model: str = "qwen2.5vl:7b"
    base_url: str = "http://127.0.0.1:11434"
    api_key_env: str = "OPENAI_API_KEY"
    vision: bool = True
    """Send annotated screenshots. Turn off for text-only models."""
    temperature: float = 0.1
    timeout_s: float = 180.0
    max_tokens: int = 1024


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
        cfg._apply_env()
        return cfg

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
