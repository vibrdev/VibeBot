"""Run logs: one machine-readable JSONL, one readable transcript.

Three reasons this exists. One: when a run goes sideways you want to see
exactly what Laya thought, what the LLM was shown, and what it actually
replied. Two: these rows are labelled training data — every step where Laya
deferred and the LLM (or you) picked the right element is a free example for
fine-tuning Laya on your own sites later. Three: a run that fails at 40 steps
is unreadable as JSON, so the same events are also written as a transcript you
can open and scroll.

Layout per run, under `trace_dir/<run id>/`:

    steps.jsonl   every event, complete, for analysis
    run.log       the same run as readable text
    step-NNN.png  what the model was looking at
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .sysmem import snapshot


class Trace:
    def __init__(self, directory: str, run_id: str):
        self.dir = Path(directory).expanduser() / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "steps.jsonl"
        self.log_path = self.dir / "run.log"
        self.run_id = run_id
        self._started = time.time()

    # ------------------------------------------------------------ machine

    def write(self, record: dict[str, Any]) -> None:
        record.setdefault("ts", time.time())
        memory = snapshot()
        if memory is not None:
            # Memory is the thing that kills runs on a small machine, and it is
            # invisible after the fact unless it is written down per step.
            record.setdefault(
                "memory",
                {
                    "headroom_mb": round(memory.headroom_mb),
                    "available_mb": round(memory.available_mb),
                    "committed_mb": round(memory.committed_mb),
                },
            )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------- human

    def note(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp}  {text}\n")

    def step_note(self, record: dict[str, Any], llm_calls: list[dict[str, Any]]) -> None:
        """The readable half of a step."""
        laya = record.get("laya", {})
        action = record.get("action", {})
        memory = snapshot()
        self.note("")
        self.note(f"--- step {record.get('step')} --- {record.get('url', '')[:110]}")
        self.note(f"    title      : {record.get('title', '')[:100]}")
        self.note(f"    candidates : {len(record.get('candidates', []))}")
        for label in record.get("candidates", [])[:12]:
            self.note(f"        {label[:110]}")
        backend = laya.get("backend", "?")
        self.note(
            f"    laya       : {backend} -> {laya.get('target')} "
            f"p={laya.get('p_top')} margin={laya.get('margin')} "
            f"op={laya.get('operation')} done_p={laya.get('done_p')} "
            f"risky_p={laya.get('risky_p')} ({laya.get('latency_ms')}ms)"
        )
        problem = (record.get("laya", {}).get("probabilities") or {}).get("error")
        if problem:
            self.note(f"    LAYA FAILED: {problem}")
        for call in llm_calls:
            self.note(f"    llm ask    : {call.get('note', '')[:160]}")
            self.note(f"    llm reply  : {str(call.get('raw', ''))[:400]}")
        self.note(
            f"    action     : [{action.get('source')}] {action.get('op')} "
            f"idx={action.get('element_idx')} text={str(action.get('text'))[:60]!r} "
            f"- {str(action.get('reason'))[:90]}"
        )
        self.note(f"    outcome    : {record.get('outcome')}  ({record.get('duration_ms')}ms)")
        if memory is not None:
            self.note(f"    memory     : {memory.short()}")

    # ------------------------------------------------------------- assets

    def save_shot(self, step: int, png: bytes | None) -> str | None:
        if not png:
            return None
        target = self.dir / f"step-{step:03d}.png"
        target.write_bytes(png)
        return str(target)

    def save_prompt(self, step: int, call: int, prompt: str) -> None:
        """The exact prompt the model saw. Too big for the transcript, but the
        first thing you want when a reply makes no sense."""
        if not prompt:
            return
        (self.dir / f"step-{step:03d}-llm{call}.prompt.txt").write_text(prompt, encoding="utf-8")

    # ------------------------------------------------------------ bookends

    def start(self, goal: str, config: dict[str, Any]) -> None:
        memory = snapshot()
        self.write({"type": "run_start", "goal": goal, "config": config})
        self.note(f"=== run {self.run_id} ===")
        self.note(f"goal: {goal}")
        if memory is not None:
            self.note(f"memory at start: {memory.short()}")

    def finish(self, status: str, summary: str) -> None:
        elapsed = round(time.time() - self._started, 2)
        self.write({"type": "run_end", "status": status, "summary": summary, "elapsed_s": elapsed})
        self.note("")
        self.note(f"=== {status} after {elapsed}s ===")
        self.note(summary)
