"""JSONL run logs.

Two reasons this exists. One: when a run goes sideways you want to see exactly
what Laya thought. Two: these rows are labelled training data — every step where
Laya deferred and the LLM (or you) picked the right element is a free example
for fine-tuning Laya on your own sites later.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Trace:
    def __init__(self, directory: str, run_id: str):
        self.dir = Path(directory).expanduser() / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "steps.jsonl"
        self.run_id = run_id
        self._started = time.time()

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def save_shot(self, step: int, png: bytes | None) -> str | None:
        if not png:
            return None
        target = self.dir / f"step-{step:03d}.png"
        target.write_bytes(png)
        return str(target)

    def finish(self, status: str, summary: str) -> None:
        self.write(
            {
                "type": "run_end",
                "status": status,
                "summary": summary,
                "elapsed_s": round(time.time() - self._started, 2),
            }
        )
