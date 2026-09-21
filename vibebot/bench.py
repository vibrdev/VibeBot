"""A repeatable score, so "better" stops being an opinion.

Every change in this repo so far has been argued from a single run, which is
how a fix that helps one goal and breaks two others gets shipped. `vibebot
bench` runs a fixed set of goals, scores the answers against a pattern, and
writes the numbers down.

What it reports per goal, and in aggregate:

    pass        did the answer match what the goal asked for
    steps       how many the agent needed
    seconds     wall clock, which is what you actually wait
    laya        share of steps the fast decider handled on its own
    llm         how many calls the slow one cost

Deliberately small and deliberately boring pages: the point is to measure the
agent, not the internet. Nothing here buys anything or signs in anywhere.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config


@dataclass
class Task:
    name: str
    goal: str
    expect: str
    """Regex the final answer has to contain, case-insensitively."""
    max_steps: int = 8

    def passed(self, summary: str) -> bool:
        return bool(re.search(self.expect, summary or "", re.I))


#: The default suite. Short goals with one checkable answer, on pages that do
#: not move much. A goal whose answer changes daily cannot tell you whether a
#: change to the agent helped.
SUITE: list[Task] = [
    Task("example-heading", "go to example.com and tell me the page heading", r"example domain", 6),
    Task("eiffel-year",
         "go to en.wikipedia.org and find what year the Eiffel Tower was completed", r"\b1889\b", 10),
    Task("python-creator",
         "go to en.wikipedia.org and find who created the Python programming language",
         r"van rossum", 10),
    Task("iana-reserved",
         "go to example.com and tell me which organisation the domain is reserved by",
         r"iana|internet assigned numbers", 8),
]


@dataclass
class Result:
    task: str
    passed: bool
    status: str
    steps: int
    seconds: float
    laya_share: float
    llm_calls: int
    answer: str
    trace: str

    def row(self) -> str:
        mark = "pass" if self.passed else "FAIL"
        return (
            f"{self.task:<18} {mark:<5} {self.status:<10} {self.steps:>3} steps "
            f"{self.seconds:>7.1f}s  laya {self.laya_share:>4.0%}  llm {self.llm_calls:>2}  "
            f"{self.answer[:44]}"
        )


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        if not self.results:
            return {}
        return {
            "tasks": len(self.results),
            "passed": sum(r.passed for r in self.results),
            "pass_rate": sum(r.passed for r in self.results) / len(self.results),
            "median_steps": statistics.median(r.steps for r in self.results),
            "total_seconds": round(sum(r.seconds for r in self.results), 1),
            "median_seconds": round(statistics.median(r.seconds for r in self.results), 1),
            "laya_share": round(
                statistics.mean(r.laya_share for r in self.results if r.steps), 3
            ),
            "llm_calls": sum(r.llm_calls for r in self.results),
        }


def _read_trace(path: str) -> tuple[int, float, int]:
    """(steps, laya share, llm calls) out of the run's own log."""
    steps = laya = calls = 0
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("type") != "step":
                continue
            steps += 1
            # Only count it if the real model decided. The heuristic fallback
            # also acts under source "laya", and a run where Laya never loaded
            # once reported a healthy 25% here.
            laya += (
                (record.get("action") or {}).get("source") == "laya"
                and (record.get("laya") or {}).get("backend") == "laya"
            )
            calls += len(record.get("llm_calls") or [])
    except Exception:  # noqa: BLE001 - a missing trace is not worth failing over
        pass
    return steps, (laya / steps if steps else 0.0), calls


async def run_task(agent: Any, task: Task) -> Result:
    """One goal against an already-running agent."""
    agent.cfg.policy.max_steps = task.max_steps
    # Start each goal from one blank tab. The previous answer still on screen
    # lets the next task "solve" itself; a tab the previous goal opened sends
    # the next one wandering - one run spent two steps switching between two
    # blank tabs before it did anything.
    for _ in range(20):  # bounded: a tab that will not close must not hang the suite
        if len(agent.browser.tabs()) <= 1:
            break
        await agent.browser.close_tab(len(agent.browser.tabs()) - 1)
    await agent.browser.goto("about:blank")

    started = time.perf_counter()
    outcome = await agent.run(task.goal)
    elapsed = time.perf_counter() - started

    steps, share, calls = _read_trace(outcome.get("trace", ""))
    summary = outcome.get("summary", "")
    return Result(
        task=task.name,
        passed=task.passed(summary) and outcome.get("status") == "done",
        status=str(outcome.get("status")),
        steps=steps,
        seconds=round(elapsed, 1),
        laya_share=share,
        llm_calls=calls,
        answer=" ".join(summary.split()),
        trace=str(outcome.get("trace", "")),
    )


async def run_suite(
    cfg: Config, tasks: list[Task], repeat: int = 1, allow_degraded: bool = False
) -> Report:
    """All the goals against one agent.

    One agent for the whole suite, not one per goal. Building a fresh one each
    time reloaded Laya's weights every goal (~35s) and, worse, ran the memory
    check while the previous goal's model was still resident: the second goal
    onwards silently fell back to the heuristic decider and the numbers were
    not comparable with the first.
    """
    from .agent import Agent

    report = Report()
    agent = Agent(cfg)

    # Nothing is watching a benchmark, so a question is a dead end, not a pause.
    async def no_one_is_there(question: str) -> str:
        return "(running unattended - carry on as best you can)"

    agent.ask_user = no_one_is_there  # type: ignore[method-assign]

    status = await agent.start()
    degraded = status.get("degraded_from")
    if degraded and not allow_degraded:
        await agent.shutdown()
        # A benchmark of the fallback, labelled as a benchmark of Laya, is
        # worse than no benchmark: it happened, and the numbers looked fine.
        raise SystemExit(
            f"Refusing to benchmark: the decider fell back to the heuristic ({degraded[1]}). "
            "Free some memory and run it again, or pass --allow-degraded to measure the fallback."
        )
    try:
        for round_number in range(1, repeat + 1):
            for task in tasks:
                label = task.name if repeat == 1 else f"{task.name}#{round_number}"
                print(f"  running {label} ...", flush=True)
                result = await run_task(agent, task)
                result.task = label
                report.results.append(result)
                print(f"    {result.row()}", flush=True)
    finally:
        await agent.shutdown()
    return report


def bench(
    cfg: Config,
    repeat: int = 1,
    out: str | None = None,
    headless: bool = True,
    allow_degraded: bool = False,
) -> int:
    cfg.browser.headless = headless
    cfg.policy.autonomy = "normal"
    print(f"\nVibeBot benchmark - {len(SUITE)} goals"
          f"{f', {repeat} rounds' if repeat > 1 else ''}"
          f"  (decider={cfg.decider.backend}, llm={cfg.llm.model},"
          f" page_text_chars={cfg.decider.page_text_chars})\n")

    report = asyncio.run(run_suite(cfg, SUITE, repeat, allow_degraded))

    print("\n" + "-" * 104)
    for result in report.results:
        print(result.row())
    print("-" * 104)
    summary = report.summary()
    print(
        f"\npassed {summary['passed']}/{summary['tasks']} ({summary['pass_rate']:.0%})  "
        f"median {summary['median_steps']:.0f} steps, {summary['median_seconds']:.0f}s per goal  "
        f"laya {summary['laya_share']:.0%} of steps  {summary['llm_calls']} llm calls  "
        f"{summary['total_seconds']:.0f}s total\n"
    )
    if out:
        Path(out).write_text(
            json.dumps(
                {"summary": summary, "config": cfg.to_dict(),
                 "results": [r.__dict__ for r in report.results]},
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        print(f"written to {out}\n")
    return 0 if summary["passed"] == summary["tasks"] else 1
