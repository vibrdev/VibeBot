"""An exam for the fast decider on the one job plan mode gives it.

    vibebot bench --executor            # Laya, and the label-matching baseline

In plan mode the fast decider is asked "which element is this step?" and
nothing else. The full benchmark measures that only indirectly - the LLM's
wording, its plans and its mistakes are all mixed in, and a fast decider may
only act a dozen times in ten goals. This asks the question directly, on real
pages of the benchmark shop, with a known right element for every step.

Half the steps are phrased the way the planner is told to write them
("click 'Price: lowest first'"); half are paraphrased ("sort so the cheapest
listings come first"). String matching should ace the first half and fail the
second. A learned decider earns its keep only if it does better than that on
the paraphrases without being worse on the rest.

Each answer is one of: right (acted, on the right element), wrong (acted,
confidently, on another one - the expensive failure), or deferred (not
confident, so the LLM would have been asked - slow but safe).
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, replace

from .agent import _quoted
from .config import Config
from .ranking import rank_candidates


@dataclass(frozen=True)
class Case:
    path: str
    step: str
    expect: str
    """The right element's visible label (or, for a text box, its name)."""
    phrased: str
    """"planner" - written the way the planner is asked to; "paraphrase"."""


LISTING = "MacBook Pro 16in M1 Max 64GB 2TB"

CASES: list[Case] = [
    Case("/search?q=macbook", "click 'Price: lowest first'", "Price: lowest first", "planner"),
    Case("/search?q=macbook", "sort so the cheapest listings come first", "Price: lowest first", "paraphrase"),
    Case("/search?q=macbook", "click 'Used'", "Used", "planner"),
    Case("/search?q=macbook", "only show second-hand listings", "Used", "paraphrase"),
    Case("/search?q=macbook", "click '96'", "96", "planner"),
    Case("/search?q=macbook", "filter to machines with 96GB of memory", "96", "paraphrase"),
    Case("/search?q=macbook", "click 'M3'", "M3", "planner"),
    Case("/search?q=macbook", "narrow it down to M3 chips", "M3", "paraphrase"),
    Case("/search?q=macbook", "click 'Refurbished'", "Refurbished", "planner"),
    Case("/search?q=macbook", "show only refurbished items", "Refurbished", "paraphrase"),
    Case("/search?q=macbook", "click 'Next page'", "Next page", "planner"),
    Case("/search?q=macbook", "go on to the following page of results", "Next page", "paraphrase"),
    Case("/search?q=macbook", f"click '{LISTING}'", LISTING, "planner"),
    Case("/search?q=macbook", "open the 16 inch M1 Max with 64GB", LISTING, "paraphrase"),
    Case("/", "click 'all MacBook listings'", "all MacBook listings", "planner"),
    Case("/", "browse every MacBook for sale", "all MacBook listings", "paraphrase"),
    Case("/", "click 'Help & returns'", "Help & returns", "planner"),
    Case("/", "find the page about returning an item", "Help & returns", "paraphrase"),
    Case("/item/m2max-96-a", "click 'Add to cart'", "Add to cart", "planner"),
    Case("/item/m2max-96-a", "put this laptop in the basket", "Add to cart", "paraphrase"),
]


def _label(el) -> str:  # noqa: ANN001
    return " ".join((el.text or el.name or el.placeholder or "").lower().split())


async def _exam(cfg: Config, backends: list[str]) -> dict[str, list[dict]]:
    from .benchsite import start
    from .browser import BrowserSession
    from .deciders import build_decider

    site = start()
    browser = BrowserSession(cfg.browser)
    deciders = {}
    for name in backends:
        cfg.decider.backend = name
        decider = build_decider(cfg.decider)
        if getattr(decider, "degraded_from", None):
            raise SystemExit(f"{name} could not load: {decider.degraded_from[1]}")
        deciders[name] = decider
    results: dict[str, list[dict]] = {name: [] for name in backends}
    try:
        await browser.start()
        for case in CASES:
            await browser.goto(site.base_url + case.path)
            obs = await browser.observe(screenshot=False)
            options = rank_candidates(obs.elements, case.step, [], limit=cfg.decider.max_candidates, url=obs.url)
            offered = any(_label(el) == case.expect.lower() for el in options)
            blind = replace(obs, text_digest="")
            for name, decider in deciders.items():
                started = time.perf_counter()
                verdict = await decider.decide(case.step, blind, options, [], _quoted(case.step))
                seconds = time.perf_counter() - started
                picked = next((el for el in options if f"e{el.idx}" == verdict.target), None)
                confident = picked is not None and verdict.is_confident(
                    cfg.decider.accept_probability, cfg.decider.accept_margin
                )
                if not confident:
                    outcome = "deferred"
                elif _label(picked) == case.expect.lower():
                    outcome = "right"
                else:
                    outcome = "WRONG"
                results[name].append({
                    "case": case, "outcome": outcome, "seconds": seconds, "offered": offered,
                    "picked": _label(picked) if picked else verdict.target, "p": verdict.p_top,
                })
    finally:
        await browser.close()
        site.stop()
    return results


def executor_bench(cfg: Config, backends: list[str]) -> int:
    cfg.browser.headless = True
    print(f"\nExecutor exam - {len(CASES)} plan steps on the benchmark shop, "
          f"gate p>={cfg.decider.accept_probability} margin>={cfg.decider.accept_margin}\n")
    results = asyncio.run(_exam(cfg, backends))

    missing = [r["case"].step for r in next(iter(results.values())) if not r["offered"]]
    if missing:
        print("!! The right element was not among the options offered for:")
        for step in missing:
            print(f"     {step}")
        print()

    width = max(len(c.step) for c in CASES)
    header = f"  {'step':<{width}}  " + "  ".join(f"{name:<28}" for name in backends)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, case in enumerate(CASES):
        cells = []
        for name in backends:
            r = results[name][i]
            mark = {"right": "right", "WRONG": "WRONG", "deferred": "defer"}[r["outcome"]]
            detail = f"{mark} {r['p']:.2f}" + (f" -> {r['picked'][:12]}" if r["outcome"] == "WRONG" else "")
            cells.append(f"{detail:<28}")
        print(f"  {case.step:<{width}}  " + "  ".join(cells))

    print()
    for name in backends:
        rows = results[name]
        for phrased in ("planner", "paraphrase"):
            subset = [r for r in rows if r["case"].phrased == phrased]
            count = {k: sum(r["outcome"] == k for r in subset) for k in ("right", "WRONG", "deferred")}
            print(f"  {name:<6} {phrased:<10}  right {count['right']:>2}/{len(subset)}   "
                  f"wrong {count['WRONG']:>2}   deferred {count['deferred']:>2}")
        median = statistics.median(r["seconds"] for r in rows)
        print(f"  {name:<6} median {median * 1000:,.0f} ms per step\n")
    return 0
