"""CLI: `vibebot serve` for the browser UI, `vibebot run "<goal>"` for headless use."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from .config import Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("vibebot", description="Local-first browser agent.")
    parser.add_argument("-c", "--config", help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    serve_cmd = sub.add_parser("serve", help="run the web UI (default)")
    serve_cmd.add_argument("--host")
    serve_cmd.add_argument("--port", type=int)

    run_cmd = sub.add_parser("run", help="run one goal from the terminal")
    run_cmd.add_argument("goal", nargs="+")
    run_cmd.add_argument("--headless", action="store_true")
    run_cmd.add_argument("--max-steps", type=int)
    run_cmd.add_argument(
        "--yolo", action="store_true", help="never ask for confirmation (you have been warned)"
    )

    sub.add_parser("doctor", help="check that Playwright, Laya and the LLM are usable")

    bench_cmd = sub.add_parser(
        "bench", help="score the agent against a fixed set of goals"
    )
    bench_cmd.add_argument("--repeat", type=int, default=1, help="rounds of the whole suite")
    bench_cmd.add_argument("--out", help="write the numbers to this JSON file")
    bench_cmd.add_argument(
        "--show", action="store_true", help="watch it work instead of running headless"
    )
    bench_cmd.add_argument("--page-text-chars", type=int, help="override decider.page_text_chars")
    bench_cmd.add_argument("--model", help="override llm.model")
    bench_cmd.add_argument(
        "--allow-degraded", action="store_true",
        help="run even if Laya could not load (measures the heuristic fallback)",
    )
    bench_cmd.add_argument(
        "--suite", choices=["shop", "web"], default="shop",
        help="shop = local deterministic test site (default); web = real websites",
    )
    bench_cmd.add_argument("--only", nargs="+", help="run only goals whose name contains these")
    bench_cmd.add_argument(
        "--llm-only", action="store_true",
        help="never let the fast decider act, to measure what it is worth",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        cfg = Config.load(args.config)
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "run":
        if args.headless:
            cfg.browser.headless = True
        if args.max_steps:
            cfg.policy.max_steps = args.max_steps
        if args.yolo:
            cfg.policy.autonomy = "yolo"
        return asyncio.run(_run_once(cfg, " ".join(args.goal)))

    if args.command == "doctor":
        return asyncio.run(_doctor(cfg))

    if args.command == "bench":
        from .bench import bench

        if args.page_text_chars is not None:
            cfg.decider.page_text_chars = args.page_text_chars
        if args.model:
            cfg.llm.model = args.model
        if args.llm_only:
            cfg.decider.backend = "off"
        return bench(
            cfg, repeat=args.repeat, out=args.out, headless=not args.show,
            allow_degraded=args.allow_degraded, suite=args.suite, only=args.only,
        )

    if args.host:
        cfg.server.host = args.host
    if getattr(args, "port", None):
        cfg.server.port = args.port
    from .server import serve

    serve(cfg)
    return 0


async def _run_once(cfg: Config, goal: str) -> int:
    from .agent import Agent

    loop = asyncio.get_running_loop()

    async def printer(event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "verdict":
            print(f"  laya: {event['target']} {event['target_confidence']:.0%} "
                  f"op={event['operation']} ({event['latency_ms']}ms)")
        elif kind == "thinking":
            print(f"  -> llm: {event['note']}")
        elif kind == "acted":
            print(f"  [{event['source']}] {event['action']} -> {event['outcome']}")
        elif kind == "observation":
            print(f"\nstep {event['step']}: {event['title'][:70]} ({event['url'][:80]})")
        elif kind in {"error", "blocked"}:
            print(f"  !! {event.get('message') or event.get('reason')}")

    agent = Agent(cfg, on_event=printer)

    # Terminal answers for ask_user, read off the main thread.
    async def ask_via_stdin(question: str) -> str:
        print(f"\n?? {question}\n> ", end="", flush=True)
        return (await loop.run_in_executor(None, sys.stdin.readline)).strip()

    original = agent.ask_user

    async def patched(question: str) -> str:
        answer = await ask_via_stdin(question)
        agent._history.append(f"you said: {answer}")  # noqa: SLF001 - same package
        return answer or "(no answer)"

    agent.ask_user = patched  # type: ignore[method-assign]
    del original

    await agent.start()
    try:
        result = await agent.run(goal)
    finally:
        await agent.shutdown()

    print(f"\n=== {result['status']} ===\n{result['summary']}\ntrace: {result['trace']}")
    return 0 if result["status"] == "done" else 1


async def _doctor(cfg: Config) -> int:
    ok = True

    from .sysmem import snapshot

    memory = snapshot()
    if memory is None:
        print("memory        : unknown on this platform")
    else:
        tight = memory.headroom_mb < cfg.decider.min_headroom_mb
        print(f"memory        : {'TIGHT' if tight else 'ok'} -> {memory.short()}")
        if tight:
            print(f"                below decider.min_headroom_mb ({cfg.decider.min_headroom_mb:,} MB); "
                  "Laya will be skipped and every step will go to the LLM.")
            ok = False

    try:
        import playwright  # noqa: F401,PLC0415

        print("playwright    : installed")
    except ImportError:
        print("playwright    : MISSING  -> pip install playwright && playwright install chromium")
        ok = False

    from .server import _websocket_support

    ws = _websocket_support()
    if ws:
        print(f"websockets    : {ws} (UI can connect)")
    else:
        print("websockets    : MISSING  -> pip install 'uvicorn[standard]'  (UI loads but never connects)")
        ok = False

    from .deciders import build_decider

    decider = build_decider(cfg.decider)
    degraded = getattr(decider, "degraded_from", None)
    if degraded:
        print(f"decider       : DEGRADED -> using heuristic because {degraded[0]}: {degraded[1]}")
        ok = False
    else:
        print(f"decider       : {decider.name} ready")

    from .llm import build_llm

    llm = build_llm(cfg.llm)
    llm_ok, message = await llm.ping()  # type: ignore[attr-defined]
    print(f"llm           : {'ready' if llm_ok else 'PROBLEM'} -> {message}")
    await llm.close()
    ok = ok and llm_ok

    try:
        from .browser import BrowserSession

        session = BrowserSession(cfg.browser)
        await session.start()
        await session.goto("https://example.com")
        obs = await session.observe(screenshot=False)
        print(f"browser       : ok ({len(obs.elements)} elements on example.com)")
        await session.close()
    except Exception as exc:  # noqa: BLE001
        print(f"browser       : FAILED -> {exc}")
        ok = False

    print("\nall good." if ok else "\nsomething above needs fixing.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
