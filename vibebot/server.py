"""Local web UI: type a goal in your browser, watch the agent work, answer it.

One process owns one browser session. Multiple UI tabs can watch the same run.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import socket
import threading
import time
import webbrowser
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from .agent import Agent
from .config import Config

log = logging.getLogger(__name__)
UI_DIR = Path(__file__).parent / "ui"


class Hub:
    """Fan-out of agent events to every connected UI, with a short replay buffer."""

    def __init__(self, limit: int = 400):
        self.clients: set[WebSocket] = set()
        self.backlog: list[dict[str, Any]] = []
        self.limit = limit

    async def publish(self, event: dict[str, Any]) -> None:
        # Images are big and stale within a second; keep them out of the replay
        # buffer, and never buffer live frames at all.
        if event.get("type") != "frame":
            self.backlog.append({k: v for k, v in event.items() if k != "screenshot"})
            del self.backlog[: max(0, len(self.backlog) - self.limit)]
        for client in list(self.clients):
            try:
                await client.send_json(event)
            except Exception:  # noqa: BLE001 - a dead socket must not stop the run
                self.clients.discard(client)


def _data_url(png: bytes) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="VibeBot")
    hub = Hub()
    agent = Agent(cfg, on_event=hub.publish)
    state: dict[str, Any] = {"task": None, "started": False, "live": None}
    token = cfg.server.token or ""

    async def ensure_started() -> None:
        if not state["started"]:
            state["started"] = True
            await agent.start()

    async def stream_frames() -> None:
        """Push a screenshot every 1/live_fps seconds while someone is watching.

        Deliberately decoupled from the step loop: the agent can be mid-think,
        paused, or waiting on you, and the view keeps updating. Read-only — it
        never touches the page, so it cannot disturb a run.
        """
        interval = 1.0 / max(0.2, cfg.server.live_fps)
        misses = 0
        try:
            while hub.clients:
                started = time.perf_counter()
                png = await agent.browser.screenshot()
                if png:
                    misses = 0
                    tabs = agent.browser.tabs()
                    await hub.publish(
                        {
                            "type": "frame",
                            "image": _data_url(png),
                            "tabs": [{"index": t.index, "url": t.url, "active": t.active} for t in tabs],
                        }
                    )
                else:
                    misses += 1
                    if misses > 10:  # browser gone — stop burning cycles
                        break
                # Capturing a PNG costs 100-200ms; sleep for what is left of the
                # frame budget so the configured fps is roughly what you get.
                await asyncio.sleep(max(0.02, interval - (time.perf_counter() - started)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the viewer must never kill a run
            log.warning("live view stopped: %s", exc)
        finally:
            state["live"] = None

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (UI_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    async def status() -> JSONResponse:
        return JSONResponse(
            {
                "running": agent.running,
                "paused": agent.paused,
                "awaiting_user": agent.awaiting_user,
                "live": state.get("live") is not None,
                "goal": agent.goal,
                "decider": agent.decider.name,
                "llm": agent.llm.name,
                "config": cfg.to_dict(),
            }
        )

    async def do_quit() -> None:
        """Bring the whole thing down tidily: stop the run, close Chromium,
        release the model, then let uvicorn exit."""
        await hub.publish({"type": "quitting"})
        agent.stop()
        task = state.get("task")
        if task:
            task.cancel()
        request_shutdown = getattr(app.state, "request_shutdown", None)
        if request_shutdown:
            request_shutdown()

    @app.post("/api/quit")
    async def quit_endpoint(request: Request) -> JSONResponse:
        """So `Stop VibeBot` can ask politely before it reaches for taskkill."""
        if token and not secrets.compare_digest(request.query_params.get("token", ""), token):
            return JSONResponse({"error": "bad token"}, status_code=401)
        await do_quit()
        return JSONResponse({"stopping": True})

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        agent.stop()
        for key in ("task", "live"):
            running = state.get(key)
            if running:
                running.cancel()
        await agent.shutdown()

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        if token and not secrets.compare_digest(socket.query_params.get("token", ""), token):
            await socket.close(code=4401)
            return
        await socket.accept()
        hub.clients.add(socket)
        try:
            await socket.send_json({"type": "backlog", "events": hub.backlog[-120:]})
            await ensure_started()
            while True:
                message = await socket.receive_json()
                await _handle(message)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            log.warning("ws error: %s", exc)
        finally:
            hub.clients.discard(socket)

    async def _handle(message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "run":
            goal = (message.get("goal") or "").strip()
            if not goal:
                return
            if agent.running:
                await hub.publish({"type": "error", "message": "Already running — stop it first."})
                return
            await ensure_started()
            state["task"] = asyncio.create_task(agent.run(goal))
        elif kind == "answer":
            if not agent.provide_answer(str(message.get("text", ""))):
                await hub.publish({"type": "error", "message": "Nothing was waiting for an answer."})
        elif kind == "stop":
            agent.stop()
            task = state.get("task")
            if task:
                task.cancel()
            await hub.publish({"type": "run_end", "status": "stopped", "summary": "Stopped by user."})
        elif kind == "live":
            want = bool(message.get("value", True))
            running = state.get("live")
            if want and not running:
                await ensure_started()
                state["live"] = asyncio.create_task(stream_frames())
            elif not want and running:
                running.cancel()
                state["live"] = None
            await hub.publish({"type": "live", "value": want, "fps": cfg.server.live_fps})
        elif kind == "pause":
            agent.paused = bool(message.get("value", True))
            await hub.publish({"type": "paused", "value": agent.paused})
        elif kind == "goto":
            await agent.browser.goto(str(message.get("url", "")))
            await hub.publish({"type": "log", "message": f"opened {message.get('url')}"})
        elif kind == "quit":
            # Closing VibeBot from the thing you are already looking at, rather
            # than hunting for the console window it was started from.
            await do_quit()

    return app


def _websocket_support() -> str:
    """Which WebSocket implementation uvicorn can use, if any.

    The UI is a WebSocket client, and plain `uvicorn` ships without a WS
    implementation: it serves the page, refuses the upgrade with "Unsupported
    upgrade request", and the UI sits there dead. Worth catching at startup
    rather than in the browser console.
    """
    for module in ("websockets", "wsproto"):
        try:
            __import__(module)
        except ImportError:
            continue
        return module
    return ""


def serve(cfg: Config) -> None:
    import uvicorn

    if not _websocket_support():
        raise SystemExit(
            "No WebSocket library installed, so the UI would load and then never connect.\n"
            "  Fix: pip install 'uvicorn[standard]'   (or: pip install websockets)"
        )

    if cfg.server.host not in {"127.0.0.1", "localhost"} and not cfg.server.token:
        raise SystemExit(
            "Refusing to listen on a non-local address without server.token set "
            "(or VIBEBOT_SERVER_TOKEN). Anyone who reaches this port can drive your browser."
        )

    url = f"http://{cfg.server.host}:{cfg.server.port}"
    if cfg.server.token:
        url += f"?token={cfg.server.token}"

    app = create_app(cfg)
    server = uvicorn.Server(
        uvicorn.Config(app, host=cfg.server.host, port=cfg.server.port, log_level="warning")
    )
    # The Quit button needs a way to bring uvicorn down cleanly, and
    # uvicorn.run() gives you no handle on the server it creates.
    app.state.request_shutdown = lambda: setattr(server, "should_exit", True)

    # flush= throughout: Python block-buffers stdout when it is not a console,
    # so without it the banner only appears when the server finally exits.
    print(flush=True)
    print("  VibeBot is starting...", flush=True)
    print(f"  UI  ->  {url}", flush=True)
    print(flush=True)
    print("  To stop it: press Quit in the page, run 'Stop VibeBot', or close", flush=True)
    print("  this window.", flush=True)
    print(flush=True)

    if cfg.server.open_browser:
        _open_when_ready(url, cfg.server.host, cfg.server.port)

    with _pid_file(cfg.server.pid_file):
        server.run()

    print(flush=True)
    print("  VibeBot has stopped. You can close this window.", flush=True)
    print(flush=True)


@contextmanager
def _pid_file(path: str):
    """Record the PID while the server runs, so stopping it is a double-click.

    Best effort throughout: a path that cannot be written is a worse reason to
    refuse to start than it is a problem.
    """
    target = Path(path).expanduser()
    written = False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(os.getpid()), encoding="ascii")
        written = True
    except OSError as exc:
        log.debug("could not write %s: %s", target, exc)
    try:
        yield
    finally:
        if written:
            try:
                target.unlink()
            except OSError:
                pass


def _open_when_ready(url: str, host: str, port: int, timeout: float = 30.0) -> None:
    """Open the UI once the port answers - opening it sooner gives a dead tab."""

    def wait_then_open() -> None:
        target = "127.0.0.1" if host == "0.0.0.0" else host  # noqa: S104 - probe only
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with socket.socket() as probe:
                probe.settimeout(0.4)
                if probe.connect_ex((target, port)) == 0:
                    break
            time.sleep(0.2)
        else:
            log.warning("server did not come up within %ss; not opening a browser", timeout)
            return
        try:
            webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001 - a headless box has no browser
            log.debug("could not open a browser: %s", exc)

    threading.Thread(target=wait_then_open, name="open-ui", daemon=True).start()
