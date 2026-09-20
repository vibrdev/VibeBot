"""Local web UI: type a goal in your browser, watch the agent work, answer it.

One process owns one browser session. Multiple UI tabs can watch the same run.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
        # Screenshots are big; keep them out of the replay buffer.
        self.backlog.append({k: v for k, v in event.items() if k != "screenshot"})
        del self.backlog[: max(0, len(self.backlog) - self.limit)]
        for client in list(self.clients):
            try:
                await client.send_json(event)
            except Exception:  # noqa: BLE001 - a dead socket must not stop the run
                self.clients.discard(client)


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="VibeBot")
    hub = Hub()
    agent = Agent(cfg, on_event=hub.publish)
    state: dict[str, Any] = {"task": None, "started": False}
    token = cfg.server.token or ""

    async def ensure_started() -> None:
        if not state["started"]:
            state["started"] = True
            await agent.start()

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
                "goal": agent.goal,
                "decider": agent.decider.name,
                "llm": agent.llm.name,
                "config": cfg.to_dict(),
            }
        )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        agent.stop()
        task = state.get("task")
        if task:
            task.cancel()
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
        elif kind == "pause":
            agent.paused = bool(message.get("value", True))
            await hub.publish({"type": "paused", "value": agent.paused})
        elif kind == "goto":
            await agent.browser.goto(str(message.get("url", "")))
            await hub.publish({"type": "log", "message": f"opened {message.get('url')}"})

    return app


def serve(cfg: Config) -> None:
    import uvicorn

    if cfg.server.host not in {"127.0.0.1", "localhost"} and not cfg.server.token:
        raise SystemExit(
            "Refusing to listen on a non-local address without server.token set "
            "(or VIBEBOT_SERVER_TOKEN). Anyone who reaches this port can drive your browser."
        )
    url = f"http://{cfg.server.host}:{cfg.server.port}"
    print(f"\n  VibeBot UI -> {url}{'?token=' + cfg.server.token if cfg.server.token else ''}\n")
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port, log_level="warning")
