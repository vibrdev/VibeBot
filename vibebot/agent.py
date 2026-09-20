"""The control loop.

One step looks like this:

    observe  ->  rank candidates  ->  Laya decides  ->  gate  ->  act

The gate is the whole design. Laya answers in a few hundred milliseconds on CPU
with a calibrated probability distribution, so "not sure" becomes a first-class
outcome rather than a confident wrong click:

    Laya picks an element, p and margin clear   -> do it            (free, fast)
    Laya picks "ask_llm", or the race is close  -> local LLM        (slow, smart)
    Action looks consequential                  -> ask you first    (safety)
    LLM says ask_user / we are stuck in a loop  -> ask you          (last resort)

Everything is logged so you can see which layer decided what.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

from .browser import BrowserSession
from .config import Config
from .deciders import Verdict, build_decider
from .llm import build_llm
from .ranking import rank_candidates, text_candidates
from .schema import GLOBAL_OPS, TERMINAL_OPS, Action, Observation, StepRecord
from .trace import Trace

log = logging.getLogger(__name__)

EventSink = Callable[[dict[str, Any]], Awaitable[None]]


class Agent:
    def __init__(self, cfg: Config, on_event: EventSink | None = None):
        self.cfg = cfg
        self.on_event = on_event or _noop
        self.browser = BrowserSession(cfg.browser)
        self.decider = build_decider(cfg.decider)
        self.llm = build_llm(cfg.llm)

        self.goal = ""
        self.running = False
        self.paused = False
        self._stop = asyncio.Event()
        self._answer: asyncio.Future[str] | None = None
        self._llm_calls = 0
        self._history: list[str] = []
        self._fingerprints: list[str] = []

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> dict[str, Any]:
        await self.browser.start()
        status = {
            "decider": self.decider.name,
            "degraded_from": getattr(self.decider, "degraded_from", None),
            "llm": self.llm.name,
        }
        ok, message = await self.llm.ping()  # type: ignore[attr-defined]
        status["llm_ready"] = ok
        status["llm_message"] = message
        await self.emit("status", **status)
        return status

    async def shutdown(self) -> None:
        await self.browser.close()
        await self.llm.close()

    def stop(self) -> None:
        self._stop.set()
        if self._answer and not self._answer.done():
            self._answer.cancel()

    # ------------------------------------------------------------------ runs

    async def run(self, goal: str) -> dict[str, Any]:
        """Pursue one goal. Returns {"status": ..., "summary": ...}."""
        self.goal = goal
        self.running = True
        self._stop.clear()
        self._llm_calls = 0
        self._history = []
        self._fingerprints = []

        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        trace = Trace(self.cfg.trace_dir, run_id)
        trace.write({"type": "run_start", "goal": goal, "config": self.cfg.to_dict()})
        await self.emit("run_start", goal=goal, run_id=run_id)

        status, summary = "max_steps", "Ran out of steps before finishing."
        try:
            status, summary = await self._loop(trace)
        except asyncio.CancelledError:
            status, summary = "stopped", "Stopped by user."
            raise
        except Exception as exc:  # noqa: BLE001 - surface crashes in the UI
            log.exception("run failed")
            status, summary = "error", f"{type(exc).__name__}: {exc}"
        finally:
            self.running = False
            trace.finish(status, summary)
            await self.emit("run_end", status=status, summary=summary, run_id=run_id)
        return {"status": status, "summary": summary, "run_id": run_id, "trace": str(trace.path)}

    async def _loop(self, trace: Trace) -> tuple[str, str]:
        policy = self.cfg.policy
        stalls = 0

        for step in range(1, policy.max_steps + 1):
            if self._stop.is_set():
                return "stopped", "Stopped by user."
            while self.paused and not self._stop.is_set():
                await asyncio.sleep(0.2)

            started = time.perf_counter()
            obs = await self.browser.observe()
            obs.step = step

            blocked = self._domain_violation(obs.url)
            if blocked:
                await self.emit("blocked", url=obs.url, reason=blocked)
                answer = await self.ask_user(f"{blocked} Current page: {obs.url}. What now?")
                self._history.append(f"policy block at {obs.url}; user said: {answer}")
                continue

            candidates = rank_candidates(
                obs.elements, self.goal, self._history, limit=self.cfg.decider.max_candidates
            )
            shot = await self.browser.screenshot(highlight=[el.idx for el in candidates])
            obs.annotated_png = shot
            trace.save_shot(step, shot)
            await self.emit(
                "observation",
                step=step,
                url=obs.url,
                title=obs.title,
                candidates=[{"idx": el.idx, "label": el.label()} for el in candidates],
                screenshot=_data_url(shot),
                frames=1 + max((el.frame_id for el in obs.elements), default=0),
                tabs=[{"index": t.index, "url": t.url, "active": t.active} for t in obs.tabs],
            )

            verdict = await self.decider.decide(
                self.goal, obs, candidates, self._history, text_candidates(self.goal)
            )
            await self.emit("verdict", step=step, **verdict.to_json())

            action, escalated = await self._resolve(obs, candidates, verdict, stalls)

            if action.op in TERMINAL_OPS - {"fail"}:
                summary = action.text or "Done."
                self._record(trace, step, obs, candidates, verdict, action, "finished", escalated, started)
                return "done", summary
            if action.op == "fail":
                self._record(trace, step, obs, candidates, verdict, action, "failed", escalated, started)
                return "failed", action.text or "The agent gave up."

            before = obs.fingerprint()
            outcome = await self._execute(action, obs)
            await self.emit("acted", step=step, action=action.describe(obs), outcome=outcome, source=action.source)
            self._history.append(f"{action.describe(obs)} -> {outcome}")
            self._record(trace, step, obs, candidates, verdict, action, outcome, escalated, started)

            after = (await self.browser.observe(screenshot=False)).fingerprint()
            if after == before:
                stalls += 1
                if stalls >= policy.stall_limit * 2:
                    answer = await self.ask_user(
                        "I am stuck — the page has not changed for several steps. "
                        "What should I do? (You can also just take over in the browser window.)"
                    )
                    self._history.append(f"user unstuck me: {answer}")
                    stalls = 0
                elif stalls >= policy.stall_limit:
                    self._history.append("warning: page did not change, previous approach is not working")
            else:
                stalls = 0

        return "max_steps", f"Hit the {policy.max_steps}-step limit. Last page: {(await self.browser.observe(screenshot=False)).url}"

    # -------------------------------------------------------------- decisions

    async def _resolve(
        self,
        obs: Observation,
        candidates: list,
        verdict: Verdict,
        stalls: int,
    ) -> tuple[Action, str | None]:
        """Turn a Verdict into a concrete Action, escalating as needed."""
        cfg = self.cfg.decider
        valid = {f"e{el.idx}": el for el in candidates}

        # 1. Laya thinks we are finished. Cheap to check, expensive to get wrong,
        #    so have the LLM confirm before we celebrate.
        if verdict.done_p >= cfg.done_confidence:
            confirmed = await self._escalate(
                obs, candidates, "Laya believes the goal is complete. Confirm with 'done' "
                "(put the result in text) or continue with another action.",
            )
            if confirmed.op in TERMINAL_OPS - {"fail"}:
                return _from_llm(confirmed), "llm"

        # 2. Fast path: Laya is confident about a real element.
        confident = verdict.is_confident(cfg.accept_probability, cfg.accept_margin)
        if not verdict.wants_llm and verdict.target in valid and confident:
            element = valid[verdict.target]
            op = verdict.operation
            is_field = element.tag in {"input", "textarea"} or element.role in {
                "searchbox", "textbox", "combobox",
            }
            if op == "type" and not verdict.text:
                op = "click" if not is_field else op
            if op == "type" and not verdict.text:
                return await self._llm_step(obs, candidates, "Laya wants to type here but has no text.")
            action = Action(
                op=op,
                element_idx=element.idx,
                text=verdict.text if op in {"type", "select"} else "",
                reason=f"laya p={verdict.p_top:.0%} margin={verdict.margin:.0%}",
                source="laya",
                confidence=verdict.target_confidence,
                risky=verdict.risky_p,
            )
            return await self._risk_gate(action, obs), None

        # 3a. "It's in another tab." With exactly two open there is nothing to
        #     reason about; with more, the LLM picks which one.
        if verdict.target == "switch_tab" and confident and len(obs.tabs) == 2:
            other = 1 - obs.active_tab
            return (
                Action(
                    op="switch_tab",
                    text=str(other),
                    reason=f"laya p={verdict.p_top:.0%}",
                    source="laya",
                    confidence=verdict.p_top,
                ),
                None,
            )

        # 3b. Cheap navigation escapes — no need to wake the LLM for these.
        if verdict.target in {"scroll", "back"} and confident:
            return (
                Action(
                    op=verdict.target,
                    text="down" if verdict.target == "scroll" else "",
                    reason=f"laya p={verdict.p_top:.0%}",
                    source="laya",
                    confidence=verdict.target_confidence,
                ),
                None,
            )

        # 4. Everything else is the LLM's problem.
        note = (
            "Laya deferred this step to you."
            if verdict.wants_llm
            else (
                f"Laya's best guess was {verdict.target} at p={verdict.p_top:.0%} "
                f"(margin {verdict.margin:.0%}) — below the acceptance gate."
            )
        )
        if stalls >= self.cfg.policy.stall_limit:
            note += " The last few actions did not change the page; try something different."
        return await self._llm_step(obs, candidates, note)

    async def _llm_step(self, obs: Observation, candidates: list, note: str) -> tuple[Action, str]:
        if self._llm_calls >= self.cfg.policy.max_llm_calls:
            answer = await self.ask_user(
                f"I have used my {self.cfg.policy.max_llm_calls} LLM calls for this run. "
                "Tell me what to do next, or say 'stop'."
            )
            return Action(op="wait", text="1", reason=f"user: {answer}", source="user"), "user"

        decision = await self._escalate(obs, candidates, note)
        if decision.op == "ask_user":
            question = decision.text or "I am not sure how to continue. What should I do?"
            answer = await self.ask_user(question)
            self._history.append(f"asked: {question} / you said: {answer}")
            return Action(op="wait", text="0.5", reason=f"user: {answer}", source="user"), "user"

        action = _from_llm(decision)
        if action.op not in GLOBAL_OPS and action.element_idx is None:
            return Action(op="scroll", text="down", reason="llm gave no target", source="policy"), "llm"
        return await self._risk_gate(action, obs), "llm"

    async def _escalate(self, obs: Observation, candidates: list, note: str):
        self._llm_calls += 1
        await self.emit("thinking", note=note, calls=self._llm_calls)
        decision = await self.llm.decide(
            self.goal, obs, candidates, self._history, note, obs.annotated_png if self.cfg.llm.vision else None
        )
        await self.emit(
            "llm",
            op=decision.op,
            element_idx=decision.element_idx,
            text=decision.text[:200],
            reason=decision.reason,
            confidence=decision.confidence,
        )
        return decision

    async def _risk_gate(self, action: Action, obs: Observation) -> Action:
        """Ask before anything that spends money, sends, deletes or logs in."""
        policy = self.cfg.policy
        if policy.autonomy == "yolo":
            return action

        label = ""
        if action.element_idx is not None:
            match = next((el for el in obs.elements if el.idx == action.element_idx), None)
            label = match.label().lower() if match else ""
        keyword_hit = any(word in label for word in policy.risky_keywords)
        risky = action.risky >= self.cfg.decider.risk_threshold or keyword_hit

        if action.op == "type" and policy.never_type_into_password and action.element_idx is not None:
            if await self.browser.element_is_password(action.element_idx):
                answer = await self.ask_user(
                    "That is a password field. I will not type into it on my own — "
                    "type it yourself in the browser window and reply 'ok', or reply with what to do instead."
                )
                return Action(op="wait", text="1", reason=f"user: {answer}", source="user")

        if policy.autonomy == "guided" or risky:
            answer = await self.ask_user(
                f"About to: {action.describe(obs)}\nReason: {action.reason}\n"
                "Reply 'ok' to go ahead, or tell me what to do instead."
            )
            if answer.strip().lower() in {"ok", "yes", "y", "go", "do it", "sure"}:
                return action
            return Action(op="wait", text="0.5", reason=f"user redirected: {answer}", source="user")
        return action

    # ------------------------------------------------------------- execution

    async def _execute(self, action: Action, obs: Observation) -> str:
        try:
            if action.op == "click":
                return await self.browser.click(int(action.element_idx))  # type: ignore[arg-type]
            if action.op == "type":
                return await self.browser.type_text(int(action.element_idx), action.text)  # type: ignore[arg-type]
            if action.op == "select":
                return await self.browser.select(int(action.element_idx), action.text)  # type: ignore[arg-type]
            if action.op == "scroll":
                return await self.browser.scroll(action.text or "down")
            if action.op == "navigate":
                return await self.browser.goto(action.text)
            if action.op == "back":
                return await self.browser.back()
            if action.op == "wait":
                return await self.browser.wait(float(action.text or 1))
            if action.op == "switch_tab":
                return await self.browser.switch_tab(_as_index(action.text, obs.active_tab))
            if action.op == "close_tab":
                return await self.browser.close_tab(_as_index(action.text, obs.active_tab))
            return f"no-op ({action.op})"
        except Exception as exc:  # noqa: BLE001 - a failed action is data, not a crash
            log.warning("action failed: %s", exc)
            return f"FAILED: {type(exc).__name__}: {str(exc)[:160]}"

    # ------------------------------------------------------------ user input

    async def ask_user(self, question: str) -> str:
        """Block the run until you answer in the UI (or the CLI)."""
        loop = asyncio.get_running_loop()
        self._answer = loop.create_future()
        await self.emit("question", question=question)
        try:
            answer = await asyncio.wait_for(self._answer, timeout=self.cfg.policy.user_reply_timeout_s)
        except asyncio.TimeoutError:
            answer = "(no answer — carry on as best you can)"
        except asyncio.CancelledError:
            answer = "(stopped)"
        finally:
            self._answer = None
        await self.emit("answer", answer=answer)
        return answer

    def provide_answer(self, answer: str) -> bool:
        if self._answer and not self._answer.done():
            self._answer.set_result(answer)
            return True
        return False

    @property
    def awaiting_user(self) -> bool:
        return self._answer is not None and not self._answer.done()

    # ---------------------------------------------------------------- plumbing

    async def emit(self, kind: str, **payload: Any) -> None:
        await self.on_event({"type": kind, "ts": time.time(), **payload})

    def _domain_violation(self, url: str) -> str | None:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return None
        policy = self.cfg.policy
        if any(host == bad or host.endswith("." + bad) for bad in policy.block_domains):
            return f"{host} is on the blocklist."
        if policy.allow_domains and not any(
            host == ok or host.endswith("." + ok) for ok in policy.allow_domains
        ):
            return f"{host} is outside the allowlist."
        return None

    def _record(
        self,
        trace: Trace,
        step: int,
        obs: Observation,
        candidates: list,
        verdict: Verdict,
        action: Action,
        outcome: str,
        escalated: str | None,
        started: float,
    ) -> None:
        record = StepRecord(
            step=step,
            url=obs.url,
            title=obs.title,
            goal=self.goal,
            candidates=[f"e{el.idx}: {el.label()}" for el in candidates],
            laya={**verdict.to_json(), "probabilities": verdict.raw},
            action={
                "op": action.op,
                "element_idx": action.element_idx,
                "text": action.text[:200],
                "reason": action.reason,
                "source": action.source,
            },
            outcome=outcome,
            escalated_to=escalated,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        trace.write({"type": "step", **record.to_json()})


def _as_index(text: str, fallback: int) -> int:
    """Tab index out of whatever the model put in `text` ("2", "tab 2", "")."""
    import re

    match = re.search(r"-?\d+", text or "")
    return int(match.group()) if match else fallback


def _from_llm(decision) -> Action:  # noqa: ANN001
    return Action(
        op=decision.op,
        element_idx=decision.element_idx,
        text=decision.text,
        reason=decision.reason or "llm",
        source="llm",
        confidence=decision.confidence,
    )


def _data_url(png: bytes | None) -> str | None:
    if not png:
        return None
    import base64

    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


async def _noop(_: dict[str, Any]) -> None:
    return None
