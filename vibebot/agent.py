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
from urllib.parse import unquote, unquote_plus, urlparse, urlsplit, urlunsplit

from .browser import BrowserSession
from .config import Config
from .deciders import Verdict, build_decider
from .llm import build_llm
from .ranking import goal_terms, is_plausible, rank_candidates, text_candidates
from .schema import GLOBAL_OPS, TERMINAL_OPS, Action, Observation, StepRecord
from .sysmem import snapshot
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
        #: Every LLM exchange in the current step, so the trace can show what
        #: the model was asked and what it actually replied.
        self._step_llm: list[dict[str, Any]] = []
        #: Every question put to you in the current step, and your answer.
        #: Without this a run you sat and approved reads back as though you
        #: were never there.
        self._step_asks: list[dict[str, Any]] = []
        #: (url, op, element) triples Laya has already been allowed to try on
        #: this run. Laya has no memory between steps, so an exact repeat means
        #: a loop, not a decision: real runs clicked eBay's "Deals" six times
        #: running, then bounced ebay.com -> "My eBay" -> sign-in -> back ->
        #: ebay.com for the whole step budget. The second time round the same
        #: page wants the same action, the LLM gets the step instead.
        self._tried: set[tuple[str, str, int | None]] = set()
        #: The site the goal is actually about, set whenever you or the LLM
        #: chose where to go. Laya wandering off it is a wrong turn, not a plan.
        self._task_host = ""
        self._drifted = ""
        #: Set when you or the LLM just chose a page on purpose. Laya does not
        #: know that happened and will cheerfully undo it.
        self._just_navigated = False
        #: Pages where the LLM has already said "no, not finished". Laya keeps
        #: reporting done_p near 1.00 on a page of search results, and asking
        #: again every step costs a call and takes the step away from Laya.
        self._not_done: set[str] = set()
        #: url -> consecutive scrolls, so a scroll that is going nowhere can
        #: be named as such in the next prompt.
        self._scrolls: dict[str, int] = {}

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> dict[str, Any]:
        # Kick the weights off first: loading them takes about as long as
        # everything else in this method put together, and none of it needs
        # them, so the load is free if it happens alongside.
        warm = getattr(self.decider, "warm", None)
        if warm:
            warm()
        await self.browser.start()
        status = {
            "decider": self.decider.name,
            "degraded_from": getattr(self.decider, "degraded_from", None),
            "llm": self.llm.name,
        }
        ok, message = await self.llm.ping()  # type: ignore[attr-defined]
        status["llm_ready"] = ok
        status["llm_message"] = message

        memory = snapshot()
        if memory is not None:
            status["memory"] = memory.short()
            if memory.headroom_mb < self.cfg.decider.min_headroom_mb:
                log.warning("starting with little memory to spare: %s", memory.short())
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
        self._tried = set()
        self._task_host = ""
        self._drifted = ""
        self._just_navigated = False
        self._not_done = set()
        self._scrolls = {}

        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        trace = Trace(self.cfg.trace_dir, run_id)
        trace.start(goal, self.cfg.to_dict())
        trace.note(
            f"decider: {self.decider.name}"
            + (f" (DEGRADED from {getattr(self.decider, 'degraded_from', None)})"
               if getattr(self.decider, "degraded_from", None) else "")
            + f" | llm: {self.llm.name} {self.cfg.llm.model}"
        )
        await self.emit("run_start", goal=goal, run_id=run_id)

        status, summary = "max_steps", "Ran out of steps before finishing."
        try:
            status, summary = await self._loop(trace)
        except asyncio.CancelledError:
            status, summary = "stopped", "Stopped by user."
            raise
        except Exception as exc:  # noqa: BLE001 - surface crashes in the UI
            import traceback

            log.exception("run failed")
            status, summary = "error", f"{type(exc).__name__}: {exc}"
            # A crash used to leave nothing behind but a one-line summary.
            trace.write({"type": "error", "error": summary, "traceback": traceback.format_exc()})
            trace.note(f"!! CRASH {summary}")
            trace.note(traceback.format_exc())
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
            self._step_llm = []
            self._step_asks = []
            obs = await self.browser.observe()
            obs.step = step

            blocked = self._domain_violation(obs.url)
            if blocked:
                await self.emit("blocked", url=obs.url, reason=blocked)
                answer = await self.ask_user(f"{blocked} Current page: {obs.url}. What now?")
                self._history.append(f"policy block at {obs.url}; user said: {answer}")
                continue

            candidates = rank_candidates(
                obs.elements,
                self.goal,
                self._history,
                limit=self.cfg.decider.max_candidates,
                url=obs.url,
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
            if action.source == "laya":
                self._tried.add((_page_key(obs.url), action.op, action.element_idx))
            if action.op == "scroll":
                key = _page_key(obs.url)
                self._scrolls[key] = self._scrolls.get(key, 0) + 1
            else:
                self._scrolls.pop(_page_key(obs.url), None)
            await self.emit("acted", step=step, action=action.describe(obs), outcome=outcome, source=action.source)
            self._history.append(f"{action.describe(obs)} -> {outcome}")
            self._record(trace, step, obs, candidates, verdict, action, outcome, escalated, started)

            landed = await self.browser.observe(screenshot=False)
            host = (urlparse(landed.url).hostname or "").lower()
            self._just_navigated = action.source in {"llm", "user"} and action.op == "navigate"
            if action.source in {"llm", "user"} and host:
                # Wherever you or the LLM deliberately went is the task's site.
                self._task_host = host
            elif (
                action.source == "laya"
                and host
                and self._task_host
                and host != self._task_host
                and not _is_search_host(self._task_host)
            ):
                # Measured: on Wikipedia's front page Laya clicked a photo
                # caption at p=0.74, then its licence link at p=0.97, and the
                # run was on creativecommons.org two steps from the goal.
                self._drifted = (
                    f"That last click left {self._task_host} and landed on {host}, which is "
                    f"not what the goal is about. Get back on track."
                )
                self._history.append(f"wandered off {self._task_host} onto {host}")

            after = landed.fingerprint()
            if after == before:
                # Succeeding and achieving nothing is its own kind of dead end:
                # clicking the link for the page you are already on returns
                # "clicked element 3" forever. Bar it whoever chose it.
                self._tried.add((_page_key(obs.url), action.op, action.element_idx))
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

        # 0. We are somewhere Laya wandered to. It has no idea that happened.
        if self._drifted:
            note, self._drifted = self._drifted, ""
            return await self._llm_step(obs, candidates, note)

        # 0b. Somebody who can read just chose this page on purpose, so Laya
        #     does not get to walk straight back off it. It used to lose every
        #     step after a navigation to the LLM, which cost more than it
        #     saved: on a page of shopping results Laya wanted "For Sale" at
        #     p=0.91 and the LLM was asked instead, and clicked "Store".
        just_navigated, self._just_navigated = self._just_navigated, False

        # 1. Laya thinks we are finished. Cheap to check, expensive to get wrong,
        #    so have the LLM confirm before we celebrate.
        if verdict.done_p >= cfg.done_confidence and _page_key(obs.url) not in self._not_done:
            # Leading the witness here gets you agreement, not a check. Asked
            # to "confirm with done", qwen3.5:4b ended a run on a page of
            # Google results with "Goal confirmed: Found search results for
            # cheapest second hand MacBook Pro M4" - which is not the answer to
            # anything. So make it produce the value instead of a verdict.
            confirmed = await self._escalate(
                obs,
                candidates,
                "Laya thinks the goal is met. Only agree if the thing the goal asks for "
                "is on this page right now: if it wants a price, a number, a date or a "
                "name, reply 'done' with that exact value in text. A page that merely "
                "lists results, or links to where the answer might be, is not the answer. "
                "If the value is not here, do NOT reply 'done' or 'extract', and do not "
                "explain what is missing - reply with the next action (click, type, "
                "navigate or scroll) that gets to it.",
            )
            if confirmed.op in TERMINAL_OPS - {"fail"}:
                return _from_llm(confirmed), "llm"
            self._not_done.add(_page_key(obs.url))
            # It declined to call it finished and named something else to do.
            # Falling through to Laya's guess here threw that away and acted on
            # the worse-informed of the two: on a page of search results, Laya
            # kept clicking back to "Deals" and losing them.
            return await self._use_llm_decision(confirmed, obs, candidates)

        # 2. Fast path: Laya is confident about a real element.
        confident = verdict.is_confident(cfg.accept_probability, cfg.accept_margin)
        if not verdict.wants_llm and verdict.target in valid and confident:
            element = valid[verdict.target]
            if _offtopic_and_unsure(
                element, goal_terms(self.goal, obs.url), verdict.p_top, cfg.trust_confidence
            ):
                return await self._llm_step(
                    obs,
                    candidates,
                    f"Laya is only {verdict.p_top:.0%} sure about {element.label(60)}, and "
                    "nothing about it relates to the goal. Pick something that does.",
                )
            op, needs_llm = _coerce_operation(verdict.operation, element, verdict.text)
            if needs_llm:
                return await self._llm_step(obs, candidates, needs_llm)
            if (_page_key(obs.url), op, element.idx) in self._tried:
                return await self._llm_step(
                    obs,
                    candidates,
                    f"Laya wants to {op} {element.label(60)} again, and this page has "
                    "already had exactly that. It did not get us anywhere — pick "
                    "something else.",
                )
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
        other_tab = 1 - obs.active_tab if len(obs.tabs) == 2 else -1
        if (
            verdict.target == "switch_tab"
            and confident
            and other_tab >= 0
            # A blank tab has nothing in it to find. A run began with two
            # about:blank tabs left over from an earlier goal, and Laya spent
            # its first two steps switching between them at p=0.72 and 0.88.
            and obs.tabs[other_tab].url not in ("", "about:blank")
        ):
            other = other_tab
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

        # 3b. Cheap navigation escapes — no need to wake the LLM for these,
        #     unless we have already used this one here and come back round.
        if verdict.target in {"scroll", "back"} and confident:
            if verdict.target == "back" and just_navigated:
                # Measured: the LLM jumped straight to the Eiffel Tower article
                # and Laya pressed back at p=0.95 on the very next step.
                return await self._llm_step(
                    obs,
                    candidates,
                    "Laya wants to go back, but this page was chosen deliberately one "
                    "step ago. Work with what is on it.",
                )
            if (_page_key(obs.url), verdict.target, None) in self._tried:
                return await self._llm_step(
                    obs,
                    candidates,
                    f"Laya wants to {verdict.target} from this page again, which is "
                    "where we were last time round. Break the loop.",
                )
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
        if verdict.wants_llm:
            note = "Laya deferred this step to you."
        elif verdict.target == "done" and _page_key(obs.url) in self._not_done:
            # Reporting p=98% as "below the acceptance gate" was simply false:
            # the gate never looked at it, because this page had already been
            # asked about and the answer was no.
            note = (
                f"Laya says the goal is met ({verdict.p_top:.0%}), but you have already "
                "been asked about this page and said it is not finished. Do not answer "
                "again — choose the action that gets closer."
            )
        elif verdict.target == "done":
            # Laya leans towards "finished" but not far enough for the done
            # check. The note above used to be sent here too, claiming a prior
            # ask that never happened: on example.com, whose text holds the
            # answer, the LLM was told not to answer, wandered off to iana.org
            # and a benchmark goal failed on a page it had already solved.
            note = (
                f"Laya leans towards the goal already being met here ({verdict.p_top:.0%}) "
                "but is not sure. If what the goal asks for is on this page, reply 'done' "
                "with that exact value; otherwise choose the next action."
            )
        else:
            note = (
                f"Laya's best guess was {verdict.target} at p={verdict.p_top:.0%} "
                f"(margin {verdict.margin:.0%}) — below the acceptance gate."
            )
        if stalls >= self.cfg.policy.stall_limit:
            note += " The last few actions did not change the page; try something different."
        scrolls = self._scrolls.get(_page_key(obs.url), 0)
        if scrolls >= 3:
            # Told firmly enough not to answer without the value, the model
            # scrolled the same results page six times instead.
            note += (
                f" You have already scrolled this page {scrolls} times without getting"
                " closer; scrolling again will not help. Sort or filter the list, or"
                " open one of the entries."
            )
        return await self._llm_step(obs, candidates, note)

    async def _llm_step(self, obs: Observation, candidates: list, note: str) -> tuple[Action, str]:
        if self._llm_calls >= self.cfg.policy.max_llm_calls:
            answer = await self.ask_user(
                f"I have used my {self.cfg.policy.max_llm_calls} LLM calls for this run. "
                "Tell me what to do next, or say 'stop'."
            )
            return Action(op="wait", text="1", reason=f"user: {answer}", source="user"), "user"

        decision = await self._escalate(obs, candidates, note)
        return await self._use_llm_decision(decision, obs, candidates)

    async def _use_llm_decision(
        self, decision, obs: Observation, candidates: list, retry: bool = True  # noqa: ANN001
    ) -> tuple[Action, str]:
        """Turn one LLM reply into the step's action."""
        if decision.op == "ask_user":
            question = decision.text or "I am not sure how to continue. What should I do?"
            answer = await self.ask_user(question)
            self._history.append(f"asked: {question} / you said: {answer}")
            return Action(op="wait", text="0.5", reason=f"user: {answer}", source="user"), "user"

        action = _from_llm(decision)
        if action.op not in GLOBAL_OPS and action.element_idx is None:
            return Action(op="scroll", text="down", reason="llm gave no target", source="policy"), "llm"
        element = next((el for el in obs.elements if el.idx == action.element_idx), None)
        if element is not None:
            action.op, _ = _coerce_operation(
                action.op, element, action.text, text_is_deliberate=True
            )
        if (
            retry
            and element is not None
            and (_page_key(obs.url), action.op, action.element_idx) in self._tried
        ):
            # Told that Laya's repeat of "Wikipedia" had got nowhere, the LLM
            # chose "Wikipedia" itself, three times in one run. Hold it to the
            # same rule once; if it insists a second time, it may know better.
            again = await self._escalate(
                obs,
                candidates,
                f"You chose to {action.op} {element.label(60)}, which has already been "
                "done on this page without getting anywhere. Choose a different element "
                "or a different action.",
            )
            return await self._use_llm_decision(again, obs, candidates, retry=False)
        return await self._risk_gate(action, obs), "llm"

    async def _escalate(self, obs: Observation, candidates: list, note: str):
        self._llm_calls += 1
        await self.emit("thinking", note=note, calls=self._llm_calls)
        decision = await self.llm.decide(
            self.goal, obs, candidates, self._history, note, obs.annotated_png if self.cfg.llm.vision else None
        )
        self._step_llm.append(
            {
                "note": note,
                "op": decision.op,
                "element_idx": decision.element_idx,
                "text": decision.text[:300],
                "reason": decision.reason,
                "confidence": decision.confidence,
                "raw": decision.raw,
                "prompt": decision.prompt,
                "vision": bool(self.cfg.llm.vision and obs.annotated_png),
            }
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
        element = (
            next((el for el in obs.elements if el.idx == action.element_idx), None)
            if action.element_idx is not None
            else None
        )
        # Laya's risky_p is not a usable signal on its own. Measured over 208
        # recorded steps it sits above the 0.60 gate on 39% of them, averages
        # 0.71 on labels that really are risky and 0.46 on labels that are not,
        # and reported 1.00 for eBay's "Deals" link. Gating on it alone meant
        # being asked to approve two steps in five, nearly all of them a plain
        # link or a search box. So it only counts where the action could do
        # something Back cannot undo; the keyword list, which is exact, still
        # stops anything that spends, sends, consents or signs in.
        reversible = element is not None and (
            _is_plain_navigation(element, action.op) or _is_search_typing(element, action.op)
        )
        risky = keyword_hit or (action.risky >= self.cfg.decider.risk_threshold and not reversible)

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
        self._step_asks.append({"question": question, "answer": answer})
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
        for number, call in enumerate(self._step_llm, 1):
            trace.save_prompt(step, number, call.get("prompt") or "")
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
        payload = record.to_json()
        # The prompt is on disk per step; keep the reply inline, it is what you
        # actually re-read when a step goes wrong.
        payload["llm_calls"] = [
            {k: v for k, v in call.items() if k != "prompt"} for call in self._step_llm
        ]
        payload["asked_you"] = list(self._step_asks)
        trace.write({"type": "step", **payload})
        trace.step_note(payload, self._step_llm)


#: Hosts that are a waypoint rather than a destination. Clicking through to
#: somewhere else is the whole point of being on one, so leaving is not drift.
SEARCH_HOSTS = (
    "google.", "bing.", "duckduckgo.", "ecosia.", "startpage.", "yahoo.",
    "search.brave.", "qwant.", "baidu.", "yandex.",
)


def _is_plain_navigation(element, op: str) -> bool:  # noqa: ANN001
    """A click that only takes you to another page, which Back undoes."""
    if op != "click":
        return False
    if element.tag != "a" and element.role != "link":
        return False
    href = (element.href or "").strip().lower()
    # No href, "#" or a javascript: target means it does something instead of
    # going somewhere, and that something is not necessarily reversible.
    return bool(href) and href != "#" and not href.startswith("javascript:")


def _is_search_typing(element, op: str) -> bool:  # noqa: ANN001
    """Putting a query in a search box. Submitting it runs a search."""
    if op != "type":
        return False
    if element.role == "searchbox" or element.input_type == "search":
        return True
    return "search" in f"{element.name} {element.placeholder}".lower()


def _page_key(url: str) -> str:
    """One key per page, whatever spelling the site hands back.

    Wikipedia served `title=Special:Search` and `title=Special%3ASearch` for
    the same results page two steps apart. Keyed on the raw string, the loop
    guard saw two different pages and let Laya bounce between that page and a
    red link for the rest of the run.
    """
    parts = urlsplit(url or "")
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            unquote(parts.path),
            unquote_plus(parts.query),
            "",  # a fragment is the same page
        )
    )


def _is_search_host(host: str) -> bool:
    host = (host or "").lower()
    return any(mark in host for mark in SEARCH_HOSTS)


def _offtopic_and_unsure(element, terms: set[str], p_top: float, trust: float) -> bool:  # noqa: ANN001
    """Is this pick worth an LLM call rather than just doing it?

    Only when it is *both* unrelated to the goal's wording and something Laya
    is not sure about. Requiring relatedness on its own was the mistake: the
    check reads words, and plenty of good controls share none with the goal.
    Replaying every recorded step, demanding it always let Laya act on 8% of
    the steps it was confident about; demanding it only below p=0.85 gives
    59%, and the picks it still stops are the vague ones.
    """
    return p_top < trust and not is_plausible(element, terms)


def _coerce_operation(
    op: str, element, text: str, *, text_is_deliberate: bool = False  # noqa: ANN001
) -> tuple[str, str | None]:
    """Match the verb to the element, or say why the LLM should take over.

    Laya answers "which element" and "what to do with it" as two independent
    questions, so nothing stops it pairing a link with `select`. Playwright
    then raises - "Element is not a <select> element" - the step fails, and
    because Laya sees no history it picks the identical pair next step. A real
    run spent six of its eight steps doing exactly that to eBay's "Deals" link.

    Returns (operation to run, note to escalate with). The note is set only
    when there is genuinely nothing sensible to do without reasoning.
    """
    is_field = element.tag in {"input", "textarea"} or element.role in {
        "searchbox", "textbox", "combobox",
    }
    is_select = element.tag == "select" or element.role == "listbox"

    if op == "click" and is_field and text and text_is_deliberate:
        # The LLM asked to click a text box and handed over the text to put in
        # it. Clicking only focuses it, and the text is dropped on the floor.
        return "type", None
    if op == "select" and not is_select:
        return ("type" if is_field and text else "click"), None
    if op == "type" and not is_field:
        # Typing into a link or a button means "activate it" often enough.
        return "click", None
    if op == "type" and not text:
        return op, "Laya wants to type here but has no text."
    return op, None


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
