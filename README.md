# VibeBot

A browser agent you run yourself. You give it a goal in your browser, it goes and
does it — clicking, typing, navigating — and asks you when it gets stuck.

Three layers, cheapest first:

| Layer | What it is | When it decides | Cost |
|---|---|---|---|
| **Laya** | 421M non-autoregressive decision model, Apache 2.0 | every step, first | free, local, ~0.3–2s CPU |
| **LLM** | any local vision model via Ollama | when Laya defers or hesitates | free, local, seconds |
| **You** | the human | risky actions, dead ends, missing info | a click |

Laya is not an LLM — it generates nothing. You hand it a state and typed
questions, and it returns a probability distribution in one forward pass. So we
ask it *"which of these 12 elements, or should the LLM take this one?"* and read
the answer's shape: a clear winner gets clicked immediately; a close race wakes
the LLM. That is the whole trick.

Default install is 100% free and offline. Hosted decision engines (Jev) and
hosted LLMs are supported but off by default.

---

## Quickstart (Windows)

Python **3.10 – 3.13**. 3.11, 3.12 and 3.13 are tested end to end, including the
real Laya weights; 3.10 is the floor and runs the app but was not exercised with
torch. 3.14 is untested — torch ships `cp314` wheels, so it will probably work,
but do not find that out on a deadline.

```powershell
git clone <this repo>
cd VibeBot
py -3 -m venv .venv          # or py -3.13 / py -3.12 to pin one
.venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium

pip install laya                 # the decision model (pulls torch, ~2.5 GB)
# CPU-only torch is much smaller, if you don't have an NVIDIA card:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# the reasoning model
winget install Ollama.Ollama
ollama pull qwen3.5:4b           # 3.4 GB; see "Choosing the LLM" below

copy config.example.yaml config.yaml
python -m vibebot doctor         # checks all four pieces
python -m vibebot serve          # then open http://127.0.0.1:8765
```

Linux/macOS is the same with `python3 -m venv .venv && source .venv/bin/activate`.

First run downloads ~800 MB of Laya weights to your Hugging Face cache. After
that it is fully offline.

## Using it

**Web UI** (`python -m vibebot serve`) — type a goal, watch the annotated
screenshots, see which layer made each decision, answer when it asks.

### Watching it work

Two ways, both optional:

- **The browser window itself.** `headless: false` is the default, so on your PC
  there is a real Chromium on screen doing the clicking. You can take over in it
  at any time — the agent re-reads the page every step, so it just carries on
  from wherever you left it.
- **The Live button in the UI.** Streams the browser continuously instead of one
  still per step, with a tab strip and a LIVE marker. Useful headless, on a
  server, or when the agent is thinking and you want to see the page *now*.

Live view is off until you press it, and stops when you press it again or close
the tab — it is a full PNG per frame, so it is not free. `server.live_fps` sets
the ceiling (2/s by default); what you actually get depends on how fast the
machine can capture, typically 2–4/s.

The stream is read-only. It cannot disturb a run, and it keeps updating while
the agent is mid-decision, paused, or waiting on your answer.

**Terminal** — for scripts and servers:

```bash
python -m vibebot run "find the cheapest 65 inch OLED on prisjakt.nu" --headless
python -m vibebot doctor
```

It asks questions on stdin in this mode.

### On a server

Headless switches on automatically when there is no display. To expose the UI,
set a token — without one, binding to a public address is refused, because
anyone who can reach that port can drive a browser that is logged into your
accounts:

```bash
VIBEBOT_SERVER_TOKEN=$(openssl rand -hex 16) \
  python -m vibebot serve --host 0.0.0.0 --port 8765
```

Put it behind a reverse proxy with TLS if it faces the internet.

## How a step works

```
observe page  ->  rank ~150 elements down to 12  ->  Laya answers 4 questions
                                                         |
       p(top) >= 0.55 and margin >= 0.15  ->  act -------+
       "ask_llm" or a close race          ->  local LLM (+ annotated screenshot)
       looks consequential                ->  ask you first
       LLM unsure, or 6 steps unchanged   ->  ask you
```

The four questions are `target` (which element, or an escape hatch), `operation`
(click/type/select), `done` and `risky`. `ask_llm` sits in the same option list
as the page elements, so Laya escalates *itself* — exactly the handoff you
wanted, rather than a hard-coded rule.

Because Laya cannot write text, "what should I type?" is also a multiple choice:
two or three strings derived from your goal, or `ask_llm` if none fit.

### Frames and tabs

Cookie walls, checkouts and embedded search boxes live in iframes, so perception
walks the main document **and every iframe** (up to `browser.max_frames`),
handing out one flat set of indices. Element `[7]` is element `[7]` whether it
sits in the page or three frames deep; the agent remembers which frame each one
came from and clicks it there. Elements inside a frame are labelled
`(iframe 2)` so both models know.

Tabs work the way a person expects: a `target="_blank"` link or popup is
followed automatically, and the agent can `switch_tab` and `close_tab`. The open
tabs are listed to the LLM, and when exactly two are open "it's in the other
tab" is a choice Laya can make by itself. It refuses to close the last tab.

### Choosing the LLM

`qwen3.5` is multimodal at every size, so even the small tags can see:

| Tag | Download | Fits | Notes |
|---|---|---|---|
| `qwen3.5:9b` | 6.6 GB | 12 GB VRAM | if you have the room |
| `qwen3.5:4b` | 3.4 GB | 8 GB VRAM | the default |
| `qwen3.5:2b` | 2.7 GB | 6 GB VRAM | still sees, noticeably dimmer |
| `qwen3.5:0.8b` | 1.0 GB | 4 GB VRAM | last resort; expect more `ask_user` |

Set `llm.model` to anything Ollama serves, or point `llm.backend: openai` at a
hosted endpoint. Whatever you pick needs **vision**, or set `llm.vision: false`
and it runs on the element list alone (worse, but it works).

`llm.base_url` defaults to whichever URL the chosen backend needs, so switching
`llm.backend` is a one-line change. Set it only to point somewhere unusual — an
OpenRouter URL, LM Studio on another port, Ollama on another host. Leaving one
backend's URL behind when you switch to the other is the failure this defaults
to avoiding: Ollama also answers `/v1/chat/completions`, so `backend: openai`
aimed at port 11434 does not error, it just quietly runs locally. `vibebot
doctor` now calls that out, along with a missing API key on *any* hosted
provider and a `model` the endpoint does not serve.

**Thinking models need `llm.think: off`, which is the default.** Ollama turns
thinking *on* by default for models that support it, and a thinking model puts
its answer in `message.thinking` while `content` comes back empty — which looks
exactly like a broken model. We send `think: false`, fall back to reading
`thinking` if a model ignores that, and retry without the field on servers that
reject it. Set `on` if you want the reasoning (slower), `auto` to leave it to
the server.

One expectation to set: a newer VLM helps less here than the benchmarks imply.
The model is never asked to *find* anything on screen — it gets a numbered
element list and a screenshot with those numbers drawn on it, so the grounding
ability those scores measure is work the DOM layer already did. What matters is
instruction-following and sticking to JSON. Judge a swap by how often runs end
in `ask_user`, not by leaderboard position.

Nothing in this table has been benchmarked inside this repo. The traces in
`.vibebot/traces/` are how you settle it on your own sites: same goal, two
models, compare escalations and failures.

### Tuning the gate

Measured against the real model, not its README:

- Laya's reported `confidence` is **1 − normalised entropy**, not the top
  option's probability. It falls as the option list grows — a *correct* pick at
  p=0.66 reports confidence 0.29. Gating on it makes the agent escalate
  constantly, so `accept_probability` + `accept_margin` are used instead.
- One batched step is ~0.3s on CPU with a short page digest, ~1.5–2.5s with the
  default 600-char one. Drop `decider.page_text_chars` to 200 for speed.
- `laya-mind2web-browser-agent` on the Hub is a fine-tune on 671 examples. It is
  not used here; base Laya with a pre-ranked shortlist generalises better.

Raise `accept_probability` if it clicks wrong things; lower it if it wakes the
LLM for everything.

## Safety

- **Confirmation** before anything matching money / send / delete / login,
  either from Laya's `risky` probability or a keyword in the element's label.
  `policy.autonomy: guided` confirms *every* action; `yolo` confirms nothing.
- **Never types into password fields.** It asks you to type it yourself in the
  browser window instead.
- **Domain allow/block lists** in `policy`.
- **Budgets**: `max_steps`, `max_llm_calls`, and a stall detector that escalates
  to you when the page stops changing.
- The browser profile in `.vibebot/profile` keeps you logged in between runs.
  It is gitignored. Treat it like a password file.

None of this makes it safe to leave unattended on your bank. It is an agent
clicking things on the internet.

## Going hosted later

Both layers are swappable and nothing else in the code changes:

```yaml
decider:
  backend: jev
  jev_base_url: https://...
  jev_api_key_env: JEV_API_KEY

llm:
  backend: openai          # also OpenRouter, LM Studio, vLLM, llama.cpp
                           # an unknown backend name is a startup error
  model: gpt-4o-mini
  base_url: https://api.openai.com
  api_key_env: OPENAI_API_KEY
```

Caveat: Jev's wire format is not verified here. `vibebot/deciders/jev_decider.py`
speaks the same request shape Laya uses locally; if the real API differs, the
two methods at the bottom of that file are all that need changing.

## Traces

Every run writes `.vibebot/traces/<run-id>/steps.jsonl` plus a screenshot per
step: the candidate list, Laya's full distribution, which layer decided, what
happened. Useful for debugging, and it is also labelled training data — every
step where Laya deferred and the LLM picked correctly is a fine-tuning example
for making Laya handle *your* sites without the LLM.

## Layout

```
vibebot/
  agent.py          the loop and the escalation ladder
  browser.py        Playwright: element extraction, screenshots, actions
  ranking.py        cheap lexical pre-filter (150 elements -> 12)
  deciders/         laya | jev | heuristic fallback
  llm/              ollama | openai-compatible | none
  server.py, ui/    local web UI (FastAPI + websocket)
  trace.py          JSONL run logs
```

## Known limits

- Single browser session per process; one goal at a time.
- Shadow DOM is only partly reachable (open roots via `[role=...]` only).
- Cross-origin iframes are read through Playwright, so they work — but a frame
  that navigates mid-step is skipped for that step rather than retried.
- Zero-shot Laya is weak on unusual pages; the LLM carries those steps. The
  trace files exist so you can fix that with a fine-tune.

Apache-2.0.
