# VibeBot

A browser agent you run yourself. You give it a goal in your browser, it goes and
does it — clicking, typing, navigating — and asks you when it gets stuck.

The question behind it: **how much of a real browsing task can a fast, cheap
"system 1" carry, with an LLM as "system 2" only for the heavier reasoning?**

How a goal runs:

| Part | What it is | What it does |
|---|---|---|
| **Reader** | the page in reading order, every element numbered inline | what the LLM sees - content first, menus last, active filters marked |
| **LLM** (system 2) | a local vision model via Ollama (`qwen3.5:4b`), or any OpenAI-compatible API | reads the page, acts, and writes a plan of the next few steps |
| **Fast decider** (system 1) | `match` by default; `laya` or `jev` pluggable | carries the plan out, one "which element is this?" at a time, and hands back to the LLM when unsure |
| **Checks** | plain code | loops, undone filters, "nothing found" after a bad search, a count from page 1 of 3 |
| **You** | the human | asked before anything that buys, sends, signs in or consents |

See "Can a fast decider carry the flow?" for what has been measured so far.

Default install is 100% free and offline. Hosted decision engines (Jev) and
hosted LLMs are supported but off by default.

---

## Starting and stopping it

Double-click **`VibeBot`** in the project folder. It sets itself up the first
time (virtual environment, packages, Chromium — a few minutes), then starts and
opens the UI in your browser. Every run after that takes seconds.

To stop it, any of these:

- press **Quit** in the page — closes the browser VibeBot drives, releases the
  language model, and shuts the server down;
- double-click **`Stop VibeBot`** — asks the server to close politely, and only
  ends the process if it will not;
- close the console window.

**Quit** and **Stop** are different buttons on purpose: *Stop* ends the current
goal and leaves VibeBot running for the next one, *Quit* shuts the whole thing
down.

Prefer a terminal, or on Linux/macOS, `python -m vibebot serve` still does the
same thing; `server.open_browser: false` turns the automatic tab off.

## Quickstart (Windows)

The double-click launcher above does all of this for you. Here it is by hand,
for when you want to pin a Python version or install pieces separately.

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

pip install laya                 # optional: only for decider.backend: laya (pulls torch, ~2.5 GB)
# CPU-only torch is much smaller, if you don't have an NVIDIA card:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# the reasoning model
winget install Ollama.Ollama
ollama pull qwen3.5:4b           # 3.4 GB; see "Choosing the LLM" below

copy config.example.yaml config.yaml
python -m vibebot doctor         # checks all four pieces
python -m vibebot serve          # opens http://127.0.0.1:8765 for you
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

The default (`decider.mode: plan`):

```
observe page (reader)  ->  is there a plan step to do?
   no, or the page no longer fits it  ->  LLM acts, and writes the next steps
   yes, "read and answer"              ->  LLM
   yes, a URL / scroll / back          ->  done directly, no model
   yes, an element                     ->  fast decider: "which element is this?"
                                            sure  -> act     unsure -> LLM
```

The rest of this section describes `decider.mode: gate`, the original design,
where the fast decider picks every step for the whole goal. It is kept for
comparison; see "Can a fast decider carry the flow?" for why it is not the
default.

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

"Fits" is VRAM. What does not fit is pinned in **system RAM**, so a card
smaller than the column above still works — as long as the rest of the machine
leaves the difference free. On a 4 GB card, `qwen3.5:4b` pins roughly 1.8 GB of
host RAM, and the browser window this agent opens is itself the hungriest thing
on the box. Run out and Ollama answers `500` on every step, which VibeBot now
reports with Ollama's own words ("out of memory") instead of a bare status code.
Close other apps, set `llm.vision: false`, or drop a size.

**Measured on a 4 GB laptop GPU (RTX 3050 Ti, 16 GB RAM)**, where every model
below spills from the card into system RAM and generation speed is the
bottleneck:

| model | generation | one call | test shop, 2 rounds | total |
|---|---|---|---|---|
| `qwen3.5:4b`, thinking off | 9.7 tok/s | ~26 s | **8/10** | **~800 s** |
| `qwen3.5:9b`, thinking off | 4.5 tok/s | ~32 s | 7/10 | 2,648 s |
| `qwen3.5:4b`, thinking on | 9.1 tok/s | ~77 s, some over 10 min | stopped | - |
| Ternary Bonsai 2 27B (PrismML fork, CUDA 12.4) | 0.13 tok/s | ~30 min | not viable | - |

- **The 9B** is not smarter enough here to pay for being 3.4x slower.
- **Thinking** triples the average call and, worse, some calls run away: one
  thought for over ten minutes and hit the timeout without answering. Ollama
  cannot cap thinking separately, and capping the whole reply cuts the answer.
- **Bonsai 2 27B** (Apache 2.0, 5.95 GB, needs PrismML's llama.cpp fork -
  stock Ollama rejects the file) generated 0.13 tokens/second, split across GPU
  and CPU and CPU-only alike, with the model fully resident (not paging).
  PrismML's own guidance is to put the whole model on the GPU and to use a
  smaller model below 4 GB of VRAM. It may be excellent on a bigger card.

What helped instead was writing less: `reason` and `confidence` were 24% of
every reply, came after the action so could not influence it, and cost ~3 s a
call at these speeds. They are gone; `seen` explains the action.

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
thinking *on* by default for models that support it. We send `think: false`,
fall back to reading `message.thinking` if a model ignores that, and retry
without the field when a server answers **400** to it (only a 400 — a 500 is the
server failing for some other reason, and treating that as a rejection used to
switch thinking back on for the rest of the run). Set `on` if you want the reasoning,
`auto` to leave it to the server.

Measured on qwen3.5:4b, not inherited from the vendor README:

- `think: true` roughly **triples latency** (5.8s -> 19.3s on the same prompt)
  and fills `message.thinking` *as well as* `content` — it does not leave
  `content` empty the way older Ollama builds did.
- `content` comes back empty only when thinking exhausts `llm.max_tokens`:
  Ollama then returns `done_reason: "length"`, `content: ""` and 500-odd
  characters of reasoning prose. There is no action in that, so it becomes a
  question for you that names the budget.
- A reply cut off at the budget while still writing JSON is **repaired, not
  discarded**. At `num_predict: 60` the model produced every field the agent
  needs and stopped one character before the closing brace; the old parser threw
  that away and asked you what to do. It is now closed and used, and the
  truncation is logged so you know to raise `llm.max_tokens`.
- Sending `think: false` to a model with no thinking support (gemma3:1b) is
  accepted with a 200 on current Ollama, not rejected. The retry-without-`think`
  path stays for older servers, but it no longer fires here.

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

Every run writes `.vibebot/traces/<run-id>/`:

| file | what is in it |
|---|---|
| `run.log` | the run as readable text — candidates, Laya's verdict, what the LLM was asked, what it replied verbatim, the action, the outcome, memory per step |
| `steps.jsonl` | the same events complete, for analysis |
| `step-NNN.png` | exactly what the model was shown |
| `step-NNN-llmN.prompt.txt` | the exact prompt behind each LLM call |

`run.log` is the one to open first. A step reads:

```
--- step 5 --- https://www.ebay.com/globaldeals
    candidates : 12
        e18: combobox "Search for anything"
        e3: a "Deals"
    laya       : laya -> e3 p=1.0 margin=1.0 op=click done_p=0.35 (9017ms)
    llm ask    : Laya wants to click a "Deals" again, and this page has already had
                 exactly that. It did not get us anywhere — pick something else.
    llm reply  : {"op":"click","element_idx":[18],"text":"MacBook Pro M4", ...}
    action     : [llm] type idx=18 text='MacBook Pro M4'
    outcome    : typed 'MacBook Pro M4' into element 18  (19480ms)
    memory     : 1,053 MB headroom (26,683/27,736 MB committed, 3,826 MB RAM free)
```

That format is not decoration — every bug fixed in the last pass was found by
reading it. The `element_idx: [18]` above is a real example: the model picked
the right box and the parser dropped the answer because it was a list.

The rows are also labelled training data — every step where Laya deferred and
the LLM picked correctly is a fine-tuning example for making Laya handle *your*
sites without the LLM.

## Can a fast decider carry the flow?

What has been measured, on the local benchmark shop (`vibebot bench`, two
rounds each unless noted).

**Asked "what should happen next, for this goal?" - no.** That was the
original design (`decider.mode: gate`): Laya picks every step and the LLM
steps in when it is unsure. It clicked "Deals", the Eiffel Tower logo, and
the filter it had just switched on, all at p=1.00. That is a planning
question, and a zero-shot classifier with no view of page state answers it
confidently and wrongly.

**So the job was split** (`decider.mode: plan`, the default). The LLM acts
and writes the next steps as intents - "click 'Price: lowest first'" - and
the fast decider only answers "which element is this step?" over candidates
ranked against the step.

**Zero-shot Laya cannot do that either.** `vibebot bench --executor` asks
the question directly, 20 steps on real shop pages, half in the planner's
wording and half paraphrased:

| | planner wording: right / wrong / deferred | paraphrased: right / wrong / deferred | per step |
|---|---|---|---|
| `laya` | 4 / 4 / 2 | 0 / 8 / 2 | 3.2 s |
| `match` (label matching, no model) | 10 / 0 / 0 | 1 / 0 / 9 | 0 ms |

With the options reversed and shuffled, Laya picked the first-listed option
3 times in 36 - chance, so not position bias. It is drawn to "Search"
whatever the step says, and the same step with the options in a different
order can get a different answer. Right in 12 of 36.

**But the system-1 slot does work, filled with something reliable:**

| setup | passed | total time | LLM calls | fast decider right when it acted |
|---|---|---|---|---|
| LLM decides every step | 8/10 | 1,011 s | 43 | - |
| gate + Laya (earlier code) | 7/10 | 1,469 s | 41 | often wrong |
| plan + Laya | 7/10 | 933 s | 38 | 3 of 11 |
| plan + match | 7/10 | 959 s | 42 | 9 of 9 |
| **plan + match, plan field first** | **8/10** | **772 s** | **35** | **8 of 8** |

Same accuracy as the LLM alone, 24% less time, 19% fewer LLM calls.

**What limits it now is the planner, not the executor.** Of 35 LLM calls in
the last run, 24 were because no plan was left: `qwen3.5:4b` often writes no
plan or a one-step one, so the fast decider only gets about one step in
seven. A stronger planner (a bigger local model, or DeepSeek through the
openai backend) should hand it more; nothing in the code needs to change to
try that.

**What would make a learned fast decider worth it** is the paraphrase
column. `match` defers every step it cannot match by label; a model that got
those right without being wrong elsewhere would take over steps the LLM does
now. Zero-shot Laya does not. Fine-tuning it on the benchmark shop's steps
and the traces is the obvious next experiment. To try Jev, or anything else,
in that slot:

```
python -m vibebot bench --executor jev match     # the exam, about a minute
python -m vibebot bench --decider jev --repeat 2 # the full suite
```

## What the models actually see

This is the part that decides whether a browser agent is any good, so it is
worth being precise about.

| | fast decider (plan mode) | the LLM |
|---|---|---|
| question | "which element is this plan step?" | the goal, and what to do next |
| page | no page text - only the step | the page in reading order, up to `llm.page_chars` (6,000) |
| elements | the top 12, ranked against the step | every one, numbered inline where it sits on the page |
| state | none | what is selected, a loud note when a page shows no results |
| memory | last 4 actions | everything it has written in `seen` and `notes`, on every step |
| context | 512-1,024 tokens for Laya | `llm.num_ctx` (8,192) |

In `gate` mode the fast decider is asked about the whole goal instead, with
the page title, URL and first `decider.page_text_chars` (200) of raw text.

Until this version the LLM got twelve elements and the first 1,200 characters
of the page. On a shop, the first 1,200 characters are the header, "Sign in"
and a promo banner - the listings and prices were never in front of it.

The reader (`browser._READ_JS`) walks the page in document order, keeps the
content first and moves header, navigation and footer to the end, and writes
each element as `[number] label` on its own line, with `(selected)` on active
filters and sorts. On the test shop's results page that is all ten listings
with spec, condition and price, sorting and pagination, in about 1,900
characters.

Memory: every reply now starts with a `seen` field - what on this page matters
for the goal - which the agent stores and hands back on every later step.
Optional notes did not work with `qwen3.5:4b` (two saved in 67 steps); a
mandatory first field did (32 of 32 replies).

Before accepting an answer the agent checks it once against three failures
the benchmark kept producing:

- "nothing matches", after a search that was simply too narrow;
- a count or "cheapest" given from page 1 of 3;
- an answer that says the part asked for is "not specified".

Each gets one push to look again. A genuine answer survives being asked twice.

## Measuring it

Every change to the agent used to be argued from one run, which is how a fix
that helps one goal and quietly breaks two others gets shipped. So there is a
benchmark:

```
python -m vibebot bench                     # the test shop, once
python -m vibebot bench --repeat 2          # two rounds - one is too noisy to trust
python -m vibebot bench --llm-only          # the same, with no fast decider
python -m vibebot bench --decider laya      # a different fast decider (match, laya, jev)
python -m vibebot bench --mode gate         # the original design, for comparison
python -m vibebot bench --executor laya match   # exam the fast decider alone
python -m vibebot bench --suite web         # real websites, as a smoke test
python -m vibebot bench --only seller --show
python -m vibebot bench --model qwen3.5:9b --out after.json
```

**The default suite is a local test shop** (`vibebot/benchsite`), not the
internet. Real sites made the numbers meaningless: eBay served bot checks to a
headless browser, listings changed between runs, and a model can recite 1889
for the Eiffel Tower without reading a word. The shop never changes and every
goal has exactly one right answer, derived from its catalogue. It contains the
traps real runs fell into: the right listing on page 3 of 3 under the default
sort, cheaper New / For-parts / Refurbished listings, a "1TB" of storage next
to 32GB of memory, site furniture on every page, and a wiki answer (about a
fictional tower) 4,000 characters down the page.

It reports whether each answer matched, steps, seconds, how many steps Laya
decided itself, and LLM calls. It **refuses to run** if Laya could not load and
the heuristic fallback stood in - one early benchmark did exactly that and its
numbers looked perfectly healthy. `--allow-degraded` measures the fallback on
purpose.

Single rounds swing by one or two goals out of five. Compare with `--repeat 2`
or more.

## Memory

The thing most likely to break a run on a laptop is not the model size, it is
the **commit limit**. Measured on a 16 GB Windows machine mid-run:

```
Committed : 22.5 GB
Limit     : 25.2 GB
HEADROOM  :  2.8 GB   <- what a new allocation must fit in
Available physical: 7,009 MB
```

With 7 GB of RAM apparently free, torch refused an 8 MB tensor and the process
segfaulted with no traceback. Windows charges every allocation against the
commit limit (RAM + pagefile) and declines it when the charge does not fit,
however much RAM is idle — and a C extension treats that refusal as fatal.

What VibeBot costs, measured on that machine:

| | commit |
|---|---|
| Laya, `preload: true` (English + multilingual) | ~4.9 GB |
| Laya, `preload: false` (default — English only, lazily) | ~2.0 GB |
| `qwen3.5:4b` resident in Ollama, any screenshot size | ~5.7 GB |
| Chromium, one window | ~1.1 GB |

So the defaults changed: Laya no longer preloads weights an English run never
touches, `vibebot doctor` prints the headroom, `decider.min_headroom_mb` skips
Laya (loudly) rather than dying when memory is short, and VibeBot tells Ollama
to release the model when a run ends instead of leaving 5.7 GB committed for
another five minutes.

If a run still dies, the biggest wins are usually outside VibeBot: an idle WSL
or Docker VM holds several GB, and so does a second browser.

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
  trace files exist so you can fix that with a fine-tune. Measured on eBay's
  home page it picks "Deals" or "My eBay" over the search box at p=1.00, so the
  loop breaker below does a lot of the real work.
- Laya has no memory between steps, so an exact repeat of (page, action) means
  a loop rather than a decision. The second time a page asks for the same
  action the step goes to the LLM instead. Without this, runs spent their whole
  budget bouncing between two pages.
- Search engines block the agent. A run aimed at Google landed on
  `/sorry/index` (the "unusual traffic" check) and never recovered; VibeBot
  will not solve a CAPTCHA for you. Go to the site you actually want.
- `qwen3.5:4b` gets the mechanics right and the judgement wrong. Asked for the
  *cheapest used* MacBook Pro M4 it searched correctly, then reported the first
  listing on the page without sorting by price or filtering to used. Treat the
  answer as a starting point, or use a bigger model.

Apache-2.0.
