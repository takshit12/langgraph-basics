# LangGraph from Scratch

A hands-on, beginner-friendly intro to [LangGraph](https://github.com/langchain-ai/langgraph)
— learn the three core ideas (**State**, **Nodes**, **Edges**) by running small, heavily
commented examples and watching data flow through a graph. Verified on **LangGraph 1.x**.

## The core idea in one picture

```text
            input
              |
              v
   +--------- STATE (a TypedDict, shared memory) ---------+
   |   START -> node -> node -> (router?) -> node -> END    |
   +-------------------------------------------------------+
              |
              v
        final state
```

- **State** — a `TypedDict` that flows through the whole graph. Each node gets it and
  returns a **partial** dict that LangGraph **merges** back in.
- **Nodes** — plain functions `(state) -> partial state dict`. (An LLM node is just a node.)
- **Edges** — wiring: `add_edge(A, B)`, the `START`/`END` sentinels, and **conditional
  edges** (a router function returns the name of the next node).
- **Lifecycle** — `StateGraph(State)` → `add_node` → `add_edge`/conditional → `compile()` → `invoke()`.

## Setup — from zero to running

This walks a brand-new machine from nothing to every example running. The commands
below use **absolute-style relative paths from the repo root** — run them after you've
`cd`-ed into `langgraph-basics`. If a `.venv/bin/...` command feels long, you can
[activate the venv](#troubleshooting) once and drop the prefix.

### 1. Prerequisites

You need **Python 3.10 or newer** (LangGraph 1.x requires 3.10+; this repo was built
and tested with **Python 3.11**) and **git**.

On macOS the 3.11 interpreter is usually called `python3.11`. Check what you have:

```bash
python3.11 --version    # e.g. Python 3.11.x  (preferred)
python3 --version       # must be >= 3.10
git --version
```

If `python3.11` is missing on macOS, install it with [Homebrew](https://brew.sh):
`brew install python@3.11`. On Linux use your package manager (e.g.
`sudo apt install python3.11 python3.11-venv`).

**Optional but recommended:** [`uv`](https://docs.astral.sh/uv/) is a very fast
Python package/venv manager. Install it with `brew install uv` (macOS) or
`curl -LsSf https://astral.sh/uv/install.sh | sh`. Every step below shows both the
**`uv`** path and a plain **`python -m venv`** fallback — pick one.

### 2. Get the code

```bash
git clone <repo-url>
cd langgraph-basics
```

All remaining commands assume you are inside the `langgraph-basics` directory.

### 3. Create the virtual environment and install dependencies

Pick **one** of the two options.

**Option A — uv (fast):**

```bash
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
```

**Option B — plain venv (no extra tools):**

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Either way you end up with a `.venv/` folder containing the interpreter at
`.venv/bin/python` and the LangGraph CLI at `.venv/bin/langgraph`. The install pulls in
`langgraph`, `langgraph-cli[inmem]` (for Studio), `langchain`, `langchain-openai`,
`openai`, `python-dotenv`, and a couple of extras.

### 4. (Optional) Add an API key for the LLM paths

**Every core example runs with NO key.** `state_flow_demo.py` never touches a model.
`simple_graph.py`'s search branch, `memory_demo.py`'s replies, and the Studio storyteller
use OpenRouter *when a key is present* and fall back to canned text when it isn't, so you
can skip this section and still see everything work. Two files do need a key:
`langsmith-simple.py` and `agent-eval.py` (the LangSmith demos).

To enable the LLM paths:

1. Get a key at <https://openrouter.ai/keys> (OpenRouter; you may need a small credit
   balance for live calls).
2. Copy the template and edit it:

   ```bash
   cp .env.example .env
   ```

3. In `.env`, set:

   ```dotenv
   OPENROUTER_API_KEY=sk-or-...your-key...
   OPENROUTER_MODEL=openai/gpt-4o-mini
   ```

   Optional, for the LangSmith demos: `LANGSMITH_API_KEY=lsv2_...` (free key at
   <https://smith.langchain.com> → Settings → API Keys) plus `LANGSMITH_TRACING=true`.
   Without it `langsmith-simple.py` still runs (no upload) and `agent-eval.py` runs the
   agent but skips the hosted evaluations.

The scripts call `load_dotenv()`, so they read this automatically. OpenRouter is used
through the OpenAI SDK with `base_url=https://openrouter.ai/api/v1`. **`.env` is
gitignored** — your key will not be committed.

### 5. Verify the install

One line confirms the core imports resolve in your venv:

```bash
.venv/bin/python -c "from langgraph.graph import StateGraph, START, END; from langgraph.checkpoint.memory import InMemorySaver; print('ok')"
```

If it prints `ok`, you're ready.

### 6. Run the examples

Run each from the repo root. The "what you'll see" note tells you it worked.

```bash
.venv/bin/python simple_graph.py
```
> **What you'll see:** the graph classifies a question, branches via a conditional edge,
> and prints the final state. The greeting branch is a plain node; the search branch
> calls the LLM if a key is set (otherwise a canned fallback).

```bash
.venv/bin/python state_flow_demo.py
```
> **What you'll see:** an interactive console that logs every hand-off
> (`IN → OUT(partial) → MERGE → EDGE → ROUTER`) as state flows through
> `ingest → analyze → (router) → {handle_faq | handle_smalltalk | escalate} → finalize → END`.
> Controls: `:step`/`:auto` (pause before each node vs. run through), `:stream <msg>`
> (LangGraph's native `stream_mode="updates"` view), `:q` (quit). Piping
> `... < /dev/null` runs a headless scripted auto-demo instead of the REPL.

```bash
.venv/bin/python memory_demo.py
```
> **What you'll see:** persistence in action — an `InMemorySaver` checkpointer plus a
> `thread_id` make the graph *remember* across turns; reducers
> (`Annotated[list, operator.add]`) append history. Controls: `:user <name>` (switch
> `thread_id` — the same name resumes saved state, a new name starts fresh), `:mem`
> (print everything remembered for the current thread), `:q` (quit). Also runs headless
> with `... < /dev/null`.

### 7. Run LangGraph Studio (the local visual builder)

Studio lets you watch a graph execute node-by-node in your browser, inspect state as it
fills in, and use threads / time-travel.

```bash
cd studio && ../.venv/bin/langgraph dev
```

> **What you'll see:** the dev server starts and the Studio UI opens in your browser,
> wired to the API at <http://127.0.0.1:2024>. The demo graph is a **multi-agent text
> adventure**: a Game Master (supervisor) reads each move you type and hands the turn to
> one of three specialist agents (`START → game_master → {narrator | combat | npc} → END`).
> Built on `MessagesState`, so the **Chat tab works**: type "Start a fantasy adventure",
> then "I attack the wolf" and "I ask the old woman about the tower", and watch a different
> box light up each turn while `scene_type` and `inventory` change in the state inspector.
> Each conversation is a thread; a new thread starts a new adventure. Runs with **no API
> key** (canned scenes, keyword routing). `studio/agent.py` exposes the compiled `graph`;
> `studio/langgraph.json` is the manifest.

Optional: create `studio/.env` (copy from `studio/.env.example`) to set
`OPENROUTER_API_KEY` (real AI-written scenes) and/or `LANGSMITH_API_KEY` (every Studio
turn shows up as a trace). Both are optional.

### 8. Troubleshooting

- **Python < 3.10 won't work.** LangGraph 1.x requires Python 3.10+. Recreate the venv
  with `python3.11` (see step 3).
- **`langgraph: command not found`.** The CLI lives inside the venv. Use the full path
  `.venv/bin/langgraph`, or activate the venv first:
  `source .venv/bin/activate` (then `langgraph`, `python`, and `pip` work without the
  `.venv/bin/` prefix; run `deactivate` to exit).
- **LLM / network errors** (timeouts, 401, 402, empty responses) on `simple_graph.py`'s
  search branch, `memory_demo.py`, or the Studio storyteller — check that
  `OPENROUTER_API_KEY` is set in the right `.env` (root vs `studio/`) and that your
  OpenRouter account has credits.
- **"LangSmith API key missing" banner** in Studio — **safe to ignore.** Tracing is
  optional; set `LANGSMITH_API_KEY` in `studio/.env` only if you want it.

## The examples

| File | Run | Teaches |
|---|---|---|
| `simple_graph.py` | `.venv/bin/python simple_graph.py` | The fundamentals: state, nodes, **conditional edges**. Greeting branch = plain node; search branch = live LLM node (OpenRouter). |
| `state_flow_demo.py` | `.venv/bin/python state_flow_demo.py` | **How state transfers between nodes.** Interactive: type a message, step the graph one node at a time, and watch every `IN → OUT(partial) → MERGE → EDGE → ROUTER` hand-off printed live. |
| `memory_demo.py` | `.venv/bin/python memory_demo.py` | **Memory / persistence.** A checkpointer (`InMemorySaver`) + `thread_id` make the graph *remember* across turns. Same `thread_id` resumes saved state; a different one starts fresh. With a key, the LLM's replies use the remembered history. |
| `studio/agent.py` | `cd studio && ../.venv/bin/langgraph dev` | **Multi-agent, made visual in LangGraph Studio.** A Game Master supervisor routes each move to a Narrator, Combat Master, or NPC agent via a conditional edge; the agents collaborate through shared state (`inventory` with an `operator.add` reducer). Chat tab, threads, time-travel. No API key required. |
| `langsmith-simple.py` | `.venv/bin/python langsmith-simple.py` | **LangSmith tracing** with one `@traceable` decorator and `wrap_openai`. A two-step pipeline (`gpt-4o-mini` → `claude-3-haiku` via one OpenRouter key) recorded as a trace tree. Needs `OPENROUTER_API_KEY`; uploads if `LANGSMITH_API_KEY` is set. |
| `agent-eval.py` | `.venv/bin/python agent-eval.py` | **A tool-calling (ReAct) agent, then evaluations.** `add`/`multiply` tools, `ToolNode`, the `agent → tools → agent` loop; then a LangSmith dataset graded by exact-match, LLM-as-a-judge, and a substring check. Exact-match scores 0 on correct sentence answers — that's the lesson. Needs `OPENROUTER_API_KEY`; evals need `LANGSMITH_API_KEY`. |

> Both console demos also run headless (`... < /dev/null`) as a scripted auto-demo.

## Breakout: build your own agent

`breakout-activity.pdf` is the hands-on activity (~15 minutes). Copy its full text into an
AI coding assistant (Claude Code, Codex, Cursor, Copilot agent mode) and send it. The
assistant asks what you want your agent to do, then builds and runs a LangGraph pipeline
of three or more nodes around your idea, printing the state after each step. Core API
only, LangGraph 1.x, and an offline fallback so it works without a key.

### `state_flow_demo.py` controls
- type any message to run it through the graph; empty = sample message
- `:step` / `:auto` — toggle stepping (pause before each node) vs. run-through
- `:stream <msg>` — show LangGraph's native `app.stream(stream_mode="updates")` view
- `:q` — quit

### `memory_demo.py` controls
- type messages to add to the current user's memory
- `:user <name>` — switch `thread_id` (a new user starts fresh; an existing one resumes)
- `:mem` — print everything remembered for the current thread
- `:q` — quit

## LangGraph Studio (the local visual builder)

Watch a graph execute node-by-node in your browser, inspect state as it fills in, and
use threads / time-travel.

```bash
cd studio && ../.venv/bin/langgraph dev
```

- API: http://127.0.0.1:2024 · Studio UI opens automatically in your browser
- `studio/agent.py` exposes a compiled `graph`; `studio/langgraph.json` is the manifest.
- Four agents: `game_master` (supervisor, writes `scene_type` and never speaks),
  `narrator`, `combat`, and `npc` (specialists, each with its own role prompt). The
  supervisor's decision is read by a conditional edge — the hand-off is plain routing.
- Runs with **no API key** (canned scenes, keyword routing); set `OPENROUTER_API_KEY` in
  `studio/.env` so all four agents are real model calls, and `LANGSMITH_API_KEY` to trace
  every turn.
- The Chat tab works because the state extends `MessagesState` (a `messages` key with the
  `add_messages` reducer). `inventory` uses `operator.add`, so loot from one agent is
  visible in the next agent's prompt.

## API surface used here

| Call | What it does |
|---|---|
| `StateGraph(State)` | create a graph typed by your state |
| `add_node(name, fn)` | register a node function |
| `add_edge(A, B)` / `add_edge(START, n)` / `add_edge(n, END)` | wire a fixed transition / entry / exit |
| `add_conditional_edges(src, router, mapping)` | branch based on a router's returned name |
| `compile(checkpointer=...)` | turn the definition into a runnable app (optional persistence) |
| `invoke(input, config)` | run the graph; `config={"configurable": {"thread_id": ...}}` selects the memory thread |
| `stream(input, stream_mode="updates")` | yield each node's partial as it finishes |
| `InMemorySaver()` | an in-memory checkpointer for per-thread memory |
| `MessagesState` / `add_messages` | built-in chat state whose `messages` reducer appends; enables Studio's Chat tab |
| `ToolNode` / `bind_tools` | prebuilt node that runs tools the model requested; the agent → tools → agent loop (`agent-eval.py`) |
