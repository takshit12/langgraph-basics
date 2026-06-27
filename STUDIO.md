# LangGraph Studio — Instructor Guide

A walkthrough for running the **local LangGraph Studio** live in class. The demo
graph needs **no API key**, so it works offline in the room.

## What this is

`langgraph dev` starts a local, in-memory LangGraph **API server** with hot
reloading, and (by default) opens **LangGraph Studio** in your browser pointed
at that server. Studio is the visual builder/runner: you see the graph's
topology, watch it execute node-by-node, inspect state, edit inputs, and
time-travel across runs.

> `langgraph dev` is for **local development**. For deployment you'd use
> `langgraph build` / `langgraph up` (Docker images) or **LangGraph Platform**.
> Those are out of scope for this session.

## Install (already done here)

The CLI ships in `requirements.txt` as `langgraph-cli[inmem]` (the `inmem`
extra pulls in the in-memory dev server, `langgraph-api`). It's installed in the
project venv at `.venv`. The in-mem server requires **Python 3.11+** (we use 3.11).

## Launch command

```bash
cd /Users/takshitmathur/Desktop/Projects/langgraph-basics/studio
/Users/takshitmathur/Desktop/Projects/langgraph-basics/.venv/bin/langgraph dev
```

What happens:
- Server binds to **http://127.0.0.1:2024** (default host `127.0.0.1`, port `2024`).
- A browser tab opens **LangGraph Studio** pointed at the local server, i.e.
  `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.
- API docs are at **http://127.0.0.1:2024/docs**.

Handy flags (verified against the CLI source):
- `--no-browser` — start the server but don't auto-open Studio.
- `--port 8000` — use a different port.
- `--host 0.0.0.0` — bind all interfaces (only on trusted networks).
- `--no-reload` — disable hot reload.
- `--tunnel` — expose via a public Cloudflare tunnel (useful if a browser/network
  blocks `localhost`).
- `--config path/to/langgraph.json` — point at a different manifest (defaults to
  `langgraph.json` in the current directory).

Stop the server with `Ctrl+C`.

## The manifest: `langgraph.json`

This file is how the CLI/Studio discovers your graph. Ours:

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "env": ".env"
}
```

Key meanings (from the CLI config schema):
- **`dependencies`** — packages/local paths to make importable. `"."` means
  "this folder is a dependency," so `agent.py` is importable and any local
  `requirements.txt` is honored.
- **`graphs`** — maps a graph id (`agent`) to `path/to/file.py:variable`. We
  point at the module-level compiled graph `graph` in `agent.py`.
- **`env`** — path to an env file (`.env`). Ours is empty by default; the graph
  runs without it. (`.env.example` documents the optional `OPENROUTER_API_KEY`.)

## What students will SEE in Studio

1. **Graph topology** — the nodes and edges as a diagram:
   `START → classify → {greeting | farewell | complaint | question} → finalize → END`.
   The branch out of `classify` is a **conditional edge** (a router).
2. **Editable input** — an input panel where you set the initial state, e.g.
   `{"message": "My app is broken and I want a refund"}`. Try different messages
   to trigger different branches (greeting / farewell / complaint / question).
3. **Node-by-node execution** — nodes light up as they run; you can follow the
   exact path taken through the conditional edge.
4. **State inspector** — watch `intent`, `sentiment`, `response`, and `final`
   get filled in step by step (each node returns a *partial* state that is
   merged in).
5. **Threads + time-travel** — the dev server adds persistence (a checkpointer)
   automatically, so each run is a **thread** you can revisit, fork, and replay
   from any step. (Note: `agent.py` does NOT pass its own checkpointer — the dev
   server supplies one.)

## How the demo maps to the concepts

The graph in `studio/agent.py` is a richer cousin of `simple_graph.py`:
- **State** — `ChatState` TypedDict flows through and is merged at each node.
- **Nodes** — plain functions `state -> partial dict` (`classify_node`,
  handler nodes, `finalize_node`).
- **Conditional edges** — `route_by_intent` reads state and returns the name of
  the next node; the mapping wires names to nodes.
- **START / END** — modern entry/exit idiom (`add_edge(START, "classify")`).
- **compile()** — turns the builder into the runnable `graph` Studio imports.
- **Optional LLM path** — the `question` node uses OpenRouter via the `openai`
  SDK **only if** `OPENROUTER_API_KEY` is set; otherwise it returns a
  deterministic answer so the live demo always works.

## Quick sanity check (no Studio needed)

```bash
cd /Users/takshitmathur/Desktop/Projects/langgraph-basics/studio
/Users/takshitmathur/Desktop/Projects/langgraph-basics/.venv/bin/python -c "import agent; print(type(agent.graph))"
# or run the built-in smoke test:
/Users/takshitmathur/Desktop/Projects/langgraph-basics/.venv/bin/python agent.py
```
