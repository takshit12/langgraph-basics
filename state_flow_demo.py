"""
state_flow_demo.py  —  SEE and STEP THROUGH how State, Nodes, and Edges move
data in LangGraph.  *Interactive* console teaching tool (no Streamlit, no UI,
no API key required).

Run interactively:   .venv/bin/python state_flow_demo.py
Run headless/auto:    .venv/bin/python state_flow_demo.py < /dev/null

Its whole job is to make the invisible visible. Every time the graph:
  - hands STATE into a node          -> we log  IN    state = {...}
  - a node RETURNS a partial dict    -> we log  OUT   partial = {...}
  - LangGraph MERGES it back in      -> we log  MERGE state  = {...}
  - traverses an EDGE                -> we log  EDGE: A --> B
  - makes a conditional ROUTE        -> we log  ROUTER ... => target
...is printed to your terminal so you can watch the data flow.

When run in a real terminal it becomes a REPL: you type a message, then press
Enter to advance the graph one node at a time (STEP mode) so you can read every
transfer before the next one happens.

Core concepts on display:
  State : a TypedDict that is THE shared data passed between every step.
  Node  : a plain function  (state) -> partial state dict.
  Edge  : add_edge(A, B), the START / END sentinels, and CONDITIONAL edges
          (a router function returns the name of the next node).
  Merge : LangGraph merges each node's returned partial back into the state.
          For a plain TypedDict this is last-writer-wins per key, and lists
          are REPLACED (not appended). So to ACCUMULATE the `steps` list we
          read the old list out of state and return a new, longer one.
          (LangGraph can automate this with a "reducer" — see note at bottom.)
"""

import os
import sys
from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# Tiny ANSI color helpers (degrade to plain text if not a terminal / NO_COLOR)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s


def CYAN(s):    return _c("36;1", s)
def GREEN(s):   return _c("32;1", s)
def YELLOW(s):  return _c("33;1", s)
def MAGENTA(s): return _c("35;1", s)
def BLUE(s):    return _c("34;1", s)
def DIM(s):     return _c("2", s)
def BOLD(s):    return _c("1", s)


def fmt(state: dict) -> str:
    """Compact, readable rendering of a state/partial dict."""
    return "{" + ", ".join(f"{k}={v!r}" for k, v in state.items()) + "}"


# ---------------------------------------------------------------------------
# Step 1: STATE — the shared data that flows through the whole graph.
# Every node receives this and returns a PARTIAL of it.
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    text: Optional[str]            # the incoming user message (the only input)
    tokens: Optional[List[str]]    # set by `ingest`
    category: Optional[str]        # set by `analyze`  -> drives the router
    sentiment: Optional[str]       # set by `analyze`
    reply: Optional[str]           # set by a handler branch, refined by finalize
    status: Optional[str]          # set by `finalize`
    steps: List[str]               # ACCUMULATES the names of nodes that ran


# ---------------------------------------------------------------------------
# INTERACTIVITY STATE (module-level switches the REPL flips)
# ---------------------------------------------------------------------------
VERBOSE = True            # print the IN/OUT/MERGE/EDGE logging?
STEP_MODE = True          # pause before each node and wait for Enter?
INTERACTIVE = False       # is a human at the keyboard? (set in main)


def step_pause(cue: str) -> None:
    """Pause the run so the learner controls the pace.

    Only pauses when we have a human at the keyboard AND step mode is on.
    The cue tells them exactly what is about to happen ("run node 'analyze'").
    In :auto mode (or headless), this is a no-op so the graph runs straight
    through.
    """
    if INTERACTIVE and STEP_MODE:
        try:
            input(DIM(f"  [press Enter to {cue}]"))
        except EOFError:
            # stdin closed mid-run: stop pausing for the rest of the session.
            pass


# ---------------------------------------------------------------------------
# A logging wrapper so every transfer in/out of a node is printed.
# This is just for teaching — a node is normally only the inner function.
# It ALSO drives interactivity: it calls step_pause() before each node runs.
# ---------------------------------------------------------------------------
_trace = {"prev": "START", "path": []}


def traced(name: str):
    def wrap(fn):
        def runner(state: ChatState) -> dict:
            # Pause BEFORE the node runs so the learner can read the upcoming
            # edge transfer and decide to advance. (No-op in :auto/headless.)
            step_pause(f"run node {name!r}  ──>")
            if VERBOSE:
                print()
                print(MAGENTA(f"  EDGE: {_trace['prev']}  ──>  {name}"))
                print(CYAN(f"  ┌── NODE  {name}"))
                print(f"  │  {DIM('IN    state   =')} {fmt(state)}")
            partial = fn(state)                 # <-- the node does its work here
            if VERBOSE:
                print(f"  │  {GREEN('OUT   partial =')} {fmt(partial)}")
                merged = {**state, **partial}   # exactly what LangGraph does (no reducer)
                print(f"  │  {YELLOW('MERGE state   =')} {fmt(merged)}")
                print(f"  │  {DIM('changed keys  :')} {list(partial.keys())}")
                print(CYAN("  └──"))
            _trace["prev"] = name
            _trace["path"].append(name)
            return partial
        return runner
    return wrap


# ---------------------------------------------------------------------------
# Step 2: NODES — each is a plain function (state) -> partial state dict.
# Notice none of them mutate `state` in place; they RETURN what changed.
# ---------------------------------------------------------------------------
FAQ_WORDS = {"refund", "password", "invoice", "cancel", "billing", "reset"}
SMALLTALK_WORDS = {"hi", "hello", "hey", "thanks", "thank", "bye"}
NEG_WORDS = {"broken", "angry", "terrible", "hate", "refund", "urgent", "asap"}


@traced("ingest")
def ingest(state: ChatState) -> dict:
    """First node: split the raw text into tokens. Reads `text`, writes `tokens`."""
    text = (state.get("text") or "").strip()
    return {"tokens": text.split(), "steps": state["steps"] + ["ingest"]}


@traced("analyze")
def analyze(state: ChatState) -> dict:
    """Reads `tokens`, writes `category` + `sentiment`. `category` drives routing."""
    tokens = [t.lower().strip("?!.,") for t in (state.get("tokens") or [])]
    if any(t in FAQ_WORDS for t in tokens):
        category = "faq"
    elif any(t in SMALLTALK_WORDS for t in tokens):
        category = "smalltalk"
    else:
        category = "other"
    sentiment = "negative" if any(t in NEG_WORDS for t in tokens) else "neutral"
    return {"category": category, "sentiment": sentiment,
            "steps": state["steps"] + ["analyze"]}


@traced("handle_faq")
def handle_faq(state: ChatState) -> dict:
    return {"reply": "Here is the self-service article that answers that.",
            "steps": state["steps"] + ["handle_faq"]}


@traced("handle_smalltalk")
def handle_smalltalk(state: ChatState) -> dict:
    return {"reply": "Hi there! How can I help you today?",
            "steps": state["steps"] + ["handle_smalltalk"]}


@traced("escalate")
def escalate(state: ChatState) -> dict:
    return {"reply": "I have routed this to a human specialist.",
            "steps": state["steps"] + ["escalate"]}


@traced("finalize")
def finalize(state: ChatState) -> dict:
    """Last node: reads several keys, stamps the reply, marks status done."""
    tag = f"[{state.get('category')}/{state.get('sentiment')}]"
    return {"status": "done",
            "reply": f"{tag} {state.get('reply', '')}",
            "steps": state["steps"] + ["finalize"]}


# ---------------------------------------------------------------------------
# Step 3: THE ROUTER — a conditional-edge function. It reads state and returns
# the NAME (a string) of the next node. The mapping in add_conditional_edges
# turns that string into the actual node to jump to.
# ---------------------------------------------------------------------------
def route_by_category(state: ChatState) -> str:
    cat = state.get("category")
    target = {"faq": "handle_faq", "smalltalk": "handle_smalltalk"}.get(cat, "escalate")
    # Pause + explain the routing decision before the conditional edge fires.
    step_pause(f"fire ROUTER (category={cat!r} → {target!r})")
    if VERBOSE:
        print(BLUE(f"  ROUTER route_by_category: reads category={cat!r}  =>  go to {target!r}"))
    return target


# ---------------------------------------------------------------------------
# Step 4: BUILD THE GRAPH — add nodes, then wire the edges.
# ---------------------------------------------------------------------------
workflow = StateGraph(ChatState)

workflow.add_node("ingest", ingest)
workflow.add_node("analyze", analyze)
workflow.add_node("handle_faq", handle_faq)
workflow.add_node("handle_smalltalk", handle_smalltalk)
workflow.add_node("escalate", escalate)
workflow.add_node("finalize", finalize)

workflow.add_edge(START, "ingest")          # entry point
workflow.add_edge("ingest", "analyze")      # normal edge

# Conditional edge: after `analyze`, run the router and jump to ONE branch.
workflow.add_conditional_edges(
    "analyze",
    route_by_category,
    {
        "handle_faq": "handle_faq",
        "handle_smalltalk": "handle_smalltalk",
        "escalate": "escalate",
    },
)

# All three branches converge on `finalize`, then END.
workflow.add_edge("handle_faq", "finalize")
workflow.add_edge("handle_smalltalk", "finalize")
workflow.add_edge("escalate", "finalize")
workflow.add_edge("finalize", END)

# Step 5: compile the definition into a runnable app.
app = workflow.compile()


# ---------------------------------------------------------------------------
# Pretty printers for the demo
# ---------------------------------------------------------------------------
GRAPH_MAP = """
        START
          |
          v
        ingest  -->  analyze  -->  (router: route_by_category)
                                          |
                       +------------------+------------------+
                       v                  v                  v
                  handle_faq      handle_smalltalk       escalate
                       +------------------+------------------+
                                          v
                                       finalize
                                          |
                                          v
                                         END
"""


def legend():
    print(BOLD("LEGEND"))
    print(MAGENTA("  EDGE  ") + "= the graph moved from one node to the next")
    print(BLUE("  ROUTER") + "= a conditional edge choosing the next node")
    print(DIM("  IN    ") + "= the full state handed INTO a node")
    print(GREEN("  OUT   ") + "= the partial dict the node RETURNED")
    print(YELLOW("  MERGE ") + "= state AFTER LangGraph merged the partial back in")


def controls_help():
    print(BOLD("CONTROLS"))
    print("  " + GREEN("type a message") + " then Enter   run it through the graph")
    print("  " + GREEN("<empty> + Enter") + "             use a rotating sample message")
    print("  " + GREEN(":step") + "                      pause before each node (default)")
    print("  " + GREEN(":auto") + "                      run straight through, no pauses")
    print("  " + GREEN(":stream <message>") + "          show LangGraph's native .stream() view")
    print("  " + GREEN(":help") + "                      reprint this help + legend")
    print("  " + GREEN(":q / quit / exit") + "           leave")
    print(DIM("  In STEP mode, press Enter at each [press Enter ...] cue to advance one node."))


def _blank_state(text: str) -> ChatState:
    """A fresh, fully-populated initial state for a single run."""
    return {"text": text, "tokens": None, "category": None, "sentiment": None,
            "reply": None, "status": None, "steps": []}


def run(text: str):
    """Invoke the graph once, with full IN/OUT/MERGE/EDGE logging."""
    _trace["prev"], _trace["path"] = "START", []
    initial = _blank_state(text)
    print("\n" + BOLD("=" * 74))
    print(BOLD(f"INVOKE  app.invoke({{'text': {text!r}, 'steps': []}})"))
    print(BOLD("=" * 74))
    print(MAGENTA("  START") + DIM(f"   initial state = {fmt(initial)}"))
    final = app.invoke(initial)
    print(MAGENTA(f"\n  EDGE: {_trace['prev']}  ──>  END"))
    print(MAGENTA("  END"))
    print("\n" + BOLD("  PATH TAKEN:  ") + GREEN("  ->  ".join(["START"] + _trace["path"] + ["END"])))
    print(BOLD("  FINAL STATE:"))
    for k, v in final.items():
        print(f"     {k:<10} = {v!r}")


def stream_demo(text: str = "please cancel my invoice"):
    """LangGraph's OWN view of the transfer: .stream(stream_mode='updates')
    yields {node_name: partial_dict} the instant each node finishes — i.e. the
    exact partial being merged into state. No custom logging needed."""
    global VERBOSE
    saved = VERBOSE
    VERBOSE = False                      # silence our wrapper; let stream speak
    print("\n" + BOLD("=" * 74))
    print(BOLD("BONUS — LangGraph's native transfer view: app.stream(stream_mode='updates')"))
    print(DIM("Each line is one node finishing and emitting the partial dict LangGraph merges."))
    print(BOLD("=" * 74))
    print(MAGENTA("  START") + DIM(f"   input = {{'text': {text!r}}}"))
    for chunk in app.stream(_blank_state(text), stream_mode="updates"):
        for node, partial in chunk.items():
            print(f"  {CYAN(node.ljust(18))} emitted ──> {GREEN(fmt(partial))}")
    VERBOSE = saved


# ---------------------------------------------------------------------------
# THE REPL  (only used when a human is at the keyboard)
# ---------------------------------------------------------------------------
# Rotating samples used when the learner just presses Enter. Each lands on a
# different branch so an impatient learner still sees all three.
SAMPLES = [
    "I need a refund and I'm really angry",   # refund -> faq branch, negative
    "hello there, thanks!",                   # hello/thanks -> smalltalk branch
    "my dashboard shows weird numbers",       # nothing matched -> escalate branch
]


def repl():
    """Interactive loop: read a message (or command), run the graph, repeat."""
    global STEP_MODE
    print(BOLD("\nGRAPH TOPOLOGY (static wiring, before we run anything):"))
    print(DIM(GRAPH_MAP))
    legend()
    print()
    controls_help()

    sample_i = 0
    while True:
        mode = "STEP" if STEP_MODE else "AUTO"
        try:
            raw = input("\n" + BOLD(f"[{mode}] ") + GREEN("> "))
        except EOFError:
            break                                   # stdin closed -> exit cleanly
        cmd = raw.strip()
        low = cmd.lower()

        if low in (":q", "quit", "exit", ":quit", ":exit"):
            break
        if low in (":help", "help", "?"):
            legend(); print(); controls_help(); continue
        if low == ":step":
            STEP_MODE = True
            print(DIM("  -> STEP mode ON: I'll pause before each node.")); continue
        if low == ":auto":
            STEP_MODE = False
            print(DIM("  -> AUTO mode ON: runs straight through, no pauses.")); continue
        if low.startswith(":stream"):
            arg = cmd[len(":stream"):].strip()
            stream_demo(arg or SAMPLES[sample_i % len(SAMPLES)]); continue

        # Empty input -> rotate through the samples so every branch is reachable.
        if cmd == "":
            cmd = SAMPLES[sample_i % len(SAMPLES)]
            sample_i += 1
            print(DIM(f"  (empty input — using sample: {cmd!r})"))

        run(cmd)

    print(DIM("\nBye! You stepped through "
              f"{len(_trace['path'])} node-runs in the last graph invocation."))


# ---------------------------------------------------------------------------
# NON-INTERACTIVE SAFETY: a scripted auto-demo for headless verification.
# Never calls input(); runs one message per branch straight through, then the
# native stream view, then exits.
# ---------------------------------------------------------------------------
def auto_demo():
    global STEP_MODE
    STEP_MODE = False                    # belt-and-braces: never pause
    print(BOLD("\nGRAPH TOPOLOGY (static wiring, before we run anything):"))
    print(DIM(GRAPH_MAP))
    legend()
    print(DIM("\n(no interactive terminal detected — running scripted auto-demo)"))

    for msg in SAMPLES:                  # one per branch: faq, smalltalk, escalate
        run(msg)

    stream_demo()

    print("\n" + DIM("Note: `steps` had to be appended by hand (state['steps'] + [name]) "
                     "because a plain TypedDict REPLACES lists on merge. LangGraph can do "
                     "this automatically with a reducer, e.g.:"))
    print(DIM("    from typing import Annotated; import operator"))
    print(DIM("    steps: Annotated[list, operator.add]   # merges by '+' instead of replace"))


if __name__ == "__main__":
    # Detect whether a human is at the keyboard. If stdin is NOT a TTY (piped,
    # redirected from /dev/null, CI, etc.) we must NOT call input() or we'd
    # hang — so we run the scripted auto-demo instead.
    INTERACTIVE = sys.stdin.isatty()
    if INTERACTIVE:
        repl()
    else:
        auto_demo()
