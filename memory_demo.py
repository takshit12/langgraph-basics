"""
memory_demo.py
==============

A SIMPLE LangGraph example that shows how MEMORY is built up and REMEMBERED
across turns using a *checkpointer* + a *thread_id*.

No LLM and NO API key are required. This is a deterministic "memory bot": the
whole point is to make the *persistence* concept vivid, not to be clever.

The three ideas this file teaches
---------------------------------
1. REDUCER:      a state field annotated with `operator.add` so LangGraph MERGES
                 a node's returned partial into the saved state by ADDING
                 (list + list -> longer list, int + int -> bigger number).
                 Your node only returns the *new* turn; LangGraph appends it.

2. CHECKPOINTER: `InMemorySaver()` persists the state of the graph after every
                 invoke, keyed by thread.

3. thread_id:    passed in `config={"configurable": {"thread_id": ...}}`.
                 SAME thread_id  -> memory is loaded from the checkpointer and
                                    keeps GROWING across invokes.
                 DIFFERENT id    -> a fresh, separate, empty memory.

Run it interactively:        python memory_demo.py
Run the headless auto-demo:  python memory_demo.py < /dev/null
"""

import sys
import operator
from typing import Annotated, TypedDict

# LangGraph 1.x idiom.
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


# ---------------------------------------------------------------------------
# 1. STATE  --  note the reducers (Annotated[..., operator.add])
# ---------------------------------------------------------------------------
class MemState(TypedDict):
    # The incoming message for THIS turn. No reducer -> it is simply overwritten
    # each invoke (we only care about the latest user message).
    user_message: str

    # The remembered conversation. The `operator.add` reducer means: whatever a
    # node returns for `history` is ADDED (list + list) to what is already saved.
    # So the node only returns the single new turn, and the list grows by itself.
    history: Annotated[list, operator.add]

    # The bot's reply for THIS turn (overwritten each invoke).
    reply: str

    # A turn counter. With the `operator.add` reducer, returning {"turn": 1}
    # INCREMENTS the saved counter (old + 1) instead of replacing it.
    turn: Annotated[int, operator.add]


# ---------------------------------------------------------------------------
# 2. THE ONE NODE  --  reads remembered history, builds a reply, appends a turn
# ---------------------------------------------------------------------------
def respond(state: MemState) -> dict:
    """Look at what we already remember + the new message, then answer.

    IMPORTANT: `state` here is the FULL saved state for this thread (the
    checkpointer loaded it for us). So `state["history"]` already contains every
    previous turn on this thread -- that is the "memory".
    """
    incoming = state["user_message"]
    remembered = state.get("history", [])          # everything said before now
    # `turn` is the saved counter; the NEXT turn number is one more than that.
    next_turn = state.get("turn", 0) + 1

    # Build a reply that EXPLICITLY references what we remember, so learners can
    # see the memory being used.
    if remembered:
        previously = "; ".join(remembered)
        reply = (
            f"This is turn {next_turn}. I remember you previously said: "
            f"[{previously}]. Now you said: '{incoming}'."
        )
    else:
        reply = (
            f"This is turn {next_turn}. I have no earlier memory yet. "
            f"You just said: '{incoming}'."
        )

    # Return only the PARTIAL update. Thanks to the reducers:
    #   - history: [incoming]  is APPENDED to the saved history (list + list)
    #   - turn:    1           is ADDED to the saved counter (old + 1)
    #   - reply:               overwrites (no reducer)
    return {"history": [incoming], "reply": reply, "turn": 1}


# ---------------------------------------------------------------------------
# 3. BUILD + COMPILE WITH A CHECKPOINTER
# ---------------------------------------------------------------------------
builder = StateGraph(MemState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

# The checkpointer is what makes memory persist between invokes.
app = builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cfg(thread_id: str) -> dict:
    """The config that selects WHICH memory to use."""
    return {"configurable": {"thread_id": thread_id}}


def remembered_for(thread_id: str) -> list:
    """Load the saved history for a thread straight from the checkpointer."""
    snapshot = app.get_state(cfg(thread_id))
    # snapshot.values is empty ({}) if this thread has never been used.
    return snapshot.values.get("history", []) if snapshot.values else []


def send(thread_id: str, message: str) -> None:
    """Send one message to a thread and show memory before & after."""
    before = remembered_for(thread_id)
    print(f"\n[thread={thread_id}] BEFORE invoke -> remembered {len(before)} turn(s): {before}")
    print(f"[thread={thread_id}] sending: {message!r}")

    result = app.invoke({"user_message": message}, config=cfg(thread_id))

    print(f"[thread={thread_id}] reply : {result['reply']}")
    print(f"[thread={thread_id}] AFTER invoke  -> remembered {len(result['history'])} turn(s): {result['history']}")
    print(f"[thread={thread_id}] turn counter now: {result['turn']}")


def show_full_memory(thread_id: str) -> None:
    """Print the entire saved state for a thread via app.get_state."""
    snapshot = app.get_state(cfg(thread_id))
    values = snapshot.values or {}
    print(f"\n=== full saved memory for thread '{thread_id}' ===")
    print(f"    turns remembered : {len(values.get('history', []))}")
    print(f"    history          : {values.get('history', [])}")
    print(f"    turn counter     : {values.get('turn', 0)}")


# ---------------------------------------------------------------------------
# 4a. NON-INTERACTIVE AUTO-DEMO  (runs when stdin is not a terminal)
# ---------------------------------------------------------------------------
def auto_demo() -> None:
    print("=" * 70)
    print("AUTO-DEMO (no terminal detected). Watch memory grow PER thread_id.")
    print("=" * 70)

    # --- alice: memory grows 1 -> 2 -> 3 ---
    print("\n--- talking as user 'alice' (thread=alice) ---")
    send("alice", "My name is Alice.")
    send("alice", "I love hiking.")
    send("alice", "What do you remember about me?")

    # --- bob: a DIFFERENT thread => fresh, empty, isolated memory ---
    print("\n--- now talking as a DIFFERENT user (thread=bob): memory starts empty ---")
    send("bob", "Hi, I am Bob and this is my first message.")

    # --- back to alice: SAME thread_id => memory RESUMES at turn 4 ---
    print("\n--- switching BACK to user 'alice': memory RESUMES (proves persistence) ---")
    send("alice", "Did you keep my earlier memory?")

    # Two independent saved memories, side by side.
    print("\n" + "=" * 70)
    print("TWO INDEPENDENT SAVED MEMORIES (one per thread_id):")
    print("=" * 70)
    show_full_memory("alice")
    show_full_memory("bob")
    print("\nDone. alice grew to 4 turns; bob has only 1 -- separate memories.")


# ---------------------------------------------------------------------------
# 4b. INTERACTIVE REPL  (runs when launched in a real terminal)
# ---------------------------------------------------------------------------
def repl() -> None:
    print("=" * 70)
    print("MEMORY DEMO  --  interactive")
    print("=" * 70)
    print("Type a message to send it to the CURRENT user/thread.")
    print("Commands:")
    print("  :user <name>   switch thread_id (new OR existing -> existing resumes!)")
    print("  :mem           show the full remembered state for the current thread")
    print("  :q             quit")
    print("-" * 70)

    current = "alice"
    print(f"(current user/thread = '{current}')")

    while True:
        try:
            line = input(f"[{current}] > ").strip()
        except EOFError:
            break

        if not line:
            continue

        if line == ":q":
            print("bye!")
            break

        if line == ":mem":
            show_full_memory(current)
            continue

        if line.startswith(":user"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                current = parts[1].strip()
                existing = remembered_for(current)
                print(
                    f"--- now talking as user '{current}' "
                    f"(thread={current}): "
                    + (
                        f"resuming memory of {len(existing)} turn(s)"
                        if existing
                        else "memory starts empty"
                    )
                    + " ---"
                )
            else:
                print("usage: :user <name>")
            continue

        # Otherwise: a normal message to the current thread.
        send(current, line)


# ---------------------------------------------------------------------------
# Entry point: pick interactive vs. headless based on stdin.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if sys.stdin.isatty():
        repl()
    else:
        # No interactive terminal (e.g. `python memory_demo.py < /dev/null`):
        # run the scripted demo instead of blocking on input().
        auto_demo()
