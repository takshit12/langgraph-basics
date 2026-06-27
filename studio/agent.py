# LangGraph Studio demo graph — NO API KEY REQUIRED.
#
# This file exposes a module-level compiled graph named `graph` so that
# `langgraph dev` (the local Studio dev server) can discover and render it.
# See langgraph.json -> "graphs": {"agent": "./agent.py:graph"}.
#
# It deliberately echoes simple_graph.py but is a bit richer: a 4-way
# classifier + conditional routing into specialised handler nodes, plus a
# final "finalize" node so Studio shows a clear multi-step topology you can
# watch execute node-by-node.
#
# Concepts on display in Studio:
#   - State (a TypedDict that flows through the graph and is merged at each node)
#   - Nodes (plain functions: state -> partial state dict)
#   - Conditional edges (a router that picks the next node by name)
#   - START / END (modern entry/exit idiom)
#   - compile() -> a runnable graph
#   - checkpointer / threads (Studio adds persistence so you can time-travel)
import os
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END


# --- Step 1: State -----------------------------------------------------------
# Every node receives the whole state and returns a PARTIAL dict, which
# LangGraph merges back in. In Studio, the right-hand "state inspector"
# shows these fields filling in as each node runs.
class ChatState(TypedDict):
    message: Optional[str]        # user input (editable in Studio's input panel)
    intent: Optional[str]         # set by the classifier node
    sentiment: Optional[str]      # naive positive/negative/neutral signal
    response: Optional[str]       # set by whichever handler node runs
    final: Optional[str]          # set by the finalize node


# --- Plain helpers (not nodes) ----------------------------------------------
# Keyword heuristics so the demo runs with ZERO external dependencies / keys.
INTENT_KEYWORDS = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "farewell": ["bye", "goodbye", "see you", "cya", "later"],
    "complaint": ["broken", "bug", "angry", "refund", "terrible", "not working", "hate"],
}

POSITIVE = ["great", "love", "thanks", "thank you", "awesome", "good", "nice"]
NEGATIVE = ["broken", "bug", "angry", "terrible", "hate", "bad", "refund"]


def classify_intent(text: str) -> str:
    t = text.lower()
    for intent, words in INTENT_KEYWORDS.items():
        if any(w in t for w in words):
            return intent
    # Anything we don't recognise is treated as a question to answer.
    return "question"


def classify_sentiment(text: str) -> str:
    t = text.lower()
    pos = any(w in t for w in POSITIVE)
    neg = any(w in t for w in NEGATIVE)
    if neg and not pos:
        return "negative"
    if pos and not neg:
        return "positive"
    return "neutral"


# --- Step 2: Nodes -----------------------------------------------------------
def classify_node(state: ChatState) -> dict:
    """First node: read the message, tag intent + sentiment."""
    message = (state.get("message") or "").strip()
    return {
        "intent": classify_intent(message),
        "sentiment": classify_sentiment(message),
    }


def greeting_node(state: ChatState) -> dict:
    return {"response": "Hello! 👋 What can I help you with today?"}


def farewell_node(state: ChatState) -> dict:
    return {"response": "Thanks for stopping by — goodbye! 👋"}


def complaint_node(state: ChatState) -> dict:
    return {
        "response": (
            "I'm sorry you're having trouble. I've logged this as a complaint "
            "and a human will follow up. Could you share more detail?"
        )
    }


def question_node(state: ChatState) -> dict:
    """The default branch. Optionally uses an LLM via OpenRouter if a key is
    present, but falls back to a deterministic reply so Studio works offline."""
    message = (state.get("message") or "").strip()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        try:
            # OpenRouter via the OpenAI SDK (kept intact for the LLM path).
            from openai import OpenAI

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                messages=[{"role": "user", "content": message}],
            )
            return {"response": completion.choices[0].message.content}
        except Exception as exc:  # keep the demo resilient in the classroom
            return {"response": f"(LLM unavailable: {exc}) You asked: '{message}'"}
    # No key: deterministic answer so the live demo always works.
    return {"response": f"Here's what I found for: '{message}' (offline demo answer)."}


def finalize_node(state: ChatState) -> dict:
    """Last node: combine signals into a final, tagged message so the graph
    has a clear convergence point that Studio renders nicely."""
    intent = state.get("intent")
    sentiment = state.get("sentiment")
    response = state.get("response") or ""
    return {"final": f"[intent={intent} | sentiment={sentiment}] {response}"}


# --- Step 3: The router ------------------------------------------------------
# A conditional edge function reads state and returns the NAME of the branch.
def route_by_intent(state: ChatState) -> str:
    return state.get("intent") or "question"


# --- Step 4: Build the graph -------------------------------------------------
workflow = StateGraph(ChatState)

workflow.add_node("classify", classify_node)
workflow.add_node("greeting", greeting_node)
workflow.add_node("farewell", farewell_node)
workflow.add_node("complaint", complaint_node)
workflow.add_node("question", question_node)
workflow.add_node("finalize", finalize_node)

# START -> classify is the modern entry-point idiom (replaces set_entry_point).
workflow.add_edge(START, "classify")

# Conditional edge: after classify, the router picks one handler branch.
# The mapping turns each router return value into a target node.
workflow.add_conditional_edges(
    "classify",
    route_by_intent,
    {
        "greeting": "greeting",
        "farewell": "farewell",
        "complaint": "complaint",
        "question": "question",
    },
)

# All handler branches converge on finalize, then END.
workflow.add_edge("greeting", "finalize")
workflow.add_edge("farewell", "finalize")
workflow.add_edge("complaint", "finalize")
workflow.add_edge("question", "finalize")
workflow.add_edge("finalize", END)


# --- Step 5: Compile ---------------------------------------------------------
# Expose a module-level `graph` for `langgraph dev` / Studio to import.
# NOTE: do NOT pass a checkpointer here — the LangGraph dev server provides its
# own persistence (threads/time-travel) automatically when running in Studio.
graph = workflow.compile()


if __name__ == "__main__":
    # Quick local smoke test (run directly, outside Studio).
    for msg in ["Hi there!", "My app is broken and I want a refund", "What is LangGraph?"]:
        print(msg, "->", graph.invoke({"message": msg})["final"])
