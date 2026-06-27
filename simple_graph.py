# LangGraph Fundamentals: a tiny graph with NO LLM and NO API keys.
# Concepts shown: State (TypedDict), Nodes (plain functions), Conditional
# edges (a router), START/END, compile(), and invoke().

# Step 1: Define the Graph State.
# The state is a TypedDict that flows through the graph. Each node receives
# the current state and returns a PARTIAL dict that gets merged back in.
import os
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Load OPENROUTER_API_KEY / OPENROUTER_MODEL from the project .env so the
# "search" node can call a real LLM. If no key is set, it falls back to a
# canned string, so this file always runs.
load_dotenv()


class GraphState(TypedDict):
    question: Optional[str]
    classification: Optional[str]
    response: Optional[str]


# Plain helper (not a node): decide if the question is a greeting.
def classify(question: str) -> str:
    greetings = ["hello", "hi", "hey"]
    if any(word in question.lower() for word in greetings):
        return "greeting"
    return "search"


# Step 2: Create the Graph (the "canvas"), typed by our state.
workflow = StateGraph(GraphState)


# Step 3: Define Nodes.
# A node is just a function: (state) -> partial state dict.
def classify_input_node(state: GraphState) -> dict:
    question = (state.get("question") or "").strip()
    return {"classification": classify(question)}


def handle_greeting_node(state: GraphState) -> dict:
    return {"response": "Hello! How can I help you today?"}


# Plain helper (not a node): answer a question via OpenRouter, or fall back.
def search_with_openrouter(question: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return f"Search result for '{question}'"  # offline fallback (no key)
    # OpenRouter speaks the OpenAI API, so we use the OpenAI SDK with a base_url.
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    completion = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        messages=[{"role": "user", "content": question}],
    )
    return completion.choices[0].message.content


def handle_search_node(state: GraphState) -> dict:
    # An LLM-backed node. Note it's STILL just a function (state) -> partial dict —
    # exactly like handle_greeting_node, it just happens to call a model.
    question = (state.get("question") or "").strip()
    return {"response": search_with_openrouter(question)}


# The router: reads state and returns the NAME of the next node to run.
def decide_next_node(state: GraphState) -> str:
    return "handle_greeting" if state.get("classification") == "greeting" else "handle_search"


# Step 4: Add nodes to the graph.
workflow.add_node("classify_input", classify_input_node)
workflow.add_node("handle_greeting", handle_greeting_node)
workflow.add_node("handle_search", handle_search_node)


# Step 5: Wire the edges.
# START -> classify_input is the modern entry-point idiom (replaces
# the older set_entry_point()).
workflow.add_edge(START, "classify_input")

# A conditional edge: after classify_input, run the router to pick the branch.
# The mapping turns each router return value into a target node.
workflow.add_conditional_edges(
    "classify_input",
    decide_next_node,
    {
        "handle_greeting": "handle_greeting",
        "handle_search": "handle_search",
    },
)

# Both branches finish at END.
workflow.add_edge("handle_greeting", END)
workflow.add_edge("handle_search", END)


# Step 6: Compile the graph into a runnable app, then invoke it.
app = workflow.compile()

if __name__ == "__main__":
    # Greeting branch -> plain node, no LLM call.
    print(app.invoke({"question": "Hi, how are you?"}))
    # Search branch -> LLM node (uses OpenRouter if OPENROUTER_API_KEY is set).
    print(app.invoke({"question": "In one sentence, what is LangGraph?"}))
