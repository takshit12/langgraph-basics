# LangGraph Studio demo graph — an INTERACTIVE STORYTELLER (like storygen.py),
# built on MessagesState so LangGraph Studio's **Chat** tab works.
#
# This file exposes a module-level compiled graph named `graph` so that
# `langgraph dev` (the local Studio dev server) can discover and render it.
# See langgraph.json -> "graphs": {"agent": "./agent.py:graph"}.
#
# Why MessagesState? Studio's Chat tab only appears when the state has a
# `messages` key (a list of LangChain chat messages). storygen.py uses a big
# structured StoryState (no `messages`) and drives its loop from Streamlit, so
# it can't power Chat directly — this is the same idea (a choose-your-own
# adventure narrator) expressed as a chat graph so you can talk to it in Studio.
#
# Concepts on display in Studio:
#   - State (here: MessagesState — the conversation, merged with the add_messages reducer)
#   - Nodes (plain functions: state -> partial state dict)
#   - A conditional edge from START (new story vs. continue the story)
#   - compile() -> a runnable graph
#   - threads / memory (Studio persists each thread, so the story continues)
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState

# Optional: read OPENROUTER_API_KEY / OPENROUTER_MODEL from studio/.env (or the
# environment). With a key you get a real AI story; without one it still runs
# with a simple canned scene, so Studio always works.
load_dotenv()


# --- The narrator's instructions ---------------------------------------------
STORY_SYSTEM = (
    "You are the narrator of a short, fun, choose-your-own-adventure story. "
    "Continue the adventure based on the conversation so far. Keep each turn to "
    "about 4-6 vivid sentences, and ALWAYS end your turn by offering the player "
    "2-3 numbered choices for what to do next."
)


def _llm():
    """Build an OpenRouter-backed chat model, or return None if no key is set."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0.8,
    )


def _narrate(state: MessagesState, offline_scene: str) -> dict:
    """Shared helper: ask the LLM to continue the story (or use a canned scene).

    Both story nodes have the SAME shape as every LangGraph node:
    (state) -> a partial dict. Here they return one new `messages` entry, which
    the built-in `add_messages` reducer APPENDS to the conversation.
    """
    llm = _llm()
    if llm is None:
        return {"messages": [AIMessage(content=offline_scene)]}
    conversation = [SystemMessage(content=STORY_SYSTEM)] + state["messages"]
    return {"messages": [llm.invoke(conversation)]}


# --- Nodes -------------------------------------------------------------------
def begin_story(state: MessagesState) -> dict:
    """First turn: set the opening scene."""
    return _narrate(
        state,
        "You wake at a misty crossroads at dawn, a worn map in your hand. "
        "(Set OPENROUTER_API_KEY in studio/.env for a real, AI-written story.)\n"
        "1) Take the forest path   2) Follow the river   3) Head for the distant tower",
    )


def continue_story(state: MessagesState) -> dict:
    """Later turns: continue based on the player's latest choice."""
    return _narrate(
        state,
        "The path twists onward and something stirs ahead. "
        "(Set OPENROUTER_API_KEY in studio/.env for a real, AI-written story.)\n"
        "1) Press forward   2) Turn back   3) Look around carefully",
    )


# --- The router: new story or continue? --------------------------------------
def route(state: MessagesState) -> Literal["begin", "continue"]:
    """If the narrator has already spoken, we're continuing; otherwise it's a
    brand-new story. (A conditional edge straight from START.)"""
    messages = state.get("messages", [])
    told_before = any(isinstance(m, AIMessage) for m in messages[:-1])
    return "continue" if told_before else "begin"


# --- Build + compile ---------------------------------------------------------
builder = StateGraph(MessagesState)
builder.add_node("begin", begin_story)
builder.add_node("continue", continue_story)

# Conditional entry point: decide the first node from START.
builder.add_conditional_edges(START, route, {"begin": "begin", "continue": "continue"})
builder.add_edge("begin", END)
builder.add_edge("continue", END)

# Expose the compiled graph for `langgraph dev` / Studio to import.
# No checkpointer here — the dev server provides its own persistence (threads),
# which is what keeps your story going across turns.
graph = builder.compile()


if __name__ == "__main__":
    # Quick local smoke test (run directly, outside Studio).
    from langchain_core.messages import HumanMessage

    s = graph.invoke({"messages": [HumanMessage(content="Start a fantasy adventure")]})
    print("TURN 1:", s["messages"][-1].content[:200])
    s = graph.invoke({"messages": s["messages"] + [HumanMessage(content="I take the forest path")]})
    print("TURN 2:", s["messages"][-1].content[:200])
