"""
agent_and_middleware.py
=======================

BONUS / self-study -- NOT part of the live walkthrough.

This is the payoff slide for Section 3: `create_agent` assembles model + tools
(+ middleware) into a working agent loop in a few lines -- the common-case
version of what learners built by hand in LangGraph.

The tie-back to land: `create_agent(...)` returns a **CompiledStateGraph** --
i.e. it is *literally built on LangGraph internally*. The agent loop they wired
node-by-node in LangGraph Basics is the same machinery `create_agent` compiles
for you. (This is also why `pip show langgraph` appears after installing this
repo -- it's a transitive dependency pulled in by `create_agent`.)

Middleware = hooks at points in the agent loop (before/after a model or tool
call). Here we use ModelRetryMiddleware, a built-in that maps directly to the
outline's named example "automatic retries on failed model calls."

NO API key is REQUIRED:
  * WITHOUT a key -> we still CONSTRUCT the agent and print its assembled
    structure (tool names, middleware names, graph nodes) -- so the shape is
    visible with no live call. We skip .invoke().
  * WITH a key -> we also run one .invoke() and print the final answer.

Run it:   python agent_and_middleware.py
"""

import os

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

# Pick up OPENROUTER_API_KEY / OPENROUTER_MODEL from the project .env (optional).
load_dotenv()
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
LIVE_MODEL = bool(os.getenv("OPENROUTER_API_KEY"))


def short_api_error(text: str) -> str:
    """Turn a noisy API error string into one short, classroom-friendly line."""
    if "401" in text or "User not found" in text or "authentication" in text.lower():
        return "401 auth error -- your OPENROUTER_API_KEY looks invalid or expired"
    return text.splitlines()[0][:160]


# ---------------------------------------------------------------------------
# 1. A TOOL  --  the same @tool building block from langchain_fundamentals.py.
#    An agent is model + tools; this is the "tools" part.
# ---------------------------------------------------------------------------
@tool
def get_word_length(word: str) -> int:
    """Return the number of letters in a single word."""
    return len(word)


# ---------------------------------------------------------------------------
# 2. MIDDLEWARE  --  one real built-in hook into the agent loop.
#    ModelRetryMiddleware automatically retries a failed model call (with
#    exponential backoff) instead of letting the whole run crash -- exactly the
#    "automatic retries on failed model calls" example from the outline. Before
#    middleware existed, this meant hand-rolling try/except + retry logic inside
#    a LangGraph node; middleware is LangChain's standard way to get it.
# ---------------------------------------------------------------------------
retry_middleware = ModelRetryMiddleware(max_retries=2, backoff_factor=2.0)


# ---------------------------------------------------------------------------
# 3. THE MODEL STEP  --  a genuine ChatOpenAI Runnable, wired to OpenRouter.
#    Constructing a ChatOpenAI makes NO network call -- that only happens on
#    .invoke(). So even with no key we build a real model object (with a
#    placeholder key) purely so the agent graph compiles and can be inspected.
# ---------------------------------------------------------------------------
def build_model():
    """Return a ChatOpenAI model. With a key it's ready to invoke; without one
    it uses a placeholder key so the graph still compiles for structure-only
    inspection (we never .invoke() it in that case)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=MODEL_NAME,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY") or "sk-placeholder-no-network",
    )


# ---------------------------------------------------------------------------
# 4. ASSEMBLE THE AGENT  --  model + tools + middleware, in a few lines.
#    Returns a CompiledStateGraph: create_agent is built on LangGraph. No
#    network call happens here -- only when we actually .invoke() the agent.
# ---------------------------------------------------------------------------
def build_agent():
    """Construct the agent from model + tools + middleware."""
    return create_agent(
        model=build_model(),
        tools=[get_word_length],
        middleware=[retry_middleware],
    )


def print_structure(agent) -> None:
    """Print the agent's assembled shape: tools, middleware, and graph nodes."""
    print(f"agent type       : {type(agent).__name__}  "
          "(a CompiledStateGraph -- create_agent is built on LangGraph)")
    print(f"tools            : {[get_word_length.name]}")
    print(f"middleware       : {[type(retry_middleware).__name__]}")
    try:
        nodes = list(agent.get_graph().nodes)
        print(f"graph nodes      : {nodes}")
    except Exception as exc:
        print(f"graph nodes      : (unavailable: {exc})")


# ---------------------------------------------------------------------------
# 5. ENTRY POINT  --  build the agent, print its structure, optionally invoke.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mode = "LIVE (key set)" if LIVE_MODEL else "STRUCTURE-ONLY (no key)"
    print("=" * 70)
    print(f"create_agent + middleware  --  {mode}")
    print("=" * 70)

    agent = build_agent()

    print("\n[1] ASSEMBLED AGENT STRUCTURE")
    print_structure(agent)

    print("\n[2] RUN")
    if not LIVE_MODEL:
        print(
            "[structure-only -- no OPENROUTER_API_KEY set] Skipping .invoke(). "
            "The graph above compiled without any network call; set a key to "
            "watch the model decide to call `get_word_length` and answer."
        )
    else:
        try:
            result = agent.invoke(
                {"messages": [{"role": "user",
                               "content": "How many letters are in the word 'strawberry'? "
                                          "Use the get_word_length tool."}]}
            )
            final = result["messages"][-1]
            content = (final.content or "").strip()
            # ModelRetryMiddleware swallows a failed call and RETURNS the failure
            # text as the final message (so nothing raises). Detect that so we
            # don't mislabel an auth failure as a real "final answer".
            low = content.lower()
            if "call failed" in low or "authenticationerror" in low or "401" in content:
                print(f"[live run failed: {short_api_error(content)}]")
                print("  The assembled STRUCTURE above is the real teaching artifact;")
                print("  the live call just needs a valid OPENROUTER_API_KEY.")
            else:
                print(f"final answer     : {content}")

                # -----------------------------------------------------------
                # [under the hood] UNROLL THE AGENT LOOP.
                #   create_agent hid the model->tool->model cycle learners
                #   wired by hand in LangGraph. result["messages"] is the FULL
                #   transcript of that cycle: walk it in order to SEE each turn
                #   -- the human question, the AI message that REQUESTS the
                #   tool, the ToolMessage carrying the tool's return value, and
                #   the final AI answer. This never raises; it only reads the
                #   messages the loop already produced.
                # -----------------------------------------------------------
                try:
                    messages = result["messages"]
                    print("\n[under the hood] the agent loop, message by message")
                    model_steps = 0
                    for i, msg in enumerate(messages):
                        role = type(msg).__name__  # HumanMessage/AIMessage/ToolMessage
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        if role == "AIMessage":
                            model_steps += 1
                        if tool_calls:
                            # AI turn that DECIDES to call a tool (no answer yet).
                            calls = [f"{c['name']}(args={c['args']})" for c in tool_calls]
                            print(f"[under the hood]   [{i}] {role} -> "
                                  f"REQUESTS tool: {', '.join(calls)}")
                        elif role == "ToolMessage":
                            # The tool ran; this carries its returned value back in.
                            name = getattr(msg, "name", "?")
                            value = (getattr(msg, "content", "") or "").strip()
                            print(f"[under the hood]   [{i}] {role} -> "
                                  f"{name} returned: {value}")
                        else:
                            # Human question or a plain AI text turn (the answer).
                            text = (getattr(msg, "content", "") or "").strip()
                            print(f"[under the hood]   [{i}] {role} -> {text}")
                    print(f"[under the hood] loop took {len(messages)} messages "
                          f"across {model_steps} model step(s) "
                          "(model -> tool -> model, compiled by create_agent).")
                except Exception as exc:  # never let the teaching demo crash
                    print(f"[under the hood] (loop unroll unavailable: {exc})")
        except Exception as exc:  # keep the demo resilient even if nothing caught it
            print(f"[live run failed: {short_api_error(str(exc))}]")

    print("\n" + "=" * 70)
    print("Done. A few lines of create_agent == a LangGraph agent loop, compiled.")
    print("=" * 70)
