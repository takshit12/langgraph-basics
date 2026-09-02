import os
import operator
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langsmith import Client, evaluate
from langsmith.schemas import Run, Example

# ── Keys: loaded from the project .env (never hardcode secrets) ────────────────
load_dotenv()
OPENROUTER_KEY   = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
LANGSMITH_KEY    = os.getenv("LANGSMITH_API_KEY", "")

if not OPENROUTER_KEY:
    raise SystemExit("Set OPENROUTER_API_KEY in your .env (copy .env.example -> .env).")

# LangSmith tracing/evals are OPTIONAL: on only when a LangSmith key is present.
os.environ["LANGSMITH_TRACING"] = "true" if LANGSMITH_KEY else "false"
os.environ.setdefault("LANGSMITH_PROJECT", "simple-agent-eval")

# ── Build the agent ───────────────────────────────────────────────────────────

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

tools = [add, multiply]

llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
).bind_tools(tools)

judge_llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

def agent_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")                 # modern entry point (was set_entry_point)
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
agent = graph.compile()

# ── Quick local sanity run (works with just the OpenRouter key) ───────────────
_demo = agent.invoke({"messages": [HumanMessage(content="What is 3 * 4?")]})
print("Agent sanity check ->", _demo["messages"][-1].content)

# The LangSmith evaluations below need a LangSmith key. Without one, we stop here
# so the file still runs end-to-end for anyone who only has an OpenRouter key.
if not LANGSMITH_KEY:
    print(
        "\n[LangSmith key not set] Skipping the hosted evaluations.\n"
        "Add LANGSMITH_API_KEY to your .env to create the dataset, run the three\n"
        "evaluators (exact-match, LLM-as-judge, trajectory), and view them at\n"
        "https://smith.langchain.com"
    )
    raise SystemExit(0)

# ── Step 1: Create dataset (LangSmith) ────────────────────────────────────────

client = Client()
dataset_name = "math-agent-dataset"

examples = [
    {"inputs": {"question": "What is 2 + 3?"},  "outputs": {"answer": "5"}},
    {"inputs": {"question": "What is 4 + 6?"},  "outputs": {"answer": "10"}},
    {"inputs": {"question": "What is 3 * 4?"},  "outputs": {"answer": "12"}},
    {"inputs": {"question": "What is 5 * 5?"},  "outputs": {"answer": "25"}},
]

existing = list(client.list_datasets(dataset_name=dataset_name))
if not existing:
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Dataset created with {len(examples)} examples.")
else:
    print(f"Dataset already exists.")

# ── Step 2: Target function ───────────────────────────────────────────────────

def run_agent(inputs: dict) -> dict:
    result = agent.invoke({
        "messages": [HumanMessage(content=inputs["question"])]
    })
    return {"answer": result["messages"][-1].content}

# ── Step 3: Three evaluators ──────────────────────────────────────────────────

# --- Evaluator 1: Exact match ---
def exact_match(run: Run, example: Example) -> dict:
    predicted = run.outputs["answer"].strip()
    expected  = example.outputs["answer"].strip()
    return {"key": "exact_match", "score": int(predicted == expected)}

# --- Evaluator 2: LLM as a judge ---
def llm_judge(run: Run, example: Example) -> dict:
    prompt = f"""
Question: {example.inputs["question"]}
Expected answer: {example.outputs["answer"]}
Agent answer: {run.outputs["answer"]}

Is the agent's answer mathematically correct?
Reply with just 1 for correct or 0 for incorrect.
"""
    result = judge_llm.invoke(prompt)
    score  = 1 if "1" in result.content else 0
    return {"key": "llm_judge", "score": score}

# --- Evaluator 3: Trajectory (did the agent use a tool?) ---
def trajectory_evaluator(run: Run, example: Example) -> dict:
    expected = example.outputs["answer"].strip()
    answer   = run.outputs["answer"]
    # Check the correct number appears in the final answer
    score = int(expected in answer)
    return {"key": "trajectory", "score": score}

# ── Run all three evaluations ─────────────────────────────────────────────────

print("\n--- Exact Match ---")
evaluate(
    run_agent,
    data=dataset_name,
    evaluators=[exact_match],
    experiment_prefix="exact-match",
)

print("\n--- LLM as Judge ---")
evaluate(
    run_agent,
    data=dataset_name,
    evaluators=[llm_judge],
    experiment_prefix="llm-judge",
)

print("\n--- Trajectory ---")
evaluate(
    run_agent,
    data=dataset_name,
    evaluators=[trajectory_evaluator],
    experiment_prefix="trajectory",
)

print("\nDone! Visit smith.langchain.com to see all three results.")