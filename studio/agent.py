# LangGraph Studio demo graph — a MULTI-AGENT CUSTOMER SUPPORT DESK.
#
# The shape of the classic "customer support agent" LangGraph tutorial
# (categorize -> analyze_sentiment -> route -> handler / escalate), made VISIBLE:
# every agent posts its own named message, and the whole thing runs in Studio's Chat tab.
#
#   categorize         tiny LLM call: Billing / Technical / General        -> category, ticket_id
#   analyze_sentiment  tiny LLM call: Positive / Neutral / Negative        -> sentiment, priority
#   route_query        conditional edge: Negative -> escalate; else by category
#   handle_general     front desk: greetings / general questions (no ticket)
#   handle_billing     Billing specialist: writes the customer reply, logs the ticket
#   handle_technical   Technical specialist: writes the customer reply, logs the ticket
#   escalate           angry customers go straight to a human
#
#   START -> categorize -> analyze_sentiment -> route_query -> escalate         -> END
#                                                          -> handle_general   -> END
#                                                          -> handle_billing   -> END
#                                                          -> handle_technical -> END
#
# Teaching points, in the same vocabulary as the rest of the session:
#   - an AGENT is a node with its own role (prompt); same (state) -> partial dict shape
#   - small analysis nodes write DECISIONS into state; one conditional edge reads BOTH (route_query)
#   - sentiment is a routing signal: angry customers skip the bots
#   - agents collaborate through SHARED STATE (category, sentiment, ticket_id); `log` uses the
#     operator.add reducer so tickets pile up across a thread
#   - built on MessagesState so Studio's Chat tab works; each thread is one customer
#
# `langgraph dev` discovers the module-level compiled `graph` via langgraph.json.
# No API key -> keyword classifiers and canned replies (every path still runs).
# With OPENROUTER_API_KEY in studio/.env every node is a real model call.
import operator
import os
import random
import sys
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END, MessagesState

load_dotenv()

COMPANY = "Nimbus"      # a fictional note-taking app with monthly / yearly plans


# --- State: the table every agent reads from and writes to -----------------------------------
class State(MessagesState):
    """MessagesState gives us `messages` (add_messages reducer -> the Chat tab works).
    The rest is the team's shared scratchpad, one key per decision."""

    ticket_id: str                          # opened by categorize for real issues, then kept
    category: str                           # billing | technical | general
    sentiment: str                          # positive | neutral | negative
    priority: str                           # normal | high
    log: Annotated[list[str], operator.add] # one line per handled/escalated ticket; APPENDS


# --- Shared plumbing ------------------------------------------------------------------------------
def _llm(temperature: float = 0.4):
    """An OpenRouter-backed chat model, or None when no key is set.
    Classifiers run at temperature 0; writers get a little warmth."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                      base_url="https://openrouter.ai/api/v1", api_key=api_key,
                      temperature=temperature)


def _ask(template: str, temperature: float = 0.4, **inputs) -> str | None:
    """prompt | llm, exactly like the tutorial's `chain = prompt | ChatOpenAI(...)`.
    Returns None when there is no key, so the caller falls back to rules / canned text."""
    llm = _llm(temperature)
    if llm is None:
        return None
    chain = ChatPromptTemplate.from_template(template) | llm
    return chain.invoke(inputs).content.strip()


def _query(state: State) -> str:
    """The customer's most recent message."""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _say(agent: str, text: str) -> AIMessage:
    """Every agent posts under its own name, so the teamwork is visible in Chat."""
    return AIMessage(content=f"[{agent}] {text}", name=agent.lower().replace(" ", "_"))


def _words(text: str) -> set[str]:
    return {w.strip("?!.,'\"()").lower() for w in text.split()}


# --- Node 1: categorize ------------------------------------------------------------------------------
BILLING_WORDS = {"charge", "charged", "invoice", "refund", "payment", "card", "bill", "billed",
                 "price", "plan", "subscription", "renewed", "twice", "double", "money", "receipt"}
TECH_WORDS = {"error", "crash", "crashes", "crashed", "bug", "broken", "sync", "syncing", "slow",
              "freezes", "loading", "install", "update", "notes", "lost", "missing", "app",
              "password", "login", "log", "locked", "email", "reset"}


def categorize(state: State) -> dict:
    """Classify the query as Billing, Technical, or General. Opens a ticket for real issues."""
    query = _query(state)
    answer = _ask(
        "Categorize the following customer query into one of these categories: "
        "Technical, Billing, General. General means a greeting, thanks, small talk, or a "
        "question with no actual problem to solve. Reply with the single category word.\n"
        "Query: {query}", temperature=0, query=query)
    words = _words(query)
    if answer and answer.lower().split()[0].strip(".") in ("technical", "billing", "general"):
        category = answer.lower().split()[0].strip(".")
    elif words & BILLING_WORDS:
        category = "billing"
    elif words & TECH_WORDS:
        category = "technical"
    else:
        category = "general"

    if category == "general":
        return {"category": category,
                "messages": [_say("Categorizer", "Category: General - no ticket needed.")]}
    ticket_id = state.get("ticket_id") or f"#{random.randint(4000, 4999)}"
    return {"category": category, "ticket_id": ticket_id,
            "messages": [_say("Categorizer", f"Category: {category.title()} · ticket {ticket_id} opened.")]}


# --- Node 2: analyze_sentiment -------------------------------------------------------------------
ANGRY_WORDS = {"unacceptable", "furious", "angry", "ridiculous", "terrible", "worst", "scam",
               "disgusting", "useless", "cancel", "lawyer", "third", "again"}


def analyze_sentiment(state: State) -> dict:
    """Positive / Neutral / Negative. Negative = upset wording, not merely 'has a problem'."""
    query = _query(state)
    answer = _ask(
        "Analyze the sentiment of the following customer query. Respond with either "
        "'Positive', 'Neutral', or 'Negative'. Negative means the customer sounds angry, "
        "frustrated or upset (insults, 'unacceptable', threats to cancel, repeated failures). "
        "A calm description of a problem is Neutral.\nQuery: {query}", temperature=0, query=query)
    if answer and answer.lower().split()[0].strip(".") in ("positive", "neutral", "negative"):
        sentiment = answer.lower().split()[0].strip(".")
    else:
        sentiment = "negative" if (_words(query) & ANGRY_WORDS or "!!" in query) else "neutral"
    priority = "high" if sentiment == "negative" else "normal"

    category = state["category"]
    if category == "general":
        note = "→ front desk will reply."
    elif sentiment == "negative":
        note = "→ priority high. Escalating to a human agent."
    else:
        note = f"→ priority {priority} → handing to {category.title()}."
    return {"sentiment": sentiment, "priority": priority,
            "messages": [_say("Sentiment", f"Tone: {sentiment.title()} {note}")]}


# --- The hand-off: one conditional edge reads both decisions -------------------------------------
def route_query(state: State) -> Literal["escalate", "handle_general", "handle_billing", "handle_technical"]:
    """Negative sentiment on a real issue -> a human. Otherwise route by category."""
    if state["category"] == "general":
        return "handle_general"
    if state["sentiment"] == "negative":
        return "escalate"
    return f"handle_{state['category']}"


# --- Handlers: the agents that actually write to the customer -----------------------------------------
def handle_general(state: State) -> dict:
    """Front desk. Greetings and general questions - no ticket, the specialists stay asleep."""
    query = _query(state)
    text = _ask(
        f"You are the front desk of {COMPANY} support ({COMPANY} is a note-taking app). "
        "Reply to the customer in one or two friendly sentences. If they haven't described "
        "a problem, ask what they need help with and mention you can route billing or "
        "technical questions to a specialist. No ticket numbers, no promises.\n"
        "Customer: {query}", query=query)
    if text is None:
        text = ("You're welcome! If anything else comes up, just tell me what's going on."
                if "thank" in query.lower() else
                f"Hi! You've reached {COMPANY} support. Tell me what's going on - a charge, or "
                "the app misbehaving - and I'll route it to the right specialist.")
    return {"messages": [_say("Front desk", text)]}


# House style every specialist follows. Given to them up front, with the ticket ID.
GUIDELINES = (
    "- Quote the ticket ID once.\n"
    "- Never guarantee a refund, credit, fix, or timeline: say a refund request has been "
    "opened / the issue has been logged, and that reversals typically take 5-7 business days.\n"
    "- Give the customer one clear next step, or ask them one question.\n"
    "- Under 90 words, warm but not gushing, and never blame the customer."
)

SPECIALIST = {
    "billing": (
        f"the Billing specialist at {COMPANY}, a note-taking app with monthly and yearly plans",
        "Hi, thanks for flagging this (ticket {ticket}). I can see two charges on your last "
        "invoice, and I've opened a refund request for the duplicate; reversals typically take "
        "5-7 business days. Could you confirm the last four digits of the card so I can "
        "prioritise it?"),
    "technical": (
        f"the Technical Support specialist at {COMPANY}, a note-taking app for web, iOS and Android",
        "Hi, thanks for the report (ticket {ticket}). I've logged the issue with our mobile "
        "team. To help them reproduce it, could you tell me your app version (Settings → About) "
        "and whether it happens on Wi-Fi as well as mobile data?"),
}


def _reply(category: str, state: State) -> dict:
    """Shared body for the two specialists: write the reply, log the ticket."""
    ticket, priority, query = state["ticket_id"], state["priority"], _query(state)
    role, canned = SPECIALIST[category]
    text = _ask("You are {role}. Write the customer-facing reply to the query below, following "
                "these guidelines:\n{guidelines}\nThe ticket ID is {ticket}.\n"
                "Reply with the message only - no preamble.\nQuery: {query}",
                role=role, guidelines=GUIDELINES, ticket=ticket, query=query)
    if text is None:
        text = canned.format(ticket=ticket)
    return {"log": [f"{ticket} {category}/{priority} · replied by {category.title()}"],
            "messages": [_say(category.title(), text)]}


def handle_billing(state: State) -> dict:
    return _reply("billing", state)


def handle_technical(state: State) -> dict:
    return _reply("technical", state)


def escalate(state: State) -> dict:
    """Angry customer: skip the bots, hand the ticket to a person."""
    ticket, cat = state["ticket_id"], state["category"]
    return {"log": [f"{ticket} {cat}/high · ESCALATED to a human (negative sentiment)"],
            "messages": [_say("Escalation", f"Ticket {ticket} handed to a human agent because of its "
                                            "negative sentiment. A person will reply within one business day.")]}


# --- Build + compile ---------------------------------------------------------------------------------
workflow = StateGraph(State)

workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_general", handle_general)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("escalate", escalate)

workflow.add_edge(START, "categorize")                              # modern entry point (not set_entry_point)
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges("analyze_sentiment", route_query, {   # the hand-off
    "escalate": "escalate",
    "handle_general": "handle_general",
    "handle_billing": "handle_billing",
    "handle_technical": "handle_technical",
})
workflow.add_edge("handle_general", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_technical", END)
workflow.add_edge("escalate", END)

# Expose the compiled graph for `langgraph dev` / Studio to import.
# No checkpointer here - the dev server provides its own persistence (threads),
# so each Studio thread is one customer conversation.
graph = workflow.compile()


# --- Run it outside Studio ------------------------------------------------------------------------
def run_customer_support(query: str, thread_id: str = "demo") -> dict:
    """Process one customer query and return what each agent decided (tutorial-style helper).
    Uses a local checkpointer so several calls with the same thread_id share a ticket."""
    from langgraph.checkpoint.memory import InMemorySaver

    app = run_customer_support.app = getattr(run_customer_support, "app", None) or \
        workflow.compile(checkpointer=InMemorySaver())
    result = app.invoke({"messages": [HumanMessage(content=query)]},
                        config={"configurable": {"thread_id": thread_id}})
    return {
        "category": result.get("category"),
        "sentiment": result.get("sentiment"),
        "response": result["messages"][-1].content,
        "transcript": [m.content for m in result["messages"] if isinstance(m, AIMessage)],
        "log": result.get("log", []),
    }


if __name__ == "__main__":
    if "--mermaid" in sys.argv:                 # paste into a slide: the graph as Mermaid
        print(graph.get_graph().draw_mermaid())
        raise SystemExit(0)

    queries = [
        "hi",
        "I was charged twice this month for the yearly plan, can you check?",
        "The iOS app crashes every time I open a note with an image in it.",
        "This is the THIRD time you've double-charged me. Unacceptable. Refund me now.",
        "thanks, that's all!",
    ]
    for i, q in enumerate(queries, 1):
        r = run_customer_support(q, thread_id=f"customer-{i}")
        print(f"\nQuery: {q}\nCategory: {r['category']}   Sentiment: {r['sentiment']}")
        for line in r["transcript"]:
            print("  " + line[:160].replace("\n", " "))
        if r["log"]:
            print(f"  log: {r['log']}")
