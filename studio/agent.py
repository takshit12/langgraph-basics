# LangGraph Studio demo graph — a MULTI-AGENT CUSTOMER SUPPORT DESK.
#
# You paste a customer message. A team of agents handles it, and EVERY agent posts its
# own named message, so the teamwork is visible in Studio's Chat tab:
#
#   triage     SUPERVISOR   classifies the ticket, sets priority, assigns a ticket ID,
#                           and hands off to ONE specialist            -> category, priority, ticket_id
#   billing    specialist   drafts the customer reply (money questions)  -> draft
#   tech       specialist   drafts the customer reply (bugs / errors)    -> draft
#   account    specialist   drafts the customer reply (login / profile)  -> draft
#   reviewer   POLICY GATE  checks the draft against the rulebook and either
#                           APPROVES it or REJECTS it with notes        -> verdict, notes
#   closer     wrap-up      sends the approved reply and logs the ticket,
#                           or escalates to a human after MAX_ROUNDS     -> log
#
#   START -> triage -> { billing | tech | account } -> reviewer -> approved? -> closer -> END
#                             ^                           |
#                             └──── rejected (rounds < MAX_ROUNDS) ─────┘
#
# Teaching points, in the same vocabulary as the rest of the session:
#   - an AGENT is a node with its own role (system prompt); same (state) -> partial dict shape
#   - the SUPERVISOR (triage) writes its decision into state; a conditional edge does the hand-off
#   - the REVIEW LOOP is a conditional edge that points BACKWARDS; MAX_ROUNDS is the safety cap
#   - agents collaborate through SHARED STATE: the reviewer's `notes` become the specialist's
#     instructions on the next round; `log` uses the operator.add reducer
#   - the reviewer is part CODE (hard rules, deterministic) and part LLM (judgement) -
#     the same split as exact-match vs LLM-as-a-judge in agent-eval.py
#   - built on MessagesState so Studio's Chat tab works; each thread is one customer
#
# `langgraph dev` discovers the module-level compiled `graph` via langgraph.json.
# No API key -> keyword triage, canned drafts, rule-based review (the loop still runs).
# With OPENROUTER_API_KEY in studio/.env every agent is a real model call.
import operator
import os
import random
import re
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState

load_dotenv()

MAX_ROUNDS = 3          # draft + up to two revisions; after that the closer escalates to a human
COMPANY = "Nimbus"      # a fictional note-taking app with monthly / yearly plans


# --- The shared table: State ---------------------------------------------------------
class DeskState(MessagesState):
    """MessagesState gives us `messages` (add_messages reducer -> the Chat tab works).
    Everything else is the team's shared scratchpad."""

    ticket_id: str                          # assigned by triage on the first turn, then kept
    category: Literal["billing", "tech", "account"]   # triage's routing decision
    priority: str                           # low | normal | high
    draft: str                              # the specialist's current draft reply
    rounds: int                             # how many drafts this turn (reset by triage)
    verdict: str                            # reviewer: approved | rejected
    notes: str                              # reviewer's reasons (fed back to the specialist)
    log: Annotated[list[str], operator.add] # one line per closed ticket; reducer APPENDS


# --- Roles ------------------------------------------------------------------------------
TRIAGE_ROLE = (
    f"You are the Triage lead on the {COMPANY} support desk. Read the customer's message "
    "and reply with EXACTLY two words: <category> <priority>.\n"
    "  category: billing (charges, invoices, refunds, plans, payment) | "
    "tech (errors, crashes, sync, things not working) | "
    "account (login, password, email, profile, deleting the account)\n"
    "  priority: high (angry, urgent, money lost, locked out) | normal | low"
)

SPECIALIST_ROLE = {
    "billing": f"You are the Billing specialist at {COMPANY}, a note-taking app with monthly and "
               "yearly plans. Write the customer-facing reply to the message below. Be warm, "
               "specific and brief.",
    "tech":    f"You are the Technical Support specialist at {COMPANY}, a note-taking app for web, "
               "iOS and Android. Write the customer-facing reply to the message below. Be warm, "
               "specific and brief.",
    "account": f"You are the Account specialist at {COMPANY}, a note-taking app. You handle logins, "
               "passwords, emails and profiles. Write the customer-facing reply to the message "
               "below. Be warm, specific and brief.",
}

# The rulebook lives with the REVIEWER. Specialists only learn it through feedback -
# which is exactly why the review loop shows itself on the first draft.
POLICY = (
    "1. Every reply must quote the ticket ID (e.g. #4821).\n"
    "2. Never guarantee a refund, credit, fix, or timeline. Say a refund request has been "
    "opened / the issue has been logged, and that reversals typically take 5-7 business days.\n"
    "3. Give the customer exactly ONE concrete next step or one question.\n"
    "4. Under 90 words. At most one exclamation mark.\n"
    "5. Never blame the customer or tell them to 'just' do something."
)

REVIEWER_ROLE = (
    f"You are the Policy Reviewer on the {COMPANY} support desk. Check the draft reply against "
    "the rulebook. Reject ONLY for a clear breach of a numbered rule; do not invent rules. "
    "Greetings and sign-offs never count as a 'next step'. Reply with exactly one line: "
    "'APPROVED' or 'REJECTED: <the rules it breaks, briefly>'.\n\nRULEBOOK:\n" + POLICY
)

# Hard rules the reviewer checks IN CODE, so they never slip through.
BANNED = re.compile(
    r"\b(guarantee[ds]?|full refund|within 24 hours|immediately|right away|straight away)\b",
    re.IGNORECASE,
)


# --- Shared plumbing --------------------------------------------------------------------
def _llm():
    """An OpenRouter-backed chat model, or None when no key is set."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0.4,
    )


def _customer_message(state: DeskState) -> str:
    """The customer's most recent message."""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _say(agent: str, text: str) -> AIMessage:
    """Every agent posts under its own name so the teamwork is visible in Chat."""
    return AIMessage(content=f"[{agent}] {text}", name=agent.lower())


# --- Agent 1: Triage (supervisor) -----------------------------------------------------------
BILLING_WORDS = {"charge", "charged", "invoice", "refund", "payment", "card", "bill", "billed",
                 "price", "plan", "subscription", "renewed", "twice", "double"}
TECH_WORDS = {"error", "crash", "crashes", "crashed", "bug", "broken", "sync", "syncing",
              "slow", "freezes", "loading", "install", "update", "notes", "lost", "missing"}
ACCOUNT_WORDS = {"password", "login", "log", "email", "username", "delete", "profile",
                 "locked", "verify", "verification", "2fa", "signin", "sign"}
URGENT_WORDS = {"urgent", "asap", "angry", "furious", "unacceptable", "immediately", "now",
                "locked", "lost", "twice", "again", "third"}


def _words(text: str) -> set[str]:
    return {w.strip("?!.,'\"()").lower() for w in text.split()}


def _rules_triage(message: str) -> tuple[str, str]:
    """Offline fallback: keyword rules, exactly like classify() in simple_graph.py."""
    words = _words(message)
    if words & BILLING_WORDS:
        category = "billing"
    elif words & ACCOUNT_WORDS:
        category = "account"
    elif words & TECH_WORDS:
        category = "tech"
    else:
        category = "tech"
    priority = "high" if words & URGENT_WORDS else "normal"
    return category, priority


def triage(state: DeskState) -> dict:
    """SUPERVISOR. Classifies the ticket and decides which specialist gets it."""
    message = _customer_message(state)
    category, priority = _rules_triage(message)
    llm = _llm()
    if llm is not None:
        verdict = llm.invoke([SystemMessage(content=TRIAGE_ROLE), HumanMessage(content=message)])
        words = re.findall(r"[a-z]+", verdict.content.lower())
        if len(words) >= 2 and words[0] in SPECIALIST_ROLE and words[1] in ("low", "normal", "high"):
            category, priority = words[0], words[1]
    ticket_id = state.get("ticket_id") or f"#{random.randint(4000, 4999)}"
    who = {"billing": "Billing", "tech": "Tech", "account": "Account"}[category]
    return {
        "ticket_id": ticket_id,
        "category": category,
        "priority": priority,
        "rounds": 0,            # a new customer message starts a fresh review cycle
        "verdict": "",
        "notes": "",
        "draft": "",
        "messages": [_say("Triage", f"Ticket {ticket_id} · category: {category} · priority: "
                                    f"{priority} → handing to {who}.")],
    }


def route_by_category(state: DeskState) -> Literal["billing", "tech", "account"]:
    """The hand-off: a conditional edge that reads the supervisor's decision."""
    return state["category"]


# --- Agents 2-4: the specialists (one node each; same shape, different role) -------------------
OFFLINE_DRAFTS = {
    # round 1 (fast draft: forgets the ticket ID, over-promises)  /  round 2 (fixed)
    "billing": (
        "Hi! So sorry about the double charge. You'll get a full refund within 24 hours, "
        "guaranteed. Cheers!",
        "Hi, thanks for flagging this (ticket {ticket}). I can see two charges on your last "
        "invoice, and I've opened a refund request for the duplicate; reversals typically "
        "take 5-7 business days. Could you confirm the last four digits of the card so I can "
        "prioritise it?",
    ),
    "tech": (
        "Hey! That sync bug is annoying, sorry. Just reinstall the app and it'll be fixed "
        "immediately. Let me know!",
        "Hi, thanks for the report (ticket {ticket}). I've logged the sync issue with our "
        "mobile team. To help them reproduce it, could you tell me your app version "
        "(Settings → About) and whether it happens on Wi-Fi as well as mobile data?",
    ),
    "account": (
        "Hi! No worries, I've reset it and you'll be back in immediately. Just check your "
        "inbox!",
        "Hi, sorry you're locked out (ticket {ticket}). I've sent a password-reset link to "
        "the email on the account; it stays valid for 30 minutes. If it doesn't arrive, "
        "could you check your spam folder and let me know?",
    ),
}


def _draft(category: str, state: DeskState) -> dict:
    """Shared body for the three specialists: draft (round 1) or revise (round 2+)."""
    round_no = state.get("rounds", 0) + 1
    ticket = state["ticket_id"]
    who = {"billing": "Billing", "tech": "Tech", "account": "Account"}[category]
    customer = _customer_message(state)

    llm = _llm()
    if llm is None:
        first, fixed = OFFLINE_DRAFTS[category]
        text = first if round_no == 1 else fixed.format(ticket=ticket)
    else:
        system = SPECIALIST_ROLE[category]
        if round_no > 1:
            # The reviewer's feedback comes back WITH the rulebook - agents learn the
            # policy through the loop, which is what makes the hand-off visible.
            system += (
                f"\n\nYour previous draft was REJECTED by the Policy Reviewer.\n"
                f"Previous draft: {state['draft']}\n"
                f"Reviewer notes: {state['notes']}\n\n"
                f"Rewrite it to satisfy EVERY rule below. The ticket ID is {ticket}.\n"
                f"RULEBOOK:\n{POLICY}\n\nReply with the new draft only."
            )
        else:
            system += " Reply with the draft only - no preamble."
        text = llm.invoke([SystemMessage(content=system), HumanMessage(content=customer)]).content.strip()

    return {
        "draft": text,
        "rounds": round_no,
        "messages": [_say(who, f"Draft v{round_no}: {text}")],
    }


def billing(state: DeskState) -> dict:
    return _draft("billing", state)


def tech(state: DeskState) -> dict:
    return _draft("tech", state)


def account(state: DeskState) -> dict:
    return _draft("account", state)


# --- Agent 5: the Policy Reviewer (part code, part LLM) ------------------------------------------
def reviewer(state: DeskState) -> dict:
    """Checks the draft against the rulebook. Hard rules in code; judgement by the model."""
    draft, ticket = state["draft"], state["ticket_id"]
    reasons: list[str] = []

    # Hard rules - deterministic, like the exact-match evaluator in agent-eval.py.
    if ticket not in draft:
        reasons.append(f"missing the ticket ID {ticket}")
    banned = BANNED.findall(draft)
    if banned:
        reasons.append(f"promises '{banned[0]}' (policy: never guarantee refunds, fixes or timelines)")
    if len(draft.split()) > 90:
        reasons.append(f"{len(draft.split())} words (limit 90)")

    # Soft rules - judgement, like the LLM-as-a-judge evaluator.
    llm = _llm()
    if llm is not None and not reasons:
        verdict = llm.invoke([SystemMessage(content=REVIEWER_ROLE),
                              HumanMessage(content=f"Ticket ID: {ticket}\n\nDRAFT:\n{draft}")]).content.strip()
        if verdict.upper().startswith("REJECTED"):
            reasons.append(verdict.split(":", 1)[-1].strip() or "policy breach")

    who = {"billing": "Billing", "tech": "Tech", "account": "Account"}[state["category"]]
    if reasons:
        notes = "; ".join(reasons)
        return {"verdict": "rejected", "notes": notes,
                "messages": [_say("Reviewer", f"❌ Rejected v{state['rounds']} — {notes} → back to {who}.")]}
    return {"verdict": "approved", "notes": "",
            "messages": [_say("Reviewer", f"✅ Approved v{state['rounds']} — meets policy.")]}


def route_after_review(state: DeskState) -> Literal["billing", "tech", "account", "closer"]:
    """The loop: rejected drafts go BACK to the same specialist, until MAX_ROUNDS."""
    if state["verdict"] == "approved" or state["rounds"] >= MAX_ROUNDS:
        return "closer"
    return state["category"]


# --- Agent 6: the Closer -------------------------------------------------------------------------
def closer(state: DeskState) -> dict:
    """Sends the approved reply, or escalates. Appends one line to the shared log."""
    ticket, cat, pri, rounds = state["ticket_id"], state["category"], state["priority"], state["rounds"]
    if state["verdict"] == "approved":
        line = f"{ticket} {cat}/{pri} · replied after {rounds} review round(s)"
        text = f"Reply sent to customer · ticket {ticket} logged ({cat} / {pri} / {rounds} review round(s))."
    else:
        line = f"{ticket} {cat}/{pri} · ESCALATED to a human after {rounds} rejected drafts"
        text = (f"Could not get a compliant draft in {rounds} rounds → ticket {ticket} escalated "
                f"to a human agent. Last reviewer notes: {state['notes']}")
    return {"log": [line], "messages": [_say("Desk", text)]}


# --- Build + compile --------------------------------------------------------------------------------
builder = StateGraph(DeskState)
builder.add_node("triage", triage)
builder.add_node("billing", billing)
builder.add_node("tech", tech)
builder.add_node("account", account)
builder.add_node("reviewer", reviewer)
builder.add_node("closer", closer)

builder.add_edge(START, "triage")                                        # supervisor first
builder.add_conditional_edges("triage", route_by_category,               # hand-off to ONE specialist
                              {"billing": "billing", "tech": "tech", "account": "account"})
builder.add_edge("billing", "reviewer")                                  # every draft gets reviewed
builder.add_edge("tech", "reviewer")
builder.add_edge("account", "reviewer")
builder.add_conditional_edges("reviewer", route_after_review,            # approve -> closer, reject -> loop back
                              {"billing": "billing", "tech": "tech", "account": "account", "closer": "closer"})
builder.add_edge("closer", END)

# Expose the compiled graph for `langgraph dev` / Studio to import.
# No checkpointer here - the dev server provides its own persistence (threads),
# so each Studio thread is one customer conversation.
graph = builder.compile()


if __name__ == "__main__":
    # Quick local smoke test (outside Studio): three tickets, one per specialist.
    from langgraph.checkpoint.memory import InMemorySaver

    test = builder.compile(checkpointer=InMemorySaver())
    tickets = [
        "I was charged twice this month for the yearly plan and I want my money back, this is unacceptable.",
        "The iOS app crashes every time I open a note with an image in it.",
        "I can't log in - it says my password is wrong and I never got the reset email.",
    ]
    for i, msg in enumerate(tickets, 1):
        s = test.invoke({"messages": [HumanMessage(content=msg)]},
                        config={"configurable": {"thread_id": f"customer-{i}"}})
        print(f"\n=== customer {i}: {msg[:60]}...")
        for m in s["messages"]:
            if isinstance(m, AIMessage):
                print("  " + m.content[:150].replace("\n", " "))
        print(f"  state: category={s['category']} priority={s['priority']} rounds={s['rounds']} "
              f"verdict={s['verdict']} log={s['log']}")
