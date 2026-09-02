# LangGraph Studio demo graph — a MULTI-AGENT text adventure.
#
# Four agents share one table (the state). One is the boss, three are specialists:
#
#   game_master  SUPERVISOR  reads the player's move, decides WHO acts next  -> writes scene_type
#   narrator     specialist  explores: describes the world, moves the plot   -> writes a message
#   combat       specialist  resolves fights and hands out loot              -> message + inventory
#   npc          specialist  voices the characters the player talks to      -> message + inventory
#
#   START -> game_master -> (route_by_scene) -> { narrator | combat | npc } -> END
#
# The teaching points, in the same vocabulary as the rest of the session:
#   - an AGENT is just a node with its own role (system prompt)   -> (state) -> partial dict
#   - the SUPERVISOR is also an agent: an LLM deciding who goes next; it writes its
#     decision into state (scene_type) and never talks to the player
#   - the HAND-OFF is a conditional edge that reads that decision
#   - the agents COLLABORATE through the shared state: `inventory` has a reducer
#     (operator.add), so loot the combat agent adds is visible to the npc agent's prompt
#   - built on MessagesState so LangGraph Studio's Chat tab works; threads keep the story
#
# `langgraph dev` discovers the module-level compiled `graph` via langgraph.json.
# No API key -> canned scenes (routing still works); with OPENROUTER_API_KEY in
# studio/.env every agent is a real model call.
import operator
import os
import re
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState

load_dotenv()


# --- The shared table: State ----------------------------------------------------
class GameState(MessagesState):
    """MessagesState gives us `messages` (with the add_messages reducer, so the
    Chat tab works). We add three more keys the agents share."""

    scene_type: str                                  # game_master's decision (replaced each turn)
    inventory: Annotated[list[str], operator.add]    # loot; reducer APPENDS (combat/npc add items)
    turn: Annotated[int, operator.add]               # each specialist returns 1 -> counter climbs


SceneType = Literal["explore", "combat", "dialogue"]


# --- Each agent's role -------------------------------------------------------------
GAME_MASTER_ROLE = (
    "You are the Game Master of a text adventure. Read the player's latest move and "
    "decide which specialist should handle it. Reply with EXACTLY one word:\n"
    "  explore  - moving, looking around, travelling, searching, anything else\n"
    "  combat   - attacking, fighting, defending, fleeing from danger\n"
    "  dialogue - talking to, asking, greeting, or persuading a character"
)

NARRATOR_ROLE = (
    "You are the Narrator of a short, fun fantasy adventure. Describe the world and move "
    "the story forward in 4-6 vivid sentences. Always include at least one character the "
    "player could talk to and one danger they could fight. End with 2-3 numbered choices."
)

COMBAT_ROLE = (
    "You are the Combat Master. Resolve the player's fight in 3-5 punchy, cinematic "
    "sentences - the player should win, but at a cost or with a twist. End with 2-3 "
    "numbered choices. Then, on the very last line, write LOOT: followed by at most two "
    "short item names the player gained (objects only, e.g. LOOT: rusty dagger, wolf pelt), "
    "or LOOT: none."
)

NPC_ROLE = (
    "You are the voice of every non-player character. Reply IN CHARACTER as the person "
    "the player addressed: 3-5 sentences of dialogue that reveal a hint, a rumour, or a "
    "secret. End with 2-3 numbered choices. Then, on the very last line, write LOOT: "
    "followed by at most one short item name if the character gives the player something "
    "(objects only, e.g. LOOT: brass key), or LOOT: none."
)


# --- Shared plumbing ----------------------------------------------------------------
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
        temperature=0.8,
    )


def _last_move(state: GameState) -> str:
    """The player's most recent message."""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _speak(role: str, state: GameState) -> str | None:
    """Ask the model to act in a role, seeing the whole conversation AND the shared
    inventory. Returns None when there is no key (the caller uses a canned scene)."""
    llm = _llm()
    if llm is None:
        return None
    carried = ", ".join(state.get("inventory", [])) or "nothing yet"
    system = SystemMessage(content=f"{role}\n\nThe player currently carries: {carried}.")
    return llm.invoke([system] + state["messages"]).content


_LOOT_LINE = re.compile(r"^\s*LOOT:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)


def _split_loot(text: str) -> tuple[str, list[str]]:
    """Pull the trailing 'LOOT: a, b' line off a specialist's reply."""
    match = _LOOT_LINE.search(text)
    if not match:
        return text.strip(), []
    raw = match.group(1).strip()
    items = [] if raw.lower() in ("", "none", "nothing") else [i.strip() for i in raw.split(",") if i.strip()]
    return _LOOT_LINE.sub("", text).strip(), items


# --- Agent 1: the Game Master (supervisor) -------------------------------------------
COMBAT_WORDS = {"attack", "fight", "strike", "stab", "slash", "shoot", "punch", "kill",
                "sword", "draw", "defend", "flee", "run", "charge", "hit"}
TALK_WORDS = {"ask", "talk", "speak", "say", "tell", "greet", "hello", "hi", "who",
              "persuade", "bargain", "question", "call"}


def _rules_of_thumb(move: str) -> SceneType:
    """Offline fallback for the supervisor: keyword rules, like simple_graph.py."""
    words = {w.strip("?!.,'\"").lower() for w in move.split()}
    if words & COMBAT_WORDS:
        return "combat"
    if words & TALK_WORDS:
        return "dialogue"
    return "explore"


def game_master(state: GameState) -> dict:
    """SUPERVISOR. Decides who acts this turn. Writes ONLY scene_type - no message."""
    told_before = any(isinstance(m, AIMessage) for m in state["messages"])
    if not told_before:
        return {"scene_type": "explore"}          # the opening scene is always the narrator's

    move = _last_move(state)
    llm = _llm()
    if llm is not None:
        verdict = llm.invoke([SystemMessage(content=GAME_MASTER_ROLE), HumanMessage(content=move)])
        first_word = re.findall(r"[a-z]+", verdict.content.lower())[:1]
        if first_word and first_word[0] in ("explore", "combat", "dialogue"):
            return {"scene_type": first_word[0]}
    return {"scene_type": _rules_of_thumb(move)}


# --- The hand-off: a conditional edge that reads the supervisor's decision -----------
def route_by_scene(state: GameState) -> Literal["narrator", "combat", "npc"]:
    return {"explore": "narrator", "combat": "combat", "dialogue": "npc"}[
        state.get("scene_type", "explore")
    ]


# --- Agents 2-4: the specialists ------------------------------------------------------
OFFLINE_OPENING = (
    "Dawn breaks over a muddy crossroads. To the north, a village chimney smokes; to the "
    "east, a black tower leans against the sky. A grey wolf watches you from the treeline, "
    "and an old woman with a walking stick hums beside a milestone. "
    "(Set OPENROUTER_API_KEY in studio/.env for an AI-written story.)\n"
    "1) Follow the road to the village   2) Draw your sword on the wolf   "
    "3) Ask the old woman about the tower"
)
OFFLINE_EXPLORE = (
    "You press on. The road narrows into a hollow of dripping trees where something has "
    "scratched a warning into the bark. A tinker's cart stands abandoned, its owner "
    "arguing with a shadow that is not his own.\n"
    "1) Search the cart   2) Attack the shadow   3) Talk to the tinker"
)
OFFLINE_COMBAT = (
    "Steel rings. The wolf lunges, you sidestep, and one clean strike ends it - but not "
    "before its teeth tear your sleeve. Among the roots where it slept you find a rusty "
    "dagger and a strip of dried meat.\n"
    "1) Search the den further   2) Head for the village   3) Call out to whoever is watching\n"
    "LOOT: rusty dagger, dried meat"
)
OFFLINE_NPC = (
    "The old woman stops humming. \"The tower? Nobody who climbs it comes back the same. "
    "But the door only opens for the ones who ask nicely.\" She presses a cold brass key "
    "into your palm and winks.\n"
    "1) Ask what the key opens   2) Thank her and head to the tower   3) Ask about the wolf\n"
    "LOOT: brass key"
)


def narrator(state: GameState) -> dict:
    """Explores. Reads the shared conversation, writes one new scene."""
    text = _speak(NARRATOR_ROLE, state)
    if text is None:
        text = OFFLINE_OPENING if state.get("turn", 0) == 0 else OFFLINE_EXPLORE
    return {"messages": [AIMessage(content=text)], "turn": 1}


def combat(state: GameState) -> dict:
    """Resolves fights. Writes a scene AND appends loot to the shared inventory."""
    text = _speak(COMBAT_ROLE, state) or OFFLINE_COMBAT
    scene, loot = _split_loot(text)
    return {"messages": [AIMessage(content=scene)], "inventory": loot, "turn": 1}


def npc(state: GameState) -> dict:
    """Voices characters. Writes dialogue AND may hand the player an item."""
    text = _speak(NPC_ROLE, state) or OFFLINE_NPC
    scene, loot = _split_loot(text)
    return {"messages": [AIMessage(content=scene)], "inventory": loot, "turn": 1}


# --- Build + compile -------------------------------------------------------------------
builder = StateGraph(GameState)
builder.add_node("game_master", game_master)
builder.add_node("narrator", narrator)
builder.add_node("combat", combat)
builder.add_node("npc", npc)

builder.add_edge(START, "game_master")                       # every turn starts with the supervisor
builder.add_conditional_edges(                               # ...who hands off to ONE specialist
    "game_master", route_by_scene,
    {"narrator": "narrator", "combat": "combat", "npc": "npc"},
)
builder.add_edge("narrator", END)
builder.add_edge("combat", END)
builder.add_edge("npc", END)

# Expose the compiled graph for `langgraph dev` / Studio to import.
# No checkpointer here - the dev server provides its own persistence (threads),
# which is what keeps the adventure going across turns.
graph = builder.compile()


if __name__ == "__main__":
    # Quick local smoke test (outside Studio). We add our own checkpointer so the
    # three turns share a thread, exactly like memory_demo.py.
    from langgraph.checkpoint.memory import InMemorySaver

    test = builder.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "smoke"}}
    for move in ("Start a fantasy adventure",
                 "I draw my sword and attack the wolf",
                 "I ask the old woman about the tower"):
        s = test.invoke({"messages": [HumanMessage(content=move)]}, config=cfg)
        print(f"\n> {move}")
        print(f"  scene_type={s['scene_type']:9} turn={s['turn']} inventory={s['inventory']}")
        print("  " + s["messages"][-1].content[:160].replace("\n", " | "))
