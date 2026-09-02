"""
langchain_fundamentals.py
=========================

The zero-friction FIRST file for LangChain Basics -- self-study / reference,
NOT run live in the session (that role belongs to rag_demo.py).

It shows the four core building blocks from Section 3, one at a time, and the
single idea that ties them together (Section 1): *every component is a
`Runnable`, which is why you can pipe them with `|` regardless of provider.*

Building blocks demonstrated (each printed firing on its own at the bottom):
  1. Prompt Template   -- ChatPromptTemplate.from_template(...)   (reusable
                          instructions with `{variables}` filled at runtime)
  2. Output Parser     -- StrOutputParser()                       (turns the
                          model's raw message into a plain string)
  3. LCEL chain        -- prompt | model | parser                 (the `|` pipe;
                          works because all three are Runnables)
  4. Tool + bind_tools -- a @tool function the model can decide to call

NO API key is REQUIRED. Only the LLM step needs OPENROUTER_API_KEY:
  * WITH a key  -> the model step is a real langchain_openai.ChatOpenAI call.
  * WITHOUT a key -> we SWAP ONLY the model step for a RunnableLambda that
                     returns a canned string. The SAME `prompt | model | parser`
                     chain runs either way -- a deliberate, live demonstration
                     of Section 1's claim that "every component is a swappable
                     Runnable," not merely an offline workaround.

Run it:   python langchain_fundamentals.py
"""

import os

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import tool

# Pick up OPENROUTER_API_KEY / OPENROUTER_MODEL from the project .env (optional).
load_dotenv()
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")

# The one-line answer used BOTH by the offline fallback and when a live call
# fails -- kept in one place so both paths stay in sync.
OFFLINE_ANSWER = (
    "A Runnable is LangChain's standard component interface, which is why "
    "prompts, models, and parsers all pipe together with `|`."
)


def short_api_error(exc: Exception) -> str:
    """Turn a noisy API exception into one short, classroom-friendly line."""
    msg = str(exc)
    if "401" in msg or "User not found" in msg or "authentication" in msg.lower():
        return "401 auth error -- your OPENROUTER_API_KEY looks invalid or expired"
    return msg.splitlines()[0][:160]


# ---------------------------------------------------------------------------
# 1. PROMPT TEMPLATE  --  reusable instructions with runtime `{variables}`.
#    `.invoke({...})` fills the blanks and returns a ready-to-send message list.
# ---------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    "You are a concise teaching assistant. In ONE short sentence, explain: {topic}"
)


# ---------------------------------------------------------------------------
# 2. OUTPUT PARSER  --  turns the model's raw AIMessage into a plain str.
#    (Output parsers are what turn LLM text into JSON / lists / typed objects;
#     StrOutputParser is the simplest one -- it just extracts the text.)
# ---------------------------------------------------------------------------
parser = StrOutputParser()


# ---------------------------------------------------------------------------
# 3. THE MODEL STEP  --  a real ChatOpenAI, OR a swappable Runnable fallback.
#
#    KEY IDEA (Section 1): the chain below is `prompt | model | parser`. Every
#    one of those is a Runnable, so the model step is just another swappable
#    link. With no key we literally swap in a RunnableLambda that returns a
#    canned answer -- the surrounding chain does not change at all. That
#    swap-ability IS the lesson.
# ---------------------------------------------------------------------------
def build_model() -> Runnable:
    """Return the model step of the chain -- a genuine LangChain Runnable
    either way, so `prompt | model | parser` is identical with or without a key.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        # A genuine LangChain Runnable wrapping any OpenAI-compatible endpoint.
        # OpenRouter speaks the OpenAI API, so ChatOpenAI works via base_url.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=MODEL_NAME,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    # No key: swap in a Runnable that returns a canned string. Because it is a
    # Runnable, StrOutputParser downstream consumes it exactly like a real model
    # response. This is the "swappable component" thesis, demonstrated live.
    def _canned(prompt_value) -> str:
        # prompt_value is whatever the prompt step produced (a PromptValue).
        return "[offline fallback -- no OPENROUTER_API_KEY set] " + OFFLINE_ANSWER

    return RunnableLambda(_canned)


model_or_fallback = build_model()
LIVE_MODEL = bool(os.getenv("OPENROUTER_API_KEY"))


# ---------------------------------------------------------------------------
# 4. THE LCEL CHAIN  --  compose the three Runnables with the `|` pipe.
#    Read left to right: fill the prompt -> run the model -> parse to a string.
# ---------------------------------------------------------------------------
chain: Runnable = prompt | model_or_fallback | parser


# ---------------------------------------------------------------------------
# 5. A TOOL  --  a plain function the model can decide to call.
#    `@tool` turns it into a LangChain tool: name + description + typed args
#    schema are auto-derived, which is exactly what gets shown to the model.
# ---------------------------------------------------------------------------
@tool
def word_counter(text: str) -> int:
    """Count how many words are in the given text."""
    return len(text.split())


def demo_tool_binding() -> None:
    """Bind the tool to the model and do ONE invoke that should trigger it.

    Single-turn only -- we just check whether the model asked to call the tool
    (`response.tool_calls`). Actually running the tool + feeding the result back
    is the full agent loop, which lives in agent_and_middleware.py.
    """
    print(f"tool name        : {word_counter.name}")
    print(f"tool description : {word_counter.description}")
    print(f"tool args schema : {word_counter.args}")

    # [under the hood] bind_tools() does not send the Python function -- it sends
    # a JSON Schema so the model knows the tool's name, purpose, and typed args.
    # This full JSON blob is exactly what travels over the wire to the model.
    import json

    _schema = word_counter.args_schema.model_json_schema()
    print("[under the hood] JSON schema bind_tools() advertises to the model:")
    for _line in json.dumps(_schema, indent=2).splitlines():
        print(f"[under the hood]   {_line}")

    if not LIVE_MODEL:
        # No key -> we can't fire a live model to decide on the tool call. Show
        # the schema that WOULD be sent to the model instead of hard-failing.
        print(
            "\n[offline fallback -- no OPENROUTER_API_KEY set] "
            "Skipping the live tool-call. The schema above is exactly what "
            "bind_tools() advertises to the model; with a key set, the model "
            "would reply with a tool_calls entry for `word_counter`."
        )
        return

    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        model=MODEL_NAME,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    model_with_tools = model.bind_tools([word_counter])
    try:
        response = model_with_tools.invoke(
            "How many words are in this sentence: 'the quick brown fox jumps'? "
            "Use the word_counter tool."
        )
        if response.tool_calls:
            print(f"\ntool_calls fired : {response.tool_calls}")
        else:
            print(
                "\ntool_calls fired : (none -- the model answered directly: "
                f"{response.content!r})"
            )
    except Exception as exc:  # keep the demo resilient
        print(f"\n[live tool-call unavailable: {short_api_error(exc)}]")


# ---------------------------------------------------------------------------
# 6. ENTRY POINT  --  print each building block firing, ONE AT A TIME.
#    (Mirrors simple_graph.py, which prints each invoke() result at the bottom.)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mode = "LIVE ChatOpenAI" if LIVE_MODEL else "OFFLINE fallback (no key)"
    print("=" * 70)
    print(f"LangChain fundamentals  --  model step: {mode}")
    print("=" * 70)

    # (a) The prompt template, filled with a runtime variable.
    print("\n[1] PROMPT TEMPLATE  ->  prompt.invoke({'topic': ...})")
    filled = prompt.invoke({"topic": "what a Runnable is in LangChain"})
    print(filled)

    # [under the hood] prompt.invoke() does NOT return a string -- it returns a
    # PromptValue object (here a ChatPromptValue) that wraps a list of Messages.
    # That is the object the model step actually receives as its input.
    print(f"[under the hood] prompt.invoke(...) returns type : {type(filled).__name__}")
    _msgs = filled.to_messages()
    print(f"[under the hood]   .to_messages() -> list of {len(_msgs)} message(s)")
    for _m in _msgs:
        print(
            f"[under the hood]     {type(_m).__name__}(role={_m.type!r}) "
            f"content={_m.content!r}"
        )

    # (b) The output parser on its own, fed a fake message-like string.
    print("\n[2] OUTPUT PARSER  ->  parser.invoke('  hello world  ')")
    print(repr(parser.invoke("  hello world  ")))

    # (c) The full LCEL chain: prompt | model | parser, one invoke.
    print("\n[3] LCEL CHAIN  ->  (prompt | model | parser).invoke(...)")
    try:
        # [under the hood] Before the parser runs, the model step emits a rich
        # message object -- not a bare string. Build a tiny (prompt | model)
        # sub-chain (STOP before the parser) so we can grab that raw object and
        # see exactly what StrOutputParser is about to strip away.
        raw = (prompt | model_or_fallback).invoke(
            {"topic": "why LangChain components chain with `|`"}
        )
        print(f"[under the hood] raw model output type : {type(raw).__name__}")
        print(f"[under the hood]   short repr          : {raw!r}"[:200])
        if hasattr(raw, "content"):
            # A live ChatOpenAI returns an AIMessage: it carries a role plus
            # metadata (token usage, model name, finish reason...). The offline
            # fallback returns a plain str, which has none of that.
            print(f"[under the hood]   .content (what parser keeps) : {raw.content!r}")
            _meta = getattr(raw, "response_metadata", {}) or {}
            _usage = getattr(raw, "usage_metadata", None)
            print(f"[under the hood]   response_metadata keys       : {list(_meta)}")
            print(f"[under the hood]   usage_metadata (parser drops): {_usage}")
        else:
            print("[under the hood]   (offline fallback already a str -- nothing to strip)")

        result = chain.invoke({"topic": "why LangChain components chain with `|`"})
        print(result)
    except Exception as exc:
        # A key IS set but the live call failed (bad/expired key, no network...).
        # A teaching script must never crash -- report cleanly, show the fallback.
        print(f"[live model call failed: {short_api_error(exc)}]")
        print("[offline fallback so the lesson still runs] " + OFFLINE_ANSWER)

    # (d) Tool definition + bind_tools + one tool-triggering invoke.
    print("\n[4] TOOL + bind_tools  ->  did the model ask to call the tool?")
    demo_tool_binding()

    print("\n" + "=" * 70)
    print("Done. Same chain, swappable parts -- that's the whole point.")
    print("=" * 70)
