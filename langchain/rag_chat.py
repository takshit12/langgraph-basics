"""
rag_chat.py  --  a CHAINLIT chat-style UI for the RAG demo (secondary option).

Same RAG chain as `rag_demo.py` / `rag_app.py`, but in a chat window: you type a
question, it streams back the grounded answer and sends the retrieved chunks
(with similarity scores) as a collapsible step, so the room still SEES retrieval.

    RAG = {"context": retriever | format_docs, "question": passthrough}
              | prompt | model | parser  ->  grounded answer

This file is deliberately STANDALONE (no import from rag_demo.py) and never
hard-fails: with no OPENROUTER_API_KEY it still retrieves and shows the chunks,
then sends a clearly-labeled extractive fallback (the top chunk) instead of a
generated answer. Build the FAISS index first (no key needed):
    .venv/bin/python build_rag_index.py --doc docs/kestrel_home_faq.md \
        --index-dir faiss_index

Run it:
    chainlit run rag_chat.py -w

(Chainlit is an OPTIONAL extra -- it's included in requirements.txt. If it isn't
installed this file is still correct and importable; just `uv pip install chainlit`.)
"""

import logging
import os
import warnings

# Quiet HuggingFace's noisy progress bar / HF-Hub notice BEFORE importing HF libs
# -- same guard the other scripts use.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# `langchain-community` (FAISS) prints a "sunset" DeprecationWarning at import.
# It remains the only working FAISS path in LangChain 1.x. Suppress it.
# ---------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore", message=".*langchain-community.*", category=DeprecationWarning
)

import chainlit as cl
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Pick up OPENROUTER_API_KEY / OPENROUTER_MODEL from the project .env (optional).
load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = os.getenv("RAG_INDEX_DIR", "faiss_index")  # override to swap the doc.
TOP_K = 3


# ---------------------------------------------------------------------------
# The RAG prompt -- same shape as rag_demo.py: answer ONLY from the context.
# ---------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a support assistant. Answer the question using ONLY the context "
    "below. If the context does not contain the answer, say you don't have that "
    "information -- do not guess.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def short_api_error(exc: Exception) -> str:
    """Turn a noisy API exception into one short, classroom-friendly line."""
    msg = str(exc)
    if "401" in msg or "User not found" in msg or "authentication" in msg.lower():
        return "401 auth error -- your OPENROUTER_API_KEY looks invalid or expired"
    return msg.splitlines()[0][:160]


def format_docs(docs) -> str:
    """Flatten a list of retrieved Documents into one context string."""
    return "\n\n".join(d.page_content for d in docs)


def make_model() -> ChatOpenAI | None:
    """A genuine LangChain Runnable wrapping OpenRouter, or None with no key."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# on_chat_start: load the (heavy) embedding model + FAISS index once per session
# and stash them in the user session, so each message reuses them.
# ---------------------------------------------------------------------------
@cl.on_chat_start
async def on_chat_start() -> None:
    if not os.path.isdir(INDEX_DIR):
        await cl.Message(
            content=(
                f"No FAISS index at `{INDEX_DIR}/`. Build it first (no key "
                f"needed):\n\n"
                f"```\n.venv/bin/python build_rag_index.py "
                f"--index-dir {INDEX_DIR}\n```"
            )
        ).send()
        return

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    # allow_dangerous_deserialization: FAISS indexes are pickled; we trust our own.
    store = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    model = make_model()

    cl.user_session.set("store", store)
    cl.user_session.set("model", model)

    key_state = (
        "Live generation is ON."
        if model is not None
        else "No `OPENROUTER_API_KEY` -- I'll retrieve and show chunks, then give "
        "an extractive top-chunk fallback instead of a generated answer."
    )
    await cl.Message(
        content=(
            f"**RAG chat** ready over `{INDEX_DIR}/` "
            f"({store.index.ntotal} vectors, 384-dim `all-MiniLM-L6-v2`). "
            f"{key_state}\n\nAsk me a question about the document."
        )
    ).send()


# ---------------------------------------------------------------------------
# on_message: retrieve -> show chunks -> generate the grounded answer.
# ---------------------------------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message) -> None:
    store = cl.user_session.get("store")
    model = cl.user_session.get("model")
    if store is None:
        await cl.Message(
            content="No index loaded -- build one, then restart the chat."
        ).send()
        return

    question = message.content.strip()

    # 1. RETRIEVE (always local). similarity_search_with_score gives us the score
    # per chunk; lower L2 distance = closer. Show them in a collapsible Step.
    scored = store.similarity_search_with_score(question, k=TOP_K)
    async with cl.Step(name=f"Retrieved {len(scored)} chunks (local, no key)") as step:
        step.output = "\n\n".join(
            f"**chunk[{i}]  ·  score {score:.4f}**\n\n{doc.page_content.strip()}"
            for i, (doc, score) in enumerate(scored)
        )

    # 2. GENERATE the grounded answer with the exact LCEL RAG chain. Never hard-
    # fail: with no key (or a failed call) fall back to the extractive top chunk.
    if model is None:
        top = scored[0][0].page_content.strip().replace("\n", " ")
        await cl.Message(
            content=(
                "_(extractive fallback -- no key set; the top retrieved chunk:)_\n\n"
                f"> {top[:200]}..."
            )
        ).send()
        return

    retriever = store.as_retriever(search_kwargs={"k": TOP_K})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | model
        | StrOutputParser()
    )

    answer = cl.Message(content="")
    try:
        # Stream the grounded answer token-by-token as the chain produces it.
        async for token in chain.astream(question):
            await answer.stream_token(token)
        await answer.send()
    except Exception as exc:  # never crash the chat.
        top = scored[0][0].page_content.strip().replace("\n", " ")
        await cl.Message(
            content=(
                f"_(live generation failed: {short_api_error(exc)} -- extractive "
                f"fallback, top chunk:)_\n\n> {top[:200]}..."
            )
        ).send()
