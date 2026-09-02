"""
rag_demo.py  --  THE live RAG demo.

The whole point of this file is one contrast, shown live:

    1. Ask a PLAIN LLM about a private document it was never trained on
       -> it hallucinates or hedges (confidently wrong / generically vague).
    2. Ask the SAME question through a full RAG pipeline
       -> it answers correctly, grounded in the retrieved chunks.
    3. Print the retrieved chunks, so the room sees WHY the answer changed.

RAG is not a new paradigm -- it is the same LangChain `Runnable` pieces from
Section 1, piped together with `|`, with a retriever spliced in:

    {"context": retriever | format_docs, "question": passthrough}
        | prompt | model | StrOutputParser()

WHAT NEEDS A KEY, WHAT DOESN'T
  - Embedding + FAISS retrieval ALWAYS work locally, with NO API key.
  - Only the final GENERATION step (the chat model) needs OPENROUTER_API_KEY.
  - With no key we NEVER hard-fail: plain mode prints a clearly-labeled canned
    "confidently wrong" answer; RAG mode prints the retrieved chunks plus a
    labeled extractive fallback (the top chunk). For the live session, use a
    real key -- the real hallucination-vs-grounded contrast is what lands.

BEFORE YOU RUN THIS, build the index once (no key needed):
    .venv/bin/python build_rag_index.py --doc docs/kestrel_home_faq.md \
        --index-dir faiss_index

Run interactively:        .venv/bin/python rag_demo.py
Run the headless demo:    .venv/bin/python rag_demo.py < /dev/null

Commands (interactive):
    <a question>       run PLAIN first, then RAG (the Section 5 run-of-show)
    :plain <question>  run ONLY the plain-LLM answer (the "before")
    :rag <question>    run ONLY the RAG answer (the "after")
    :index <dir>       switch index dir (e.g. faiss_index / faiss_index_library)
    :chunks            reprint the chunks retrieved by the last RAG run
    :verbose           toggle printing the FULL augmented prompt sent to the model
    :help              reprint this help
    :q / quit / exit   leave
"""

import logging
import os
import sys
import warnings

# Quiet HuggingFace's noisy `\r`-based "Loading weights" progress bar and the
# "unauthenticated requests to the HF Hub" notice BEFORE importing HF libs --
# on some terminals the progress bar's carriage returns smear over our print
# banners. Must be set before langchain_huggingface (which imports huggingface_hub).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# `langchain-community` (which provides FAISS) prints a "sunset"
# DeprecationWarning at import. It remains the only working FAISS/TextLoader path
# in LangChain 1.x (langchain-classic just re-exports it; langchain-faiss 0.1.1
# is an unmaintained stub). Suppress the noisy sunset warning; see requirements.txt.
# ---------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore", message=".*langchain-community.*", category=DeprecationWarning
)

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
DEFAULT_INDEX = "faiss_index"

# A good demo question: the answer (a 12% restocking fee) lives ONLY in the
# Kestrel Home doc, so a plain LLM cannot possibly know it, but RAG can.
DEMO_QUESTION = (
    "What is Kestrel Home's restocking fee on returns outside the 45-day window?"
)


# ---------------------------------------------------------------------------
# The RAG prompt. It tells the model to answer ONLY from the retrieved context
# and to admit when the context doesn't contain the answer -- this is what keeps
# a grounded RAG answer from drifting back into guessing.
# ---------------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_template(
    "You are a support assistant for Kestrel Home. Answer the question using "
    "ONLY the context below. If the context does not contain the answer, say "
    "you don't have that information -- do not guess.\n\n"
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


# ---------------------------------------------------------------------------
# Model factory. Returns a real ChatOpenAI Runnable if a key is set, else None.
# (When None, the caller uses a clearly-labeled offline fallback instead.)
# ---------------------------------------------------------------------------
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


def format_docs(docs) -> str:
    """Flatten a list of retrieved Documents into one context string."""
    return "\n\n".join(d.page_content for d in docs)


# ---------------------------------------------------------------------------
# Retrieval: load the FAISS index (always offline) and make a retriever.
# ---------------------------------------------------------------------------
def load_retriever(index_dir: str, embeddings: HuggingFaceEmbeddings):
    """Load a persisted FAISS index and return a top-3 retriever, or None if the
    index directory does not exist yet (with a friendly hint)."""
    if not os.path.isdir(index_dir):
        print(f"\n[!] No index found at '{index_dir}/'.")
        print("    Build one first (no API key needed), e.g.:")
        print(f"    .venv/bin/python build_rag_index.py --index-dir {index_dir}")
        return None
    # allow_dangerous_deserialization: FAISS indexes are pickled; we trust our own.
    store = FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )
    return store.as_retriever(search_kwargs={"k": 3})


# ---------------------------------------------------------------------------
# The demo state: one object holds the model, embeddings, current index dir,
# retriever, and the last-retrieved chunks (so :chunks can reprint them).
# ---------------------------------------------------------------------------
class RagDemo:
    def __init__(self) -> None:
        self.model = make_model()
        # Embeddings load once and are reused across index switches.
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.index_dir = DEFAULT_INDEX
        self.retriever = load_retriever(self.index_dir, self.embeddings)
        self.last_chunks: list = []
        # [under the hood] parallel list of L2 similarity scores for last_chunks,
        # so :chunks can reprint the same score annotations after the fact.
        self.last_scores: list = []
        # [under the hood] when True, run_rag prints the FULL filled prompt that
        # augmentation builds (toggle at runtime with the :verbose command).
        self.verbose: bool = False

    def _offline_plain(self, reason: str) -> None:
        """A clearly-LABELED, illustrative 'confidently wrong' plain answer, so
        the hallucination point still lands with no key OR when a live call fails."""
        print(f"[{reason}]")
        print("[ILLUSTRATIVE 'confidently wrong' answer a plain LLM might give:]")
        print(
            "  \"Kestrel Home charges a standard 25% restocking fee on all "
            "returns after 30 days.\"\n"
            "  ^ Made up. A plain LLM has never seen Kestrel Home's internal "
            "doc, so it guesses -- and it's wrong (the real fee is 12%)."
        )

    # --- mode 1: PLAIN LLM (no retrieval) -- the "before" ------------------
    def run_plain(self, question: str) -> None:
        print("\n" + "=" * 72)
        print("PLAIN LLM  (no document access -- this is the 'before')")
        print("=" * 72)
        if self.model is None:
            self._offline_plain("offline fallback -- no OPENROUTER_API_KEY set")
            return
        # With a key: a real LLM answer -- watch it hedge or invent specifics.
        # If the call fails (bad/expired key, no network), never crash the REPL --
        # report cleanly and drop to the same illustrative fallback.
        try:
            answer = (self.model | StrOutputParser()).invoke(question)
            print(answer.strip())
        except Exception as exc:
            print(f"[live model call failed: {short_api_error(exc)}]")
            self._offline_plain("showing an illustrative answer instead")

    # --- mode 2: FULL RAG (retrieve -> augment -> generate) -- the "after" -
    def run_rag(self, question: str) -> None:
        print("\n" + "=" * 72)
        print(f"RAG  (grounded in '{self.index_dir}/' -- this is the 'after')")
        print("=" * 72)
        if self.retriever is None:
            print("[no index loaded -- build one, or switch with :index <dir>]")
            return

        # [under the hood] RETRIEVAL step 1 -- embed the QUESTION into the SAME
        # vector space as the chunks. Nearest-neighbour search only works because
        # question and chunks share this geometry. Show the vector's dimensionality.
        q_vector = self.embeddings.embed_query(question)
        print(
            f"\n[under the hood] question embedded -> {len(q_vector)}-dim vector "
            f"(model {EMBED_MODEL.split('/')[-1]})"
        )

        # Retrieve first so we can both show the chunks AND feed the chain.
        # NOTE: retrieval is fully local and always works -- so the chunks (the
        # load-bearing part of the demo) are shown even if generation later fails.
        # [under the hood] similarity_search_with_score does the retrieval ONCE and
        # hands back (doc, score) pairs; we reuse these SAME docs for the chain below
        # instead of re-retrieving. score is L2 distance: LOWER = closer/more relevant.
        k = self.retriever.search_kwargs.get("k", 3)
        scored = self.retriever.vectorstore.similarity_search_with_score(question, k=k)
        self.last_chunks = [doc for doc, _ in scored]
        self.last_scores = [score for _, score in scored]
        self._print_chunks()

        if self.model is None:
            self._extractive_fallback("offline fallback -- no OPENROUTER_API_KEY set")
            return

        # [under the hood] AUGMENTATION (the 'A' in RAG): flatten the retrieved
        # chunks into ONE context string and splice it into the prompt. We build it
        # here, once, and feed the same text to the chain (no second retrieval).
        context_text = format_docs(self.last_chunks)
        print(
            f"\n[under the hood] augmentation: injecting {len(context_text)} chars "
            f"of retrieved context into the prompt, then generating with model "
            f"'{self.model.model_name}'"
        )
        if self.verbose:
            print("[under the hood] FULL augmented prompt actually sent to the model:")
            print("-" * 72)
            print(RAG_PROMPT.format(context=context_text, question=question))
            print("-" * 72)
        else:
            print("[under the hood] (run :verbose to see the FULL augmented prompt)")

        # With a key: the exact LCEL RAG chain -- context piped into the prompt,
        # the model, and an output parser. Same `|` pattern (and same dict-mapping
        # chain shape) as Section 1. We feed the ALREADY-retrieved context via a
        # passthrough lambda so retrieval is not repeated inside the chain.
        chain = (
            {"context": lambda _: context_text, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self.model
            | StrOutputParser()
        )
        print("\n[grounded answer]")
        # If generation fails (bad/expired key, no network), never crash -- the
        # retrieved chunks above already made the point; fall back to extractive.
        try:
            print(chain.invoke(question).strip())
        except Exception as exc:
            print(f"[live generation failed: {short_api_error(exc)}]")
            self._extractive_fallback("showing an extractive answer instead")

    def _extractive_fallback(self, reason: str) -> None:
        """Labeled extractive answer -- the top retrieved chunk, verbatim-ish."""
        print(f"\n[{reason}]")
        print("[EXTRACTIVE answer -- first ~200 chars of the top chunk:]")
        top = self.last_chunks[0].page_content.strip().replace("\n", " ")
        print(f"  {top[:200]}...")
        print(
            "  ^ With a key, the chat model would read ALL the chunks above "
            "and answer in one clean sentence grounded in them."
        )

    def _print_chunks(self) -> None:
        """Show the chunks retrieval pulled in -- the 'why the answer changed'."""
        print(f"\n[retrieved {len(self.last_chunks)} chunks from '{self.index_dir}/']")
        for i, doc in enumerate(self.last_chunks):
            # Chunks cap at ~500 chars (see build_rag_index.py), so print them in
            # full -- truncating here would hide the very fact the demo is proving
            # lives in the retrieved context (e.g. the 12% restocking fee).
            text = doc.page_content.strip().replace("\n", " ")
            # [under the hood] annotate each chunk with its L2 similarity score
            # (lower = closer); this is exactly what ranked the retrieval order.
            score_note = ""
            if i < len(self.last_scores):
                score_note = f"  [score={self.last_scores[i]:.2f}]"
            print(f"  chunk[{i}]: {text}{score_note}")

    def reprint_chunks(self) -> None:
        if not self.last_chunks:
            print("(no chunks yet -- run a question or `:rag <q>` first)")
            return
        self._print_chunks()

    def switch_index(self, new_dir: str) -> None:
        retriever = load_retriever(new_dir, self.embeddings)
        if retriever is None:
            return  # load_retriever already printed a friendly hint
        self.index_dir = new_dir
        self.retriever = retriever
        self.last_chunks = []
        self.last_scores = []
        print(f"--- switched to index '{new_dir}/' ---")


# ---------------------------------------------------------------------------
# Headless auto-demo: runs when there is no interactive terminal, so
# `python rag_demo.py < /dev/null` demonstrates the whole thing without hanging.
# ---------------------------------------------------------------------------
def auto_demo(demo: RagDemo) -> None:
    print("=" * 72)
    print("RAG AUTO-DEMO (no terminal detected -- running the scripted demo)")
    print("=" * 72)
    key_state = "REAL LLM answers" if demo.model else "offline labeled fallbacks"
    print(f"Generation: {key_state}. Retrieval is always local (no key needed).")
    print(f"\nQuestion: {DEMO_QUESTION}")

    # The exact Section 5 run-of-show: plain first, then RAG (with chunks).
    demo.run_plain(DEMO_QUESTION)
    demo.run_rag(DEMO_QUESTION)

    print("\n" + "=" * 72)
    print("Takeaway: same question, same model -- only RAG saw the document, so")
    print("only RAG got the 12% restocking fee right. That's retrieval doing its job.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Interactive REPL.
# ---------------------------------------------------------------------------
HELP = """\
Commands:
  <a question>       run PLAIN first, then RAG (plain=before, RAG=after)
  :plain <question>  run ONLY the plain-LLM answer
  :rag <question>    run ONLY the RAG answer
  :index <dir>       switch index dir (e.g. faiss_index / faiss_index_library)
  :chunks            reprint the chunks retrieved by the last RAG run
  :verbose           toggle printing the FULL augmented prompt sent to the model
  :help              reprint this help
  :q / quit / exit   leave"""


def repl(demo: RagDemo) -> None:
    print("=" * 72)
    print("RAG DEMO  --  interactive")
    print("=" * 72)
    key_state = "REAL LLM answers" if demo.model else "offline labeled fallbacks (no key)"
    print(f"Generation: {key_state}. Retrieval always works locally.")
    print(f"Current index: '{demo.index_dir}/'")
    print("-" * 72)
    print(HELP)

    while True:
        try:
            line = input(f"\n[{demo.index_dir}] > ").strip()
        except EOFError:
            break
        if not line:
            continue

        low = line.lower()
        if low in (":q", ":quit", ":exit", "quit", "exit"):
            print("bye!")
            break
        if low in (":help", "help", "?"):
            print(HELP)
            continue
        if low == ":chunks":
            demo.reprint_chunks()
            continue
        if low == ":verbose":
            # [under the hood] flip whether run_rag prints the FULL augmented prompt.
            demo.verbose = not demo.verbose
            state = "ON" if demo.verbose else "OFF"
            print(f"[under the hood] full augmented-prompt display: {state}")
            continue
        if low.startswith(":index"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                demo.switch_index(parts[1].strip())
            else:
                print("usage: :index <dir>   (e.g. :index faiss_index_library)")
            continue
        if low.startswith(":plain"):
            q = line[len(":plain"):].strip()
            if q:
                demo.run_plain(q)
            else:
                print("usage: :plain <question>")
            continue
        if low.startswith(":rag"):
            q = line[len(":rag"):].strip()
            if q:
                demo.run_rag(q)
            else:
                print("usage: :rag <question>")
            continue

        # A bare question: the full Section 5 run-of-show -- plain then RAG.
        demo.run_plain(line)
        demo.run_rag(line)


# ---------------------------------------------------------------------------
# Entry point: interactive vs. headless based on stdin.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = RagDemo()
    if sys.stdin.isatty():
        repl(demo)
    else:
        auto_demo(demo)
