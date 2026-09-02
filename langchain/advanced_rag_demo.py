"""
advanced_rag_demo.py  --  Advanced RAG retrieval strategies, side by side.

`rag_demo.py` shows plain "naive" RAG: embed the question, grab the top-k nearest
chunks, answer. That is the baseline. In the real world the *retrieval* step is
where RAG quality is won or lost -- so this file demonstrates SIX retrieval
strategies against the SAME FAISS index, so you can see how each one changes
WHICH chunks come back:

  1. BASELINE    -- naive top-k dense (vector) similarity.           (the "before")
  2. MMR         -- Maximal Marginal Relevance: diversify results,
                    drop near-duplicate chunks.
  3. RERANK      -- retrieve a wide net, then re-score with a
                    cross-encoder and keep the best few.             (the headline)
  4. HYBRID      -- dense (meaning) + BM25 (exact keywords), fused.
  5. MULTI-QUERY -- an LLM rewrites your question into several
                    variants, retrieves for each, unions the hits.
  6. COMPRESS    -- retrieve wide, then filter each chunk down to
                    only the parts actually relevant.

Everything is LangChain-native. The only step that needs an API key is
MULTI-QUERY (an LLM rewrites the query); the other five run fully local. Missing
pieces degrade gracefully (a labeled "skipped", never a crash).

BEFORE YOU RUN THIS, build the index once (no key needed):
    .venv/bin/python build_rag_index.py --doc docs/kestrel_home_faq.md \
        --index-dir faiss_index

Run interactively:        .venv/bin/python advanced_rag_demo.py
Run the headless demo:    .venv/bin/python advanced_rag_demo.py < /dev/null

Commands (interactive):
    <a question>          run ALL strategies for the question, one after another
    :baseline <q>         naive top-k dense only
    :mmr <q>              MMR (diversity) only
    :rerank <q>           cross-encoder reranking only
    :hybrid <q>           dense + BM25 hybrid only
    :multiquery <q>       LLM multi-query expansion only (needs a key)
    :compress <q>         contextual compression / filtering only
    :index <dir>          switch index dir (faiss_index / faiss_index_library)
    :help                 reprint this help
    :q / quit / exit      leave
"""

import logging
import os
import sys
import warnings

# Quiet HuggingFace's `\r` progress bar (it smears our print banners on some
# terminals) BEFORE importing anything that pulls in huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# `langchain-community` / `langchain-classic` print a "sunset" DeprecationWarning
# at import. In LangChain 1.x, langchain-classic is the CURRENT home of these
# higher-level retrievers/compressors; community still hosts FAISS/BM25. Suppress
# the noisy sunset notice; see requirements.txt for the full rationale.
warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain-classic.*", category=DeprecationWarning)

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Higher-level retrievers + compressors (LangChain 1.x -> langchain_classic).
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker,
    EmbeddingsFilter,
)

# Optional pieces -- guarded so the file still runs if they are missing.
try:
    from langchain_community.retrievers import BM25Retriever
    import rank_bm25  # noqa: F401  (BM25Retriever needs this at runtime)
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

try:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    _HAS_CROSS_ENCODER = True
except Exception:
    _HAS_CROSS_ENCODER = False

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# A small, standard reranking cross-encoder trained on MS MARCO (~80MB, local).
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_INDEX = "faiss_index"

# A question with BOTH a meaning side and an exact-keyword side ("promo code",
# "goodwill") -- good for showing where hybrid/rerank differ from naive dense.
DEMO_QUESTION = "What internal promo code can support agents give as a goodwill gesture?"


def short_api_error(exc: Exception) -> str:
    """Turn a noisy API exception into one short, classroom-friendly line."""
    msg = str(exc)
    if "401" in msg or "User not found" in msg or "authentication" in msg.lower():
        return "401 auth error -- your OPENROUTER_API_KEY looks invalid or expired"
    return msg.splitlines()[0][:160]


def make_model() -> ChatOpenAI | None:
    """A real ChatOpenAI Runnable if a key is set, else None (multi-query needs it)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def preview(doc, width: int = 150) -> str:
    """A one-line preview of a retrieved chunk."""
    text = " ".join(doc.page_content.split())
    return text[:width] + ("..." if len(text) > width else "")


def show_docs(docs, note: str = "") -> None:
    """Print retrieved chunks compactly, with a cross-encoder score if present."""
    if not docs:
        print("  (no chunks returned)")
        return
    for i, doc in enumerate(docs):
        score = doc.metadata.get("relevance_score")
        tag = f"  [rerank_score={score:.3f}]" if score is not None else ""
        print(f"  [{i}] {preview(doc)}{tag}")
    if note:
        print(f"  -> {note}")


# ---------------------------------------------------------------------------
# The demo state: loads the FAISS index + embeddings once, builds the shared
# pieces (raw docs for BM25, the cross-encoder), and exposes one method per
# strategy so they can be run individually or all together.
# ---------------------------------------------------------------------------
class AdvancedRag:
    def __init__(self, index_dir: str = DEFAULT_INDEX) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        self.model = make_model()
        self.index_dir = index_dir
        self.store = None
        self.docs = []
        self.cross_encoder = None
        self._load_index(index_dir)
        self._load_cross_encoder()

    # --- setup -------------------------------------------------------------
    def _load_index(self, index_dir: str) -> bool:
        if not os.path.isdir(index_dir):
            print(f"\n[!] No index at '{index_dir}/'. Build one first:")
            print(f"    .venv/bin/python build_rag_index.py --index-dir {index_dir}")
            return False
        self.store = FAISS.load_local(
            index_dir, self.embeddings, allow_dangerous_deserialization=True
        )
        self.index_dir = index_dir
        # BM25 needs the raw chunk texts. FAISS keeps them in its docstore, so we
        # pull the original chunks straight back out (no re-splitting needed).
        self.docs = list(self.store.docstore._dict.values())
        print(f"[loaded '{index_dir}/' -- {len(self.docs)} chunks]")
        return True

    def _load_cross_encoder(self) -> None:
        """Load the reranking cross-encoder once (first run downloads ~80MB)."""
        if not _HAS_CROSS_ENCODER:
            return
        try:
            self.cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
        except Exception as exc:  # offline / download failure -> rerank will skip
            print(f"[under the hood] cross-encoder unavailable ({exc}); rerank will skip")
            self.cross_encoder = None

    # --- strategies --------------------------------------------------------
    def baseline(self, q: str, k: int = 4):
        """1. Naive top-k dense retrieval -- the baseline everything else improves on."""
        print("\n--- 1. BASELINE (naive top-k dense similarity) ---")
        docs = self.store.as_retriever(search_kwargs={"k": k}).invoke(q)
        show_docs(docs, "the plain vector search: closest k chunks by meaning, nothing more")
        return docs

    def mmr(self, q: str, k: int = 4, fetch_k: int = 12):
        """2. MMR -- Maximal Marginal Relevance: relevant AND diverse (less overlap)."""
        print("\n--- 2. MMR (diversity -- avoid near-duplicate chunks) ---")
        docs = self.store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5},
        ).invoke(q)
        show_docs(docs, "picks k chunks that are each relevant but NOT redundant with each other")
        return docs

    def rerank(self, q: str, k: int = 4, fetch_k: int = 12):
        """3. RERANK -- retrieve a wide net (fetch_k), then a cross-encoder re-scores
        each (query, chunk) pair jointly and keeps the top k. Slower, far sharper."""
        print("\n--- 3. RERANK (cross-encoder re-scores a wide candidate set) ---")
        if self.cross_encoder is None:
            print("  [skipped] no cross-encoder available (needs the reranker model /"
                  " internet on first run).")
            return []
        base = self.store.as_retriever(search_kwargs={"k": fetch_k})
        reranker = CrossEncoderReranker(model=self.cross_encoder, top_n=k)
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base
        )
        docs = retriever.invoke(q)
        # The reranker returns the top_n but doesn't attach its scores; compute them
        # for display so learners SEE the cross-encoder's relevance judgement (a
        # cross-encoder reads query + chunk TOGETHER, so its score is much sharper
        # than the bi-encoder cosine distance that did the first-pass retrieval).
        try:
            scores = self.cross_encoder.score([(q, d.page_content) for d in docs])
            for d, s in zip(docs, scores):
                d.metadata["relevance_score"] = float(s)
        except Exception:
            pass
        show_docs(docs, f"grabbed {fetch_k} candidates by vector, then the cross-encoder "
                        f"re-ranked and kept the {k} best (higher rerank_score = more relevant)")
        return docs

    def hybrid(self, q: str, k: int = 4):
        """4. HYBRID -- dense (meaning) + BM25 (exact keywords), fused by the
        EnsembleRetriever. Dense blurs exact tokens (SKUs, codes); BM25 nails them."""
        print("\n--- 4. HYBRID (dense meaning + BM25 keywords, fused) ---")
        if not _HAS_BM25:
            print("  [skipped] BM25 needs `rank-bm25` (pip install rank-bm25).")
            return []
        bm25 = BM25Retriever.from_documents(self.docs)
        bm25.k = k
        vector = self.store.as_retriever(search_kwargs={"k": k})
        ensemble = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])
        docs = ensemble.invoke(q)
        show_docs(docs, "union of keyword hits (BM25) and meaning hits (vectors), "
                        "rank-fused -- built to catch exact tokens (codes, IDs) that "
                        "dense search can miss at scale (our tiny doc won't show a big win)")
        return docs

    def multiquery(self, q: str, k: int = 4):
        """5. MULTI-QUERY -- an LLM rewrites the question into several paraphrases,
        retrieves for each, and unions the results (beats one phrasing's blind spots)."""
        print("\n--- 5. MULTI-QUERY (LLM expands your question into variants) ---")
        if self.model is None:
            print("  [skipped] needs an OPENROUTER_API_KEY (an LLM writes the query variants).")
            return []
        base = self.store.as_retriever(search_kwargs={"k": k})
        retriever = MultiQueryRetriever.from_llm(retriever=base, llm=self.model)
        try:
            docs = retriever.invoke(q)
        except Exception as exc:
            print(f"  [live call failed: {short_api_error(exc)}] -- skipping multi-query.")
            return []
        show_docs(docs, "the LLM asked the same thing several ways; this is the deduped union")
        return docs

    def compress(self, q: str, k: int = 8, threshold: float = 0.3):
        """6. COMPRESS -- retrieve wide (k), then an EmbeddingsFilter drops any chunk
        below a relevance threshold, so only the strongly-related context survives."""
        print("\n--- 6. COMPRESS (retrieve wide, then filter out weak chunks) ---")
        base = self.store.as_retriever(search_kwargs={"k": k})
        filt = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=threshold)
        retriever = ContextualCompressionRetriever(base_compressor=filt, base_retriever=base)
        docs = retriever.invoke(q)
        show_docs(docs, f"pulled {k} chunks, then kept only those above similarity "
                        f"{threshold} -- trims noise before it reaches the model")
        return docs

    def run_all(self, q: str) -> None:
        print("\n" + "=" * 74)
        print(f"QUESTION: {q}")
        print("=" * 74)
        self.baseline(q)
        self.mmr(q)
        self.rerank(q)
        self.hybrid(q)
        self.multiquery(q)
        self.compress(q)
        print("\n" + "-" * 74)
        print("Same index, same question -- each strategy changed WHICH chunks came")
        print("back. That is the lever advanced RAG pulls: better context in = better")
        print("answer out, before the model ever sees a token.")

    def switch_index(self, new_dir: str) -> None:
        if self._load_index(new_dir):
            print(f"--- switched to index '{new_dir}/' ---")


# ---------------------------------------------------------------------------
# Headless auto-demo (runs when there is no interactive terminal).
# ---------------------------------------------------------------------------
def auto_demo(rag: AdvancedRag) -> None:
    print("=" * 74)
    print("ADVANCED RAG AUTO-DEMO  (no terminal detected -- scripted run)")
    print("=" * 74)
    key_state = "ON (multi-query enabled)" if rag.model else "OFF (multi-query will skip)"
    ce = "loaded" if rag.cross_encoder else "unavailable (rerank will skip)"
    bm = "yes" if _HAS_BM25 else "no (hybrid will skip)"
    print(f"Generation key: {key_state} | cross-encoder: {ce} | rank-bm25: {bm}")
    rag.run_all(DEMO_QUESTION)


# ---------------------------------------------------------------------------
# Interactive REPL.
# ---------------------------------------------------------------------------
HELP = """\
Commands:
  <a question>       run ALL six strategies for the question
  :baseline <q>      naive top-k dense only
  :mmr <q>           MMR (diversity) only
  :rerank <q>        cross-encoder reranking only
  :hybrid <q>        dense + BM25 hybrid only
  :multiquery <q>    LLM multi-query expansion only (needs a key)
  :compress <q>      contextual compression / filtering only
  :index <dir>       switch index dir (faiss_index / faiss_index_library)
  :help              reprint this help
  :q / quit / exit   leave"""

# Map a :command prefix to the method that handles it.
_SINGLE = {
    ":baseline": "baseline",
    ":mmr": "mmr",
    ":rerank": "rerank",
    ":hybrid": "hybrid",
    ":multiquery": "multiquery",
    ":compress": "compress",
}


def repl(rag: AdvancedRag) -> None:
    print("=" * 74)
    print("ADVANCED RAG DEMO  --  interactive")
    print("=" * 74)
    print(f"Current index: '{rag.index_dir}/'  |  cross-encoder: "
          f"{'ready' if rag.cross_encoder else 'off'}  |  multi-query: "
          f"{'ready' if rag.model else 'off (no key)'}")
    print("-" * 74)
    print(HELP)

    while True:
        try:
            line = input(f"\n[{rag.index_dir}] > ").strip()
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
        if low.startswith(":index"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                rag.switch_index(parts[1].strip())
            else:
                print("usage: :index <dir>")
            continue

        # A single-strategy command?
        handled = False
        for cmd, method in _SINGLE.items():
            if low.startswith(cmd):
                q = line[len(cmd):].strip()
                if q:
                    getattr(rag, method)(q)
                else:
                    print(f"usage: {cmd} <question>")
                handled = True
                break
        if handled:
            continue

        # Otherwise: a bare question -> run every strategy.
        rag.run_all(line)


# ---------------------------------------------------------------------------
# Entry point: interactive vs. headless based on stdin.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rag = AdvancedRag()
    if rag.store is None:
        sys.exit(1)  # _load_index already printed the "build the index first" hint
    if sys.stdin.isatty():
        repl(rag)
    else:
        auto_demo(rag)
