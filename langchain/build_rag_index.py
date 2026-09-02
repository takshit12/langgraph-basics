"""
build_rag_index.py
==================

Build a local FAISS vector index from a plain-text document, so `rag_demo.py`
can retrieve from it later. This is the FIRST half of a RAG pipeline:

    Document  ->  Split into chunks  ->  Embed into vectors  ->  Store in FAISS

NO API KEY IS NEEDED. Everything here runs fully on your machine:
  - the embedding model (`all-MiniLM-L6-v2`) runs locally via sentence-transformers
  - FAISS is a local file on disk, not a hosted service

Run it (defaults shown):
    .venv/bin/python build_rag_index.py --doc docs/kestrel_home_faq.md \
        --index-dir faiss_index

Build the "swap the document" index for the secondary demo doc:
    .venv/bin/python build_rag_index.py --doc docs/meridian_library_policy.md \
        --index-dir faiss_index_library

NOTE: the FIRST time you run this, sentence-transformers downloads the ~80MB
embedding model into ~/.cache/huggingface (outside this repo). That needs
internet and takes a minute or two. Every run after that is offline + fast.
Pre-build your indexes the night before a live session, not in front of the room.
"""

import argparse
import logging
import os
import time
import warnings

# Quiet HuggingFace's noisy `\r`-based "Loading weights" progress bar and the
# "unauthenticated requests to the HF Hub" notice BEFORE importing HF libs --
# on some terminals the progress bar's carriage returns smear over our print
# banners. Must be set before langchain_huggingface (which imports huggingface_hub).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# `langchain-community` (which provides FAISS + TextLoader) prints a "sunset"
# DeprecationWarning at import. It remains the only working FAISS/TextLoader path
# in LangChain 1.x (langchain-classic just re-exports it; langchain-faiss 0.1.1
# is an unmaintained stub). Suppress the noisy sunset warning; see requirements.txt.
# ---------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore", message=".*langchain-community.*", category=DeprecationWarning
)

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# The local, no-key embedding model. 384-dimensional vectors, small + fast, the
# standard "hello world" sentence-transformers model. Shared with rag_demo.py so
# the index we build here matches what the demo loads back.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# The pipeline, one function so the four stages read top to bottom.
# ---------------------------------------------------------------------------
def build_index(doc_path: str, index_dir: str) -> None:
    """Load a text doc, split it, embed the chunks, and persist a FAISS index."""

    # 1. LOADING -- pull the raw document into a LangChain Document object.
    print(f"[1/4] Loading document: {doc_path}")
    t_load = time.perf_counter()
    loader = TextLoader(doc_path)
    documents = loader.load()
    total_chars = sum(len(d.page_content) for d in documents)
    print(f"      loaded {len(documents)} document(s), {total_chars} characters total")
    print(f"[under the hood] loading took {time.perf_counter() - t_load:.3f}s")

    # 2. SPLITTING -- cut the doc into ~500-char chunks with 50-char overlap.
    # Smaller chunks retrieve more precisely; the overlap keeps a sentence that
    # straddles a boundary from being cut in half and losing its meaning.
    print("[2/4] Splitting into chunks (chunk_size=500, chunk_overlap=50)")
    t_split = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"      produced {len(chunks)} chunks")
    print(f"[under the hood] splitting took {time.perf_counter() - t_split:.3f}s")

    # Simple chunk-size stats so learners see how uniform the split is: with a
    # 500-char target, most chunks land near it, but the tail chunk is shorter.
    chunk_lengths = [len(c.page_content) for c in chunks]
    print(
        "[under the hood] chunk length (chars): "
        f"min={min(chunk_lengths)}  max={max(chunk_lengths)}  "
        f"avg={sum(chunk_lengths) / len(chunk_lengths):.1f}"
    )

    # Show a couple of real chunks so a learner SEES splitting happen concretely.
    print("\n      --- sample chunks (so you can see what 'splitting' produced) ---")
    for i in (0, len(chunks) // 2):
        preview = chunks[i].page_content.strip().replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + " ..."
        print(f"      chunk[{i}] ({len(chunks[i].page_content)} chars): {preview}")
    print("      ----------------------------------------------------------------\n")

    # 3. EMBEDDING -- turn each chunk of text into a vector of numbers. Similar
    # meanings land at similar vectors. Runs locally, no API key.
    print(f"[3/4] Embedding chunks with {EMBED_MODEL}")
    print("      (first run downloads the ~80MB model into ~/.cache/huggingface)")
    t_embed = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Prove that "text becomes numbers": one probe string -> a fixed-width vector.
    # Every chunk, no matter its length, collapses to this many floats.
    probe_vector = embeddings.embed_query("sample")
    dim = len(probe_vector)
    print(f"[under the hood] embedding vector dimension: {dim} floats per chunk")

    # Now embed one REAL chunk and show its first components -- this is literally
    # what a chunk of your document looks like once it is turned into numbers.
    real_vector = embeddings.embed_query(chunks[0].page_content)
    first_components = ", ".join(f"{x:+.4f}" for x in real_vector[:8])
    print(f"[under the hood] chunk[0] embeds to a {len(real_vector)}-dim vector")
    print(f"[under the hood] chunk[0] first 8 components: [{first_components}, ...]")
    print(f"[under the hood] embedding-model load + probe took "
          f"{time.perf_counter() - t_embed:.3f}s (this is the slow part)")

    # 4. STORING -- build a FAISS index over those vectors and save it to disk.
    print(f"[4/4] Building FAISS index and saving to: {index_dir}")
    t_store = time.perf_counter()
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(index_dir)

    # The FAISS index holds one vector per chunk. Confirm nothing was dropped:
    # ntotal (vectors stored) should exactly match the chunk count.
    n_vectors = store.index.ntotal
    match = "OK" if n_vectors == len(chunks) else "MISMATCH!"
    print(f"[under the hood] FAISS vectors stored (index.ntotal): {n_vectors}")
    print(f"[under the hood] vectors == chunks? {n_vectors} == {len(chunks)} -> {match}")
    print(f"[under the hood] embedding-all-chunks + FAISS build + save took "
          f"{time.perf_counter() - t_store:.3f}s")

    print(f"\nDone. Indexed {len(chunks)} chunks into '{index_dir}/'.")
    print("You can now retrieve from it with: .venv/bin/python rag_demo.py")


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local FAISS vector index from a text document.",
    )
    parser.add_argument(
        "--doc",
        default="docs/kestrel_home_faq.md",
        help="Path to the source text document (default: docs/kestrel_home_faq.md).",
    )
    parser.add_argument(
        "--index-dir",
        default="faiss_index",
        help="Directory to write the FAISS index into (default: faiss_index).",
    )
    args = parser.parse_args()
    build_index(args.doc, args.index_dir)


if __name__ == "__main__":
    main()
