from pathlib import Path
import os
from typing import Dict

import faiss
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import QueryFusionRetriever

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "index"
DOC_DIR = "data/raw"
MODELS_DIR = Path("models")

# Set embedding model for chunking, indexing and retrieving from vector store
def _set_embeddings():
    import torch
    assert torch.cuda.is_available(), "CUDA not available; install CUDA PyTorch."
    print("[RAG] embeddings device:", device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Settings.embed_model = HuggingFaceEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device,
    )


def _set_llm():
    Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    # llama.cpp for fast inference
    ggufs = sorted(MODELS_DIR.glob("*.gguf"), key=lambda p: p.stat().st_size)
    if ggufs:
        from llama_index.llms.llama_cpp import LlamaCPP
        Settings.llm = LlamaCPP(
            model_path=str(ggufs[0]),
            temperature=0.2,
            max_new_tokens=256,
            context_window=4096,
            model_kwargs={
                "n_threads": os.cpu_count() or 8,
                "n_ctx": 4096,
                "n_gpu_layers": -1,
            },
            verbose=False,
        )
        return

    # OpenAI API key if available
    if os.getenv("OPENAI_API_KEY"):
        try:
            from llama_index.llms.openai import OpenAI
            Settings.llm = OpenAI(model="gpt-4o-mini")
            return
        except Exception as e:
            print(f"[rag] OpenAI fallback failed: {e}")

    raise RuntimeError(
        "No LLM configured. Put a .gguf file in app/models/ (ignored by git) "
        "or set OPENAI_API_KEY."
    )

# Read documents from data corpus, chunk them, embed, and store in vector store
def build_index(doc_dir: str) -> None:
    _set_embeddings()
    docs = SimpleDirectoryReader(doc_dir).load_data()
    
    # Output = 384 dim vectors
    dim = 384
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    storage_context.persist(persist_dir=str(INDEX_DIR))
    
def _maybe_bm25(docs, top_k=3):
    try:
        from llama_index.retrievers.bm25 import BM25Retriever as _BM25
    except Exception:
        return None
    for kwargs in (
        {"documents": docs, "similarity_top_k": top_k},
        {"docs": docs, "similarity_top_k": top_k},
        {"documents": docs},
        {"docs": docs},
    ):
        try:
            return _BM25.from_defaults(**kwargs)
        except TypeError:
            continue
        except Exception:
            return None
    return None


# Create embedding and initialize LLM and get the vector embeddings and fuse with BM25
def load_query_bundle(doc_dir: str):
    # _set_embeddings()
    _set_llm()
    # Load documents/corpus for BM25 exact matches
    docs = SimpleDirectoryReader(doc_dir).load_data()

    # Reload the FAISS vector store and storage context for query 
    vector_store = FaissVectorStore.from_persist_dir(str(INDEX_DIR))
    storage = StorageContext.from_defaults(
    persist_dir=str(INDEX_DIR),
    vector_store=vector_store
    )
    index = load_index_from_storage(storage)
    
    # Create retrievers for vector search (semantics) and BM25 (exact matches and rare terms)
    vector_retriever = index.as_retriever(similarity_top_k=3)
    bm25 = _maybe_bm25(docs, top_k=3)
    
    # Fuse the retrievers
    if bm25:
        retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25], similarity_top_k=3
        )
    else:
        retriever = vector_retriever

    # Create a response synthesizer object to give context aware answers based on retrieved nodes
    synthesizer = get_response_synthesizer(response_mode="compact")
    return retriever, synthesizer

# Generate a response to the query that is concise based on retrieved nodes and query similarity in vector store
def answer_query(q: str, retriever, synthesizer) -> Dict:
    # Use fused retriever to get relevant nodes to the query q
    nodes = retriever.retrieve(q)
    
    # Generate a response using the synthesizer that uses an LLM
    resp = synthesizer.synthesize(q, nodes)
    sources = []
    # For each relevant node, extract source and text snippet from source metadata
    for n in nodes:
        meta = n.metadata or {}
        sources.append({
            "source": meta.get("file_name") or meta.get("source") or "",
            "text": n.get_text()[:500]
        })
    return {"answer": getattr(resp, "response", str(resp)), "sources": sources}