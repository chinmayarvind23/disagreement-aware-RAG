from pathlib import Path
import os
from typing import Dict
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext,
    get_response_synthesizer
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "index"
DOC_DIR = "data/raw"
MODELS_DIR = Path("models")

def _set_llm():
    # Set embedding model for chunking and retrieving from vector store
    Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    # llama.cpp for fast inference
    ggufs = list(MODELS_DIR.glob("*.gguf"))
    if ggufs:
        try:
            from llama_index.llms.llama_cpp import LlamaCPP
            Settings.llm = LlamaCPP(
                model_path=str(ggufs[0]),
                temperature=0.2, max_new_tokens=256, context_window=2048, n_threads=8, verbose=False,
            )
            return
        except Exception as e:
            print(f"[rag] llama.cpp not available ({e}), trying OpenAI…")

    # OpenAI API key if available
    if os.getenv("OPENAI_API_KEY"):
        try:
            from llama_index.llms.openai import OpenAI
            Settings.llm = OpenAI(model="gpt-4o-mini")
            return
        except Exception as e:
            print(f"[rag] OpenAI fallback failed: {e}")

    raise RuntimeError(
        "No LLM configured. Put a GGUF in /models (llama.cpp) or set OPENAI_API_KEY."
    )

# Read documents from data corpus, chunk them, embed, and store in vector store
def build_index(doc_dir: str) -> None:
    _set_llm()
    docs = SimpleDirectoryReader(doc_dir).load_data()
    vs = FaissVectorStore.from_defaults()
    index = VectorStoreIndex.from_documents(docs, vector_store=vs)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))

# Create embedding and initialize LLM and get the vector embeddings and fuse with BM25
def load_query_bundle(doc_dir: str):
    _set_llm()
    # Load documents/corpus for BM25 exact matches
    docs = SimpleDirectoryReader(doc_dir).load_data()

    # Reload the FAISS vector store and storage context for query 
    storage = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    index = VectorStoreIndex.from_vector_store(
        FaissVectorStore.from_persist_dir(str(INDEX_DIR)),
        storage_context=storage
    )

    # Create retrievers for vector search (semantics) and BM25 (exact matches and rare terms)
    vector_retriever = index.as_retriever(similarity_top_k=6)
    bm25 = BM25Retriever.from_defaults(documents=docs, similarity_top_k=6)
    
    # Fuse the retrievers
    retriever = QueryFusionRetriever(retrievers=[vector_retriever, bm25], similarity_top_k=6)

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
    return {"answer": str(resp), "sources": sources}