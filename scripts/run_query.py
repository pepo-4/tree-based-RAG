import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from raptor.config import Config
from raptor.query_engine import RetrievalRerankGenerate

def main():
    parser = argparse.ArgumentParser(description="Query RAPTOR RAG Pipeline")
    parser.add_argument("--query", "-q", type=str, default="Esiste qualche bando per i musei?", help="Query question string")
    parser.add_argument("--faiss-path", type=str, default=Config.FAISS_INDEX_PATH, help="Path to saved FAISS index directory")
    parser.add_argument("--retrieval-k", type=int, default=30, help="Number of documents for vector retrieval")
    parser.add_argument("--rerank-k", type=int, default=10, help="Number of top documents for CrossEncoder reranking")
    parser.add_argument("--ollama-host", type=str, default=Config.OLLAMA_HOST, help="Ollama server host URL/IP")
    parser.add_argument("--ollama-model", type=str, default=Config.OLLAMA_MODEL, help="Ollama LLM model name")
    
    args = parser.parse_args()

    rag_system = RetrievalRerankGenerate(
        faiss_path=args.faiss_path,
        embedding_model=Config.EMBEDDING_MODEL,
        reranker_model=Config.RERANKER_MODEL,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
        temperature=Config.OLLAMA_TEMPERATURE
    )

    rag_system.process_query(
        query=args.query,
        retrieval_k=args.retrieval_k,
        rerank_k=args.rerank_k,
        verbose=True
    )

if __name__ == "__main__":
    main()
