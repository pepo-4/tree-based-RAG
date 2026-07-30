import argparse
import sys
import os
import glob
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from raptor.config import Config
from raptor.chunking import TextChunker
from raptor.tree_builder import RaptorTreeBuilder
from raptor.vectorstore import VectorStoreManager
from raptor.summarization import GeminiClusterSummarizer

def load_custom_texts(input_path: str) -> List[Dict[str, str]]:
    """Loads custom text content from a file or folder of files (.txt, .md, .csv, .json)."""
    texts = []
    
    if os.path.isfile(input_path):
        filename = os.path.basename(input_path)
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            texts.append({"title": filename, "text": content})
    elif os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "**", "*.*"), recursive=True)
        for filepath in files:
            if filepath.endswith((".txt", ".md", ".json", ".csv")):
                filename = os.path.basename(filepath)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if content.strip():
                            texts.append({"title": filename, "text": content})
                except Exception as e:
                    print(f" Error reading file '{filepath}': {e}")
    else:
        raise FileNotFoundError(f"Input path '{input_path}' not found!")
        
    return texts

def main():
    parser = argparse.ArgumentParser(description="Run RAPTOR on Custom Text Files & Models")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input text file or directory")
    parser.add_argument("--output-index", "-o", type=str, default="custom_faiss_index", help="Output directory for generated FAISS vectorstore")
    parser.add_argument("--embedding-model", type=str, default=Config.EMBEDDING_MODEL, help="HuggingFace / SentenceTransformer embedding model name")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash", help="Google Gemini model name for cluster summarization")
    parser.add_argument("--max-levels", type=int, default=3, help="Max RAPTOR tree hierarchy depth")
    parser.add_argument("--chunk-size", type=int, default=250, help="Text splitter token chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Text splitter token overlap")
    
    args = parser.parse_args()

    print(f" Loading custom text data from '{args.input}'...")
    raw_documents = load_custom_texts(args.input)
    print(f" Loaded {len(raw_documents)} raw document source files.")

    print(f" Chunking text (Model: '{args.embedding_model}', Chunk Size: {args.chunk-size})...")
    chunker = TextChunker(
        model_id=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    docs = chunker.chunk_texts(raw_documents, text_key="text", title_key="title")
    print(f" Generated {len(docs)} initial Level 0 chunks.")

    print(f" Initializing Gemini Summarizer ({args.gemini-model})...")
    summarizer = GeminiClusterSummarizer(model_name=args.gemini_model)

    print(f" Building Multi-Level RAPTOR Tree (Max Levels: {args.max-levels})...")
    tree_builder = RaptorTreeBuilder(
        embedding_model_name=args.embedding_model,
        summarizer=summarizer
    )
    dataset_grafo = tree_builder.build_tree_for_documents(
        documents=docs,
        max_levels=args.max_levels
    )

    print(f"\n Building and Saving FAISS index to '{args.output_index}'...")
    vector_manager = VectorStoreManager(
        embedding_model_name=args.embedding_model,
        faiss_path=args.output-index
    )
    vector_manager.create_and_save_index(dataset_grafo=dataset_grafo)

    print(f"\n Custom RAPTOR Index built successfully at '{args.output_index}'!")
    print(f" You can now run queries against it using:")
    print(f"   python scripts/run_query.py --faiss-path {args.output-index} --query \"Your question here\"")

if __name__ == "__main__":
    main()
