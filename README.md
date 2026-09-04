# RAPTOR – Hierarchical RAG

A Python implementation of **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval), a hierarchical take on Retrieval-Augmented Generation.

Instead of searching over a flat list of text chunks, RAPTOR builds a tree over the corpus: it groups similar chunks together, summarizes each group with an LLM, then groups and summarizes again, level after level. When you ask a question, the system can pull from both the detailed chunks and the higher-level summaries.

## How it works

There are two parts: building the tree, and querying it.

**Building the tree**

1. Split the input documents into chunks.
2. Embed the chunks with SentenceTransformers.
3. Reduce the embeddings (PCA + UMAP) and cluster them with a soft Gaussian Mixture Model.
4. Summarize each cluster with Google Gemini.
5. Repeat the process on the summaries to build the next level, up to a chosen number of levels.

All the nodes (original chunks + summaries) end up in a single FAISS index. There's also an optional step to load the tree into Neo4j if you want it as a graph.

**Querying (3 stages)**

1. Retrieve candidate passages from FAISS by similarity.
2. Rerank them with a BGE cross-encoder.
3. Generate the final answer with a local Ollama model (Qwen 2.5).

## Setup

```bash
git clone https://github.com/your-username/RAPTOR.git
cd RAPTOR

python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate

pip install -r requirements.txt
```

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

```env
# Gemini API key for cluster summarization
GEMINI_API_KEY_1=your_gemini_api_key_here

# Ollama server
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b-instruct

# Models
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
FAISS_INDEX_PATH=faiss_index_all_texts
```

## Usage

**Ask a question on an existing index**

```bash
python scripts/run_query.py --query "Esiste qualche bando per i musei?"
```

**Build the tree from a dataset**

```bash
python scripts/run_training_pipeline.py --csv-path ../bandi_neo4j.csv --max-levels 3
```

Add `--ingest-neo4j` if you also want to load the tree into Neo4j.

**Build the tree from your own text files**

```bash
python scripts/run_custom_pipeline.py \
  --input /path/to/your/text_files/ \
  --output-index my_custom_faiss_index \
  --max-levels 3
```

Then query it:

```bash
python scripts/run_query.py --faiss-path my_custom_faiss_index --query "Your question here"
```

## Using it from Python

```python
from raptor.query_engine import RetrievalRerankGenerate

rag = RetrievalRerankGenerate(
    faiss_path="faiss_index_all_texts",
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    reranker_model="BAAI/bge-reranker-v2-m3",
    ollama_host="http://localhost:11434",
    ollama_model="qwen2.5:14b-instruct",
    temperature=0.0,
)

result = rag.process_query("Esiste qualche bando per i musei?", retrieval_k=30, rerank_k=10)
print(result["response"])
```
