# RAPTOR – Hierarchical RAG

A Python implementation of **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval), a hierarchical take on Retrieval-Augmented Generation.

Instead of searching over a flat list of text chunks, RAPTOR builds a tree over the corpus: it groups similar chunks together, summarizes each group with an LLM, then groups and summarizes again, level after level. When you ask a question, the system can pull from both the detailed chunks and the higher-level summaries.

## How it works

### The idea behind it

Standard RAG splits a document into chunks, embeds them, and retrieves the top-k chunks most similar to your question. That works well when the answer lives in a single spot, but it falls apart in two common cases: when the answer is spread across many chunks that each only hold a piece of it, and when the question is broad and really needs a high-level understanding of a topic rather than one specific passage.

RAPTOR tries to fix this by not stopping at the raw chunks. It builds a tree on top of them. The bottom level (level 0) is the original text, split into chunks. Above that, chunks that talk about similar things get grouped and summarized into new nodes. Those summaries then get grouped and summarized again, and so on, so each level up is a bit more abstract than the one below. The summaries themselves become searchable nodes, so retrieval can land on a broad overview or a precise detail depending on what the question needs.

### Building the tree

The tree is built bottom-up, one level at a time.

**1. Chunking.** The input documents are split into level-0 chunks using a tokenizer-based recursive splitter, so chunks stay within a token budget instead of being cut at arbitrary character counts.

**2. Embedding.** Each chunk is turned into a vector with a SentenceTransformers model (a multilingual mpnet model here, so it handles non-English text). These vectors are what everything downstream works on.

**3. Dimensionality reduction.** Embeddings are high-dimensional (hundreds of dimensions), and clustering in that space is noisy and slow. So before clustering, the vectors get reduced with **PCA** first and then **UMAP**. UMAP is run in two passes: a *global* pass that looks at the whole set of vectors to find broad structure, and a *local* pass that refines the grouping within each broad region. This global-then-local approach keeps both the big-picture topics and the finer distinctions.

**4. Clustering.** Grouping is done with a **soft Gaussian Mixture Model**. "Soft" is the important word: a chunk isn't forced into exactly one cluster. It gets a probability of belonging to each cluster, and it's assigned to every cluster where that probability crosses a threshold (cumulative assignment). This matters because a piece of text often relates to more than one topic, and forcing it into a single bucket would lose that. Clustering happens globally first, then locally inside each global cluster, mirroring the two-pass UMAP.

**5. Summarization.** For each cluster, the text of its members is concatenated and sent to **Google Gemini**, which writes an abstractive summary of the whole group. That summary becomes a new node one level up. Because summarizing every cluster of a large corpus means a lot of API calls, the summarizer **rotates through multiple API keys** to spread the load and stay under rate limits.

**6. Recursion.** The summaries produced at this level become the input for the next level: they get embedded, reduced, clustered, and summarized again. The loop keeps going until it reaches the maximum number of levels you set, or until there are too few nodes left to keep clustering (near the top of the tree there might only be a handful of summaries, eventually collapsing toward a root).

### A note on the two clustering pipelines

The clustering step doesn't use the exact same settings at every level, because the situation is very different at the bottom versus the top of the tree.

- **Level 0 → 1** deals with a large number of chunks, so it uses fixed PCA (keeping ~90% of the variance), a global UMAP with the neighbor count scaled to the dataset size, and a local UMAP with fixed `n_neighbors=10` and a Euclidean metric. These settings are stable on big arrays.
- **Level 1 → N** deals with far fewer nodes, and the settings that work on thousands of chunks actually crash on tiny arrays. So here it switches to dynamic PCA bounds, skips the local UMAP entirely when there are 3 or fewer nodes, and uses `min_dist=0.0` with a cosine metric and random init. This keeps dimensionality reduction from blowing up when there's barely anything left to reduce.

### The index

Once the tree is built, all the nodes from every level — the original chunks and every summary — are flattened into a single **FAISS** vector index. There's no separate store per level; retrieval searches the whole tree at once, which is what lets a query surface a detailed chunk and a high-level summary side by side. Optionally, the tree can also be loaded into **Neo4j** if you want to keep the parent/child structure as an actual graph.

### Answering a question (3 stages)

Querying runs the retrieved text through three stages, from fast-and-rough to slow-and-precise.

**Stage 1 – Retrieval.** The question is embedded with the same model used for the chunks, and FAISS does a similarity search over the whole index, returning the top-k candidates (30 by default). This is fast but only approximate — it's a wide net meant to not miss anything relevant.

**Stage 2 – Reranking.** The candidates from stage 1 are passed to a **BGE cross-encoder reranker**. Unlike the embedding search, which compares the question and each document separately, a cross-encoder reads the question and the document *together* and scores how well they actually match. It's slower, which is exactly why it's only run on the shortlist from stage 1 rather than the whole corpus. The top-k after reranking (10 by default) are kept.

**Stage 3 – Generation.** The best reranked passages are stitched into a prompt along with the question and sent to a **local Ollama model (Qwen 2.5)**, which writes the final answer grounded in that context. Running the LLM locally keeps the whole query side off external APIs.

## Setup

```bash
git clone https://github.com/your-username/RAPTOR.git
cd RAPTOR

python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then fill in your keys
```

Key things to set in `.env`: your `GEMINI_API_KEY_1` (for summarization), the Ollama host/model, and the embedding, reranker, and FAISS index settings.

## Usage

Ask a question on an existing index:

```bash
python scripts/run_query.py --query "Esiste qualche bando per i musei?"
```

Build the tree from a dataset (add `--ingest-neo4j` to also load it into Neo4j):

```bash
python scripts/run_training_pipeline.py --csv-path ../bandi_neo4j.csv --max-levels 3
```

Build the tree from your own text files, then query it:

```bash
python scripts/run_custom_pipeline.py --input /path/to/text_files/ --output-index my_index --max-levels 3
python scripts/run_query.py --faiss-path my_index --query "Your question here"
```

Or use it from Python:

```python
from raptor.query_engine import RetrievalRerankGenerate

rag = RetrievalRerankGenerate(faiss_path="faiss_index_all_texts", temperature=0.0)
result = rag.process_query("Esiste qualche bando per i musei?", retrieval_k=30, rerank_k=10)
print(result["response"])
```
