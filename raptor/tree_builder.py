import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from raptor.config import Config
from raptor.clustering import perform_soft_clustering
from raptor.summarization import GeminiClusterSummarizer, estrai_titolo_e_contenuto

class RaptorTreeBuilder:
    """Orchestrates multi-level RAPTOR tree creation across raw documents and abstractive summaries."""
    
    def __init__(
        self,
        embedding_model_name: str = Config.EMBEDDING_MODEL,
        summarizer: GeminiClusterSummarizer = None
    ):
        self.embedding_model_name = embedding_model_name
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.summarizer = summarizer or GeminiClusterSummarizer()

    def build_tree_for_documents(
        self,
        documents: List[Document],
        max_levels: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Builds a multi-level tree starting from raw documents (Level 0).
        Returns dataset_grafo containing nodes for raw chunks and multi-level summaries.
        """
        dataset_grafo = []
        
        processed_chunks = set()
        
        # Add Level 0 nodes
        for idx, doc in enumerate(documents):
            chunk_id = doc.metadata.get("chunk_id", f"chunk_{idx}")
            
            if chunk_id in processed_chunks:
                continue
                
            titolo, contenuto = estrai_titolo_e_contenuto(doc.page_content)
            
            if not titolo and "titolo_bando" in doc.metadata:
                titolo = doc.metadata["titolo_bando"]

            dataset_grafo.append({
                "id": chunk_id,
                "tipo": "chunk",
                "titolo": titolo,
                "contenuto": contenuto,
                "livello": 0,
                "sons": []
            })
            processed_chunks.add(chunk_id)

        current_level_docs = documents
        current_level = 1

        while current_level <= max_levels and len(current_level_docs) > 1:
            print(f"\n🌳 Building RAPTOR Tree Level {current_level} for {len(current_level_docs)} documents...")
            
            # 1. Embed documents
            texts = [doc.page_content for doc in current_level_docs]
            embeddings = np.array(self.embeddings_model.embed_documents(texts))
            
            # 2. Perform soft clustering
            clusters, stats = perform_soft_clustering(
                documents=current_level_docs,
                embeddings=embeddings,
                level=current_level
            )
            if not clusters:
                print("No further clusters formed. Stopping tree construction.")
                break
                
            next_level_docs = []
            
            # 3. Summarize each cluster/topic
            for g_id, topics in clusters.items():
                for t_id, docs_in_topic in topics.items():
                    if len(docs_in_topic) < 1:
                        continue
                        
                    # Format context for summarizer
                    context_blocks = []
                    child_ids = []
                    for d_idx, doc in enumerate(docs_in_topic):
                        context_blocks.append(f"--- Frammento {d_idx+1} ---\n{doc.page_content}")
                        child_ids.append(doc.metadata.get("chunk_id", f"doc_{d_idx}"))
                        
                    context_str = "\n\n".join(context_blocks)
                    
                    # Generate summary
                    titolo_summary, contenuto_summary = self.summarizer.generate_summary(context_str)
                    
                    summary_id = f"summary_L{current_level}_g{g_id}_t{t_id}"
                    summary_node = {
                        "id": summary_id,
                        "tipo": "summary",
                        "titolo": titolo_summary,
                        "contenuto": contenuto_summary,
                        "livello": current_level,
                        "sons": child_ids
                    }
                    dataset_grafo.append(summary_node)
                    
                    # Create Document object for the next level
                    summary_text = f"Titolo: {titolo_summary}\nContenuto: {contenuto_summary}"
                    next_doc = Document(
                        page_content=summary_text,
                        metadata={
                            "chunk_id": summary_id,
                            "titolo": titolo_summary,
                            "livello": current_level
                        }
                    )
                    next_level_docs.append(next_doc)
                    
            print(f"✅ Generated {len(next_level_docs)} Level {current_level} summaries.")
            current_level_docs = next_level_docs
            current_level += 1

        return dataset_grafo
