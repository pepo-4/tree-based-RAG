import os
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from raptor.config import Config

class VectorStoreManager:
    """Manages FAISS index creation, batch document embedding, saving, and loading."""
    
    def __init__(
        self,
        embedding_model_name: str = Config.EMBEDDING_MODEL,
        faiss_path: str = Config.FAISS_INDEX_PATH
    ):
        self.embedding_model_name = embedding_model_name
        self.faiss_path = faiss_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def create_and_save_index(
        self,
        dataset_grafo: List[Dict[str, Any]],
        batch_size: int = 100,
        output_path: str = None
    ) -> FAISS:
        """
        Converts dataset_grafo graph nodes into LangChain Document objects,
        embeds them in batches, builds a FAISS index, and saves it to disk.
        """
        save_path = output_path or self.faiss_path
        documents = []
        
        print(f"Creating FAISS vector index from {len(dataset_grafo)} nodes...")
        for node in dataset_grafo:
            titolo = node.get("titolo", "Senza titolo")
            contenuto = node.get("contenuto", "")
            testo_completo = f"Titolo: {titolo}\nContenuto: {contenuto}"
            
            meta = {
                "id": node.get("id"),
                "tipo": node.get("tipo", "chunk"),
                "titolo": titolo,
                "livello": node.get("livello", 0),
                "sons": node.get("sons", [])
            }
            documents.append(Document(page_content=testo_completo, metadata=meta))

        # First batch
        primo_batch = documents[:batch_size]
        print(f"Building initial FAISS vectorstore with first batch ({len(primo_batch)} docs)...")
        vectorstore = FAISS.from_documents(primo_batch, self.embeddings)

        # Remaining batches
        if len(documents) > batch_size:
            print("Embedding remaining documents incrementally...")
            for i in tqdm(range(batch_size, len(documents), batch_size), desc="Indexing Batches"):
                batch = documents[i : i + batch_size]
                vectorstore.add_documents(batch)

        print(f"💾 Saving FAISS index to '{save_path}'...")
        vectorstore.save_local(save_path)
        return vectorstore

    def load_index(self, path: str = None) -> FAISS:
        """Loads a FAISS index from disk."""
        target_path = path or self.faiss_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"FAISS index folder '{target_path}' not found.")
            
        return FAISS.load_local(
            target_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
