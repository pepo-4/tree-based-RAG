import time
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from ollama import Client
from raptor.config import Config

class RetrievalRerankGenerate:
    """
    3-Stage Hierarchical RAG Query Engine for RAPTOR trees:
    1. FAISS Vector Retrieval
    2. BGE CrossEncoder Reranking
    3. Ollama LLM Response Generation
    """
    
    def __init__(
        self,
        faiss_path: str = Config.FAISS_INDEX_PATH,
        embedding_model: str = Config.EMBEDDING_MODEL,
        reranker_model: str = Config.RERANKER_MODEL,
        ollama_host: str = Config.OLLAMA_HOST,
        ollama_model: str = Config.OLLAMA_MODEL,
        temperature: float = Config.OLLAMA_TEMPERATURE
    ):
        self.faiss_path = faiss_path
        self.ollama_model = ollama_model
        self.temperature = temperature
        
        print("Initializing RetrievalRerankGenerate Engine...")
        print(f"• Embedding Model: {embedding_model}")
        self.hf_embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        print(f"• Loading FAISS Index from '{faiss_path}'...")
        self.vectorstore = FAISS.load_local(
            faiss_path,
            self.hf_embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"• Reranker Model: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model, max_length=512)
        
        print(f"• Ollama Client: {ollama_host} (Model: {ollama_model})")
        self.ollama_client = Client(host=ollama_host)
        print("✅ Query Engine successfully initialized!\n")

    def retrieve(self, query: str, top_k: int = 30) -> Tuple[List[Any], float]:
        """Stage 1: Vector Retrieval from FAISS."""
        start_time = time.time()
        retrieved_docs = self.vectorstore.similarity_search(query, k=top_k)
        elapsed_time = (time.time() - start_time) * 1000
        return retrieved_docs, elapsed_time

    def rerank(self, query: str, documents: List[Any], top_k: int = 10) -> Tuple[List[Tuple[Any, float]], float]:
        """Stage 2: CrossEncoder Reranking."""
        start_time = time.time()
        if not documents:
            return [], (time.time() - start_time) * 1000
            
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        reranked_results = list(zip(documents, scores))
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        reranked_results = reranked_results[:top_k]
        
        elapsed_time = (time.time() - start_time) * 1000
        return reranked_results, elapsed_time

    def generate(
        self,
        query: str,
        reranked_docs: List[Tuple[Any, float]],
        system_prompt: Optional[str] = None
    ) -> Tuple[str, float]:
        """Stage 3: LLM Response Generation using Ollama."""
        start_time = time.time()
        
        context_parts = []
        for i, (doc, score) in enumerate(reranked_docs, 1):
            titolo = doc.metadata.get('titolo', 'Senza titolo')
            livello = doc.metadata.get('livello', 'Chunk')
            contenuto = doc.page_content
            context_parts.append(
                f"[Documento {i}] - Livello {livello}: {titolo}\n"
                f"Contenuto: {contenuto}\n"
                f"Score: {score:.4f}\n"
            )
            
        context = "\n".join(context_parts)
        
        if system_prompt is None:
            system_prompt = (
                "Sei un assistente esperto che risponde a domande basandosi esclusivamente "
                "sul contesto fornito. Usa solo le informazioni presenti nei documenti per "
                "rispondere. Se non trovi la risposta nel contesto, dillo chiaramente."
            )
            
        user_message = f"""
DOMANDA: {query}

CONTESTO:
{context}

Istruzioni: Rispondi alla domanda basandoti SOLO sul contesto fornito. 
Se la risposta non è presente nel contesto, dillo esplicitamente.
"""
        
        full_response = ""
        try:
            stream = self.ollama_client.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                options={"temperature": self.temperature},
                stream=True,
            )
            for chunk in stream:
                full_response += chunk["message"]["content"]
        except Exception as e:
            full_response = f"❌ Errore durante la generazione Ollama: {e}"
            
        elapsed_time = (time.time() - start_time) * 1000
        return full_response, elapsed_time

    def process_query(
        self,
        query: str,
        retrieval_k: int = 30,
        rerank_k: int = 10,
        system_prompt: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Executes full 3-stage RAG query pipeline."""
        if verbose:
            print("\n" + "=" * 70)
            print(f"🔍 QUERY PROCESSING: '{query}'")
            print(f"⏰ Start Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            print("=" * 70)

        retrieved_docs, time_retrieval = self.retrieve(query, retrieval_k)
        if verbose:
            print(f"   ✅ Stage 1 Vector Retrieval ({len(retrieved_docs)} docs) in {time_retrieval:.2f} ms")

        reranked_docs, time_rerank = self.rerank(query, retrieved_docs, rerank_k)
        if verbose:
            print(f"   ✅ Stage 2 CrossEncoder Reranking ({len(reranked_docs)} docs) in {time_rerank:.2f} ms")

        response, time_generation = self.generate(query, reranked_docs, system_prompt)
        if verbose:
            print(f"   ✅ Stage 3 Ollama Generation in {time_generation:.2f} ms")

        total_time = time_retrieval + time_rerank + time_generation

        result = {
            "query": query,
            "retrieval_time": time_retrieval,
            "rerank_time": time_rerank,
            "generation_time": time_generation,
            "total_time": total_time,
            "retrieved_count": len(retrieved_docs),
            "reranked_count": len(reranked_docs),
            "retrieved_docs": retrieved_docs,
            "reranked_docs": reranked_docs,
            "response": response
        }

        if verbose:
            print("\n" + "=" * 70)
            print("📊 EXECUTION LATENCY METRICS")
            print("=" * 70)
            print(f"⏱️ Retrieval:    {time_retrieval:.2f} ms ({time_retrieval/total_time*100:.1f}%)")
            print(f"⏱️ Reranking:    {time_rerank:.2f} ms ({time_rerank/total_time*100:.1f}%)")
            print(f"⏱️ Generation:   {time_generation:.2f} ms ({time_generation/total_time*100:.1f}%)")
            print(f"⏱️ TOTAL:        {total_time:.2f} ms")
            print("=" * 70)
            print("\n💬 GENERATED ANSWER:")
            print("-" * 70)
            print(response)
            print("-" * 70)

        return result
