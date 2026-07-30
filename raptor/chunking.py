import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from raptor.config import Config

class TextChunker:
    """Handles document chunking and metadata generation using HuggingFace Tokenizers."""
    
    def __init__(
        self,
        model_id: str = Config.EMBEDDING_MODEL,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "; ", ". ", " ", ""]
        )

    def chunk_dataframe_by_sections(
        self,
        df: pd.DataFrame,
        section_columns: List[str],
        id_col: str = "bando_id",
        title_col: str = "title"
    ) -> Dict[str, List[Document]]:
        """
        Splits text content from dataframe columns into Document objects grouped by section.
        """
        datasets_per_sezione = {
            col.replace("section_", "").capitalize(): []
            for col in section_columns
        }

        for _, row in df.iterrows():
            titolo = row.get(title_col, "Titolo Sconosciuto")
            bando_id = str(row.get(id_col, "ID_Sconosciuto"))
            
            for colonna in section_columns:
                if pd.notna(row.get(colonna)):
                    testo_sezione = str(row[colonna]).strip()
                    if testo_sezione:
                        nome_sezione = colonna.replace("section_", "").capitalize()
                        frammenti_testo = self.text_splitter.split_text(testo_sezione)
                        
                        for idx, frammento in enumerate(frammenti_testo):
                            testo_arricchito = f"Titolo: {titolo} - {nome_sezione}\nContenuto: {frammento}"
                            chunk_id = f"{bando_id}_{nome_sezione}_{idx}"
                            
                            meta_chunk = {
                                "chunk_id": chunk_id,
                                "bando_id": bando_id,
                                "titolo_bando": titolo,
                                "sezione_testo": nome_sezione,
                                "livello": 0
                            }
                            
                            doc = Document(
                                page_content=testo_arricchito,
                                metadata=meta_chunk
                            )
                            datasets_per_sezione[nome_sezione].append(doc)
                            
        return datasets_per_sezione

    def chunk_texts(
        self,
        texts: List[Dict[str, str]],
        text_key: str = "text",
        title_key: str = "title"
    ) -> List[Document]:
        """
        Generic helper to chunk a list of text dicts for custom user datasets.
        """
        documents = []
        for idx, item in enumerate(texts):
            title = item.get(title_key, f"Document_{idx}")
            raw_text = item.get(text_key, "").strip()
            if not raw_text:
                continue
                
            chunks = self.text_splitter.split_text(raw_text)
            for c_idx, chunk in enumerate(chunks):
                doc_content = f"Titolo: {title}\nContenuto: {chunk}"
                meta = {
                    "chunk_id": f"doc_{idx}_chunk_{c_idx}",
                    "titolo": title,
                    "livello": 0
                }
                documents.append(Document(page_content=doc_content, metadata=meta))
                
        return documents
