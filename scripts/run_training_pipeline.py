import argparse
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from raptor.config import Config
from raptor.chunking import TextChunker
from raptor.tree_builder import RaptorTreeBuilder
from raptor.vectorstore import VectorStoreManager
from raptor.graph import Neo4jGraphManager

def main():
    parser = argparse.ArgumentParser(description="Run RAPTOR Multi-Level Tree Training Pipeline")
    parser.add_argument("--csv-path", type=str, default="../bandi_neo4j.csv", help="Input CSV dataset file path")
    parser.add_argument("--faiss-output", type=str, default=Config.FAISS_INDEX_PATH, help="FAISS vector store output folder")
    parser.add_argument("--max-levels", type=int, default=3, help="Max tree levels (Level 0 -> Level N)")
    parser.add_argument("--ingest-neo4j", action="store_true", help="Ingest generated tree into Neo4j graph database")
    
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f" Input CSV file '{args.csv_path}' not found!")
        return

    print(f" Loading CSV dataset from '{args.csv_path}'...")
    df = pd.read_csv(args.csv_path)

    section_columns = [
        col for col in df.columns if col.startswith("section_")
    ]
    if not section_columns:
        print(" No columns starting with 'section_' found. Available columns:", list(df.columns))
        return

    print(f" Splitting document text across sections: {section_columns}...")
    chunker = TextChunker()
    datasets_per_sezione = chunker.chunk_dataframe_by_sections(
        df=df,
        section_columns=section_columns,
        id_col="bando_id",
        title_col="title"
    )

    all_dataset_nodes = []
    tree_builder = RaptorTreeBuilder()

    # Process each section
    for section_name, docs in datasets_per_sezione.items():
        if not docs:
            continue
        print(f"\n==================================================")
        print(f" Processing RAPTOR Tree for Section: '{section_name}' ({len(docs)} chunks)")
        print(f"==================================================")
        
        section_grafo = tree_builder.build_tree_for_documents(
            documents=docs,
            max_levels=args.max_levels
        )
        all_dataset_nodes.extend(section_grafo)

    print(f"\n Total RAPTOR Graph Nodes generated: {len(all_dataset_nodes)}")

    # Save to FAISS
    vector_manager = VectorStoreManager(faiss_path=args.faiss_output)
    vector_manager.create_and_save_index(dataset_grafo=all_dataset_nodes)

    # Optional Neo4j Ingestion
    if args.ingest_neo4j:
        print("\n Ingesting RAPTOR tree into Neo4j...")
        graph_manager = Neo4jGraphManager()
        graph_manager.ingest_raptor_tree(dataset_grafo=all_dataset_nodes)

    print("\n RAPTOR Pipeline Execution Completed Successfully!")

if __name__ == "__main__":
    main()
