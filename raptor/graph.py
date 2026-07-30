from typing import List, Dict, Any
from neo4j import GraphDatabase
from raptor.config import Config

class Neo4jGraphManager:
    """Handles optional ingestion of RAPTOR trees into Neo4j graph databases."""
    
    def __init__(
        self,
        uri: str = Config.NEO4J_URI,
        user: str = Config.NEO4J_USER,
        password: str = Config.NEO4J_PASSWORD
    ):
        self.uri = uri
        self.user = user
        self.password = password

    def ingest_raptor_tree(self, dataset_grafo: List[Dict[str, Any]]) -> None:
        """Ingests RAPTOR tree nodes and parent-child HAS_CHILD relationships into Neo4j."""
        print(f"🔹 Connecting to Neo4j ({self.uri})...")
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        with driver.session() as session:
            print("🧹 Clearing existing Neo4j graph data...")
            session.run("MATCH (n) DETACH DELETE n")
            
            nodi_per_livello = {}
            for row in dataset_grafo:
                lvl = row.get("livello", 0)
                if lvl not in nodi_per_livello:
                    nodi_per_livello[lvl] = []
                nodi_per_livello[lvl].append(row)
                
            for lvl, batch in nodi_per_livello.items():
                if lvl == 0 or lvl is None:
                    label = "Chunk"
                else:
                    label = f"SummaryL{lvl}"
                
                query_nodi = f"""
                UNWIND $batch AS row
                CREATE (n:`{label}` {{id: row.id}})
                SET n.livello = row.livello,
                    n.tipo = row.tipo,
                    n.titolo = row.titolo,
                    n.contenuto = row.contenuto
                """
                session.run(query_nodi, batch=batch)
                print(f"✅ Created {len(batch)} nodes for level {label}")

            # Create parent-child HAS_CHILD relationships
            query_rel = """
            UNWIND $batch AS row
            MATCH (parent {id: row.id})
            WHERE row.sons IS NOT NULL AND size(row.sons) > 0
            UNWIND row.sons AS son_id
            MATCH (son {id: son_id})
            MERGE (parent)-[:HAS_CHILD]->(son)
            """
            session.run(query_rel, batch=dataset_grafo)
            print("✅ Created HAS_CHILD graph relationships.")
            
        driver.close()
