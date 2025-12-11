import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    CHROMA_DB_PATH, 
    SENTENCE_TRANSFORMER_MODEL,
    CHROMA_COLLECTION_NAME,
    MAX_EVIDENCE_PER_CLAIM
)



class WikiRetriever:
    """
    A retriever class that interfaces with ChromaDB to fetch relevant Wikipedia chunks.
    
    Usage:
        retriever = WikiRetriever()
        
        # Single query
        results = retriever.search("Tesla electric vehicles", top_k=5)
        
        # Batch search for multiple claims
        claims = ["Tesla reported revenue", "Elon Musk is CEO"]
        all_evidence = retriever.search_claims(claims)
    """
    
    def __init__(
        self, 
        db_path: str = None, 
        model_name: str = None,
        collection_name: str = None
    ):
        """
        Initialize the retriever with embedding model and database client.

        Args:
            db_path (str): Path to the ChromaDB persistence directory (default from config).
            model_name (str): Hugging Face model name for sentence embeddings (default from config).
        """
        self.db_path = db_path or CHROMA_DB_PATH
        self.model_name = model_name or SENTENCE_TRANSFORMER_MODEL
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME

        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name, device="cpu")

        print(f"Connecting to ChromaDB at: {self.db_path}...")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"Database path '{self.db_path}' does not exist.\n"
                f"Please run the WikiDB notebook first or set CHROMA_DB_PATH in .env"
            )
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"WikiRetriever initialized successfully. Collection: {self.collection_name}")
        except Exception as e:
            raise ValueError(
                f"Collection '{self.collection_name}' not found in database.\n"
                f"Available collections: {[c.name for c in self.client.list_collections()]}\n"
                f"Error: {e}"
            )

    def search(self, query, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the database for the most relevant documents.
        
        Args:
            query: The search query (e.g., a claim from news).
            top_k: Number of documents to retrieve.
            
        Returns:
            List of dictionaries containing:
                - text: The document content
                - source: Wikipedia article title
                - score: Distance score (lower is better)
                - id: Document ID
        """      
        if not query or not query.strip():
            return []
    
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        evidence_list = []

        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                raw_text = results['documents'][0][i]
                raw_metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                raw_dist = results['distances'][0][i] if results['distances'] else 0.0
                raw_id = results['ids'][0][i] if results['ids'] else f"doc_{i}"

                evidence = {
                    "text": str(raw_text) if raw_text else "",
                    "source": str(raw_metadata.get('title', 'Unknown Source')),
                    "score": float(raw_dist),
                    "id": str(raw_id)
                }
                evidence_list.append(evidence)

        return evidence_list
    
    def search_claims(
        self,
        claims: List[str],
        top_k: int = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for evidence for multiple claims.
        
        This is a convenience method for the pipeline integration,
        returning evidence in a format suitable for the Explainer.
        
        Args:
            claims: List of claims to search for.
            top_k: Number of results per claim (default from config).
            
        Returns:
            Dictionary mapping each claim to its evidence list.
        """
        if top_k is None:
            top_k = MAX_EVIDENCE_PER_CLAIM

        results = []

        for claim in claims:
            evidence = self.search(claim, top_k=top_k)
            results[claims] = evidence
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "db_path": self.db_path
        }

if __name__ == "__main__":
    try:
        retriever = WikiRetriever()
        
        # Show collection stats
        stats = retriever.get_collection_stats()
        print(f"\nCollection Stats:")
        print(f"  Name: {stats['name']}")
        print(f"  Documents: {stats['count']}")
        print(f"  Path: {stats['db_path']}")
        
        # Test single search
        test_query = "What is machine learning?"
        print(f"\n--- Single Search ---")
        print(f"Query: {test_query}")
        
        results = retriever.search(test_query, top_k=3)
        for i, item in enumerate(results, 1):
            print(f"\n[{i}] Source: {item['source']}")
            print(f"    Score: {item['score']:.4f}")
            print(f"    Text: {item['text'][:150]}...")
        
        # Test batch search
        test_claims = [
            "Tesla is an electric vehicle company",
            "Climate change affects global temperatures"
        ]
        print(f"\n--- Batch Search ---")
        
        batch_results = retriever.search_claims(test_claims, top_k=2)
        for claim, evidence in batch_results.items():
            print(f"\nClaim: {claim[:50]}...")
            print(f"Found {len(evidence)} evidence(s)")
            if evidence:
                print(f"  Top result: [{evidence[0]['source']}] {evidence[0]['text'][:80]}...")

    except FileNotFoundError as e:
        print(f"\n❌ Database not found: {e}")
        print("\nTo set up the database:")
        print("1. Run notebooks/Big_data_WikiDB.ipynb in Colab")
        print("2. Download the generated chroma_db_wiki folder")
        print("3. Place it in ./data/chroma_db_wiki/")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")