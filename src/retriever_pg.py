"""
Wikipedia Evidence Retriever - PostgreSQL + pgvector Version

This module provides retrieval of relevant Wikipedia content using
Cloud SQL PostgreSQL with pgvector extension.
"""

import os
import sys
from typing import List, Dict, Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    SENTENCE_TRANSFORMER_MODEL,
    SENTENCE_TRANSFORMER_PATH,
    MAX_EVIDENCE_PER_CLAIM
)


class WikiRetrieverPG:
    """
    A retriever class that interfaces with PostgreSQL + pgvector
    to fetch relevant Wikipedia chunks.
    
    Usage:
        retriever = WikiRetrieverPG()
        
        # Single query
        results = retriever.search("Tesla electric vehicles", top_k=5)
        
        # Batch search for multiple claims
        claims = ["Tesla reported revenue", "Elon Musk is CEO"]
        all_evidence = retriever.search_claims(claims)
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the retriever with embedding model and database connection.

        Args:
            model_name: Hugging Face model name for sentence embeddings.
        """
        self.model_name = model_name or SENTENCE_TRANSFORMER_MODEL
        
        # Load embedding model
        print(f"Loading embedding model from: {SENTENCE_TRANSFORMER_PATH}...")
        self.model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH, device="cpu")
        
        # Connect to PostgreSQL
        print(f"Connecting to PostgreSQL at {POSTGRES_HOST}...")
        try:
            self.conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD
            )
            # Register pgvector type
            register_vector(self.conn)
            print("WikiRetrieverPG initialized successfully.")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to PostgreSQL: {e}\n"
                f"Host: {POSTGRES_HOST}, DB: {POSTGRES_DB}"
            )
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the database for the most relevant documents.
        
        Args:
            query: The search query (e.g., a claim from news).
            top_k: Number of documents to retrieve.
            
        Returns:
            List of dictionaries containing:
                - text: The document content
                - source: Wikipedia article title
                - score: Similarity score (higher is better)
                - id: Document ID
        """
        if not query or not query.strip():
            return []
        
        # Generate embedding for query
        query_embedding = self.model.encode(query).tolist()
        
        # Search using cosine similarity
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT 
                id,
                title,
                content,
                1 - (embedding <=> %s::vector) AS similarity
            FROM wiki_articles
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        
        results = cursor.fetchall()
        cursor.close()
        
        evidence_list = []
        for row in results:
            evidence = {
                "text": row["content"],
                "source": row["title"],
                "score": float(row["similarity"]) if row["similarity"] else 0.0,
                "id": str(row["id"])
            }
            evidence_list.append(evidence)
        
        return evidence_list
    
    def search_claims(
        self, 
        claims: List[str], 
        top_k: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for evidence for multiple claims.
        
        Args:
            claims: List of claims to search for.
            top_k: Number of results per claim.
            
        Returns:
            Dictionary mapping each claim to its evidence list.
        """
        if top_k is None:
            top_k = MAX_EVIDENCE_PER_CLAIM
            
        results = {}
        for claim in claims:
            evidence = self.search(claim, top_k=top_k)
            results[claim] = evidence
            
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM wiki_articles")
        count = cursor.fetchone()[0]
        cursor.close()
        
        return {
            "host": POSTGRES_HOST,
            "database": POSTGRES_DB,
            "document_count": count
        }
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# For backward compatibility - factory function
def get_retriever(use_postgres: bool = None):
    """
    Get the appropriate retriever based on configuration.
    
    Args:
        use_postgres: Force PostgreSQL (True) or ChromaDB (False).
                     If None, uses USE_POSTGRES from config.
    
    Returns:
        WikiRetrieverPG or WikiRetriever instance.
    """
    from config.config import USE_POSTGRES
    
    if use_postgres is None:
        use_postgres = USE_POSTGRES
    
    if use_postgres:
        return WikiRetrieverPG()
    else:
        from src.retriever import WikiRetriever
        return WikiRetriever()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" PostgreSQL + pgvector Retriever Test")
    print("=" * 60)
    
    try:
        retriever = WikiRetrieverPG()
        
        # Show stats
        stats = retriever.get_stats()
        print(f"\nDatabase Stats:")
        print(f"  Host: {stats['host']}")
        print(f"  Database: {stats['database']}")
        print(f"  Documents: {stats['document_count']}")
        
        if stats['document_count'] == 0:
            print("\n⚠️  Database is empty. Please run the ETL notebook to load data.")
        else:
            # Test search
            test_query = "What is machine learning?"
            print(f"\nTest Query: {test_query}")
            
            results = retriever.search(test_query, top_k=3)
            for i, r in enumerate(results, 1):
                print(f"\n[{i}] {r['source']} (score: {r['score']:.4f})")
                print(f"    {r['text'][:100]}...")
        
        retriever.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")