import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class WiliRetriever:
    """
    A retriever class that interfaces with ChromaDB to fetch relevant Wikipedia chunks.
    """
    
    def __init__(self, db_path: str = "./data/chroma_db_wiki", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retriever with embedding model and database client.
        
        Args:
            db_path (str): Path to the ChromaDB persistence directory.
            model_name (str): Hugging Face model name for sentence embeddings.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, device="cpu")

        print("Connecting to ChromaDB...")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database path '{db_path}' does not exist.")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="wiki_knowledge")
        print("WikiRetriever initialized successfully.")

    def search(self, query, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the database for the most relevant documents.
        
        Args:
            query (str): The search query (e.g., a claim from news).
            top_k (int): Number of documents to retrieve.
            
        Returns:
            List[Dict]: A list of dictionaries containing text, source, and score.
        """        
        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        evidence_list = []

        if results['documents']:
            for i in range(len(results['documents'][0])):
                raw_text = results['documents'][0][i]
                raw_metadata = results['metadatas'][0][i]
                raw_dist = results['distances'][0][i] if results['distances'] else 0.0
                raw_id = results['ids'][0][i]

                source_title = str(raw_metadata.get('title', 'Unknown Source'))

                clean_text = str(raw_text) if raw_text is not None else ""

                evidence = {
                    "text": clean_text,
                    "source": source_title,
                    "score": raw_dist,
                    "id": str(raw_id)
                }
                evidence_list.append(evidence)

        return evidence_list
    
if __name__ == "__main__":
    try:
        retriever = WiliRetriever()
        test_query = "Who is Avengers movies?"
        results = retriever.search(test_query)
        
        print(f"\nSearch results for '{test_query}':")
        for item in results:
            safe_text = str(item['text'])
            safe_source = str(item['source'])
            print(f"- [{safe_source}] {safe_text[:100]}...")

    except Exception as e:
        print(f"Error: {e}")