import os

# Gemini API Configuration
GEMINI_MODEL_NAME = 'gemini-2.5-flash'

# Google Fact Check API Configuration
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")

# Model paths
DETECTOR_MODEL_NAME = "roberta-base"
DETECTOR_MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH", "./models/checkpoint_roberta")

# ChromaDB Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db_wiki")

# Sentence Transformer Model for WikiRetriever
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
