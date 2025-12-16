"""
Centralized Configuration for Explainable AI News Integrity System
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# API Keys
# =============================================================================

# Google Gemini API (for LLM Explainer)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Groq API (for Claim Extractor)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

# Google Fact Check API
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")

# Perplexity API (for Fact Checking)
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_MODEL = "llama-3.1-sonar-small-128k-online"

# =============================================================================
# Model Paths
# =============================================================================

# Fake News Detector (RoBERTa)
DETECTOR_MODEL_NAME = "roberta-base"
DETECTOR_MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH", "./models/checkpoint_roberta")

# =============================================================================
# Vector Database (ChromaDB)
# =============================================================================

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db_wiki")
CHROMA_COLLECTION_NAME = "wiki_knowledge"

# =============================================================================
# Vector Database (PostgreSQL + pgvector) - Cloud SQL
# =============================================================================

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "wikidb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# Use PostgreSQL instead of ChromaDB
USE_POSTGRES = os.getenv("USE_POSTGRES", "true").lower() == "true"

# Sentence Transformer Model for Embeddings
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# =============================================================================
# Cloud Configuration (for future Cloud Run deployment)
# =============================================================================

# Google Cloud Storage
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "news-integrity-wikidb")
GCS_WIKIDB_PATH = os.getenv("GCS_WIKIDB_PATH", "chroma_db_wiki.zip")

# Cloud Run Retriever API (optional, for cloud deployment)
CLOUD_RETRIEVER_URL = os.getenv("CLOUD_RETRIEVER_URL", "")

# =============================================================================
# Extractor Configuration
# =============================================================================

# Default extractor mode: "simple" or "claimify"
DEFAULT_EXTRACTOR_MODE = os.getenv("DEFAULT_EXTRACTOR_MODE", "simple")

# Claimify settings
CLAIMIFY_MAX_SENTENCES = int(os.getenv("CLAIMIFY_MAX_SENTENCES", "5"))
CLAIMIFY_USE_PREFILTER = os.getenv("CLAIMIFY_USE_PREFILTER", "true").lower() == "true"

# =============================================================================
# Application Settings
# =============================================================================

# Maximum claims to extract per article
MAX_CLAIMS_PER_ARTICLE = int(os.getenv("MAX_CLAIMS_PER_ARTICLE", "10"))

# Maximum evidence per claim from Wikipedia
MAX_EVIDENCE_PER_CLAIM = int(os.getenv("MAX_EVIDENCE_PER_CLAIM", "3"))

# Maximum fact-checks per claim
MAX_FACTCHECKS_PER_CLAIM = int(os.getenv("MAX_FACTCHECKS_PER_CLAIM", "3"))