# System Architecture

## Overview
This document describes the architecture of the Explainable AI for News Integrity system, a comprehensive fact-checking pipeline that combines multiple AI models, vector databases, and cloud infrastructure to provide explainable news analysis.

## System Pipeline

```
News Article Input
    ↓
1. Classification (RoBERTa)
    ↓
2. Claim Extraction (Simple/Claimify via Groq)
    ↓
3. Evidence Retrieval (ChromaDB/PostgreSQL)
    ↓
4. Fact-Checking (Perplexity API)
    ↓
5. Explanation Generation (Gemini API)
    ↓
Final Verdict with Sources
```

## Architecture Components

### 1. Source Modules (Core Business Logic)

#### [src/classifier.py](src/classifier.py) - Fake News Classification
- **Class**: `FakeNewsDetector`
- **Model**: Fine-tuned RoBERTa-base
- **Function**: Binary classification (FAKE/REAL) with confidence scores
- **Features**:
  - Auto-detects Cloud Run environment for model path selection
  - Loads from GCS-mounted volume in cloud, local path in development
- **Returns**: Classification label and confidence score (0-1)

#### [src/explainer.py](src/explainer.py) - AI Explanation Generation
- **Class**: `LLMExplainer`
- **Model**: Google Gemini 2.5 Flash
- **Function**: Generates comprehensive, human-readable explanations
- **Input**: Article text, classification, claims, Wikipedia evidence, fact-check results
- **Output**: Structured analysis with:
  - `display_status`: Verdict (False, Misleading, Unverified, Partially Verified, Verified, etc.)
  - `explanation`: Detailed reasoning
  - `key_flags`: Important indicators
  - `claim_analysis`: Per-claim verification status

#### [src/extractor.py](src/extractor.py) - Simple Claim Extraction
- **Class**: `ClaimExtractor`
- **Model**: Groq API (Llama 3.1 8B Instant)
- **Mode**: Simple (single prompt, fast)
- **Function**: Extracts verifiable factual claims from article text
- **Returns**: List of `Claim` objects with text and metadata

#### [src/extractor_claimify.py](src/extractor_claimify.py) - Advanced Claim Extraction
- **Class**: `ClaimifyExtractor`
- **Model**: Groq API (Llama 3.1 8B Instant)
- **Mode**: Claimify (3-stage pipeline for higher quality)
- **Stages**:
  1. **Prefilter**: Identifies check-worthy sentences
  2. **Extraction**: Extracts claims from filtered sentences
  3. **Deduplication**: Removes redundant claims
- **Configuration**: Adjustable max_sentences and max_claims
- **Returns**: `ExtractionResult` with claims and metadata

#### [src/retriever.py](src/retriever.py) - ChromaDB Retrieval
- **Class**: `WikiRetriever`
- **Database**: ChromaDB (local vector database)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Function**: Semantic search over Wikipedia knowledge base
- **Use Case**: Local development and testing
- **Returns**: Top-k relevant Wikipedia passages with sources

#### [src/retriever_pg.py](src/retriever_pg.py) - PostgreSQL Retrieval
- **Class**: `WikiRetrieverPG`
- **Database**: PostgreSQL + pgvector extension
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Function**: Scalable semantic search for cloud deployment
- **Use Case**: Production deployment on Cloud SQL
- **Features**:
  - Connection pooling for Cloud SQL
  - Efficient vector similarity search using pgvector
- **Returns**: Top-k relevant Wikipedia passages with sources

#### [src/perplexity_fact_checker.py](src/perplexity_fact_checker.py) - AI Fact-Checking
- **Class**: `PerplexityFactChecker`
- **API**: Perplexity Search API
- **Function**: Web-search powered fact verification
- **Process**:
  1. Query Perplexity with fact-check prompt
  2. Analyze search results with heuristics
  3. Determine verdict (TRUE, FALSE, PARTIALLY TRUE, UNVERIFIED)
  4. Extract evidence snippets and source citations
- **Returns**: Verdict, explanation, and source URLs per claim

#### [src/__init__.py](src/__init__.py)
- **Exports**: All core modules for clean imports
- **Usage**: `from src import FakeNewsDetector, LLMExplainer, ClaimExtractor, etc.`

### 2. Application Layer

#### [app/app.py](app/app.py) - Streamlit Web Interface
- **Framework**: Streamlit with custom CSS styling
- **Features**:
  - Interactive sidebar for API key configuration
  - Extractor mode selection (Simple/Claimify)
  - Evidence retrieval settings
  - Perplexity fact-checking toggle
  - Real-time analysis progress tracking
- **Model Caching**: Uses `@st.cache_resource` for efficient loading
- **Pipeline Orchestration**:
  1. Loads all required models and APIs
  2. Accepts article input (title + text)
  3. Executes 5-stage analysis pipeline
  4. Displays comprehensive results with expandable sections
- **UI Components**:
  - Color-coded verdict cards (red/yellow/green)
  - Claim cards with status badges
  - Wikipedia evidence in expandable sections
  - Perplexity fact-check results with clickable sources
  - System status indicators

#### [app/__init__.py](app/__init__.py)
- **Purpose**: Marks app directory as Python package

### 3. Configuration

#### [config/config.py](config/config.py) - Centralized Configuration
- **API Keys**:
  - `GEMINI_API_KEY`: Google Gemini 2.5 Flash for explanations
  - `GROQ_API_KEY`: Groq Llama 3.1 for claim extraction
  - `PERPLEXITY_API_KEY`: Perplexity for fact-checking
  - `GOOGLE_FACTCHECK_API_KEY`: (Optional) Google Fact Check API
- **Model Paths**: Environment-aware path selection
  - `DETECTOR_MODEL_PATH`: Auto-detects Cloud Run (`/mnt/gcs/models/`) vs local (`./models/`)
  - `SENTENCE_TRANSFORMER_PATH`: Auto-detects Cloud Run vs local
- **Database Configuration**:
  - `CHROMA_DB_PATH`: Local ChromaDB path
  - `USE_POSTGRES`: Toggle PostgreSQL vs ChromaDB
  - PostgreSQL connection settings (host, port, database, user, password)
- **Cloud Settings**:
  - `GCS_BUCKET_NAME`: Google Cloud Storage bucket
  - `CLOUD_RETRIEVER_URL`: Optional remote retriever API
- **Application Settings**:
  - `DEFAULT_EXTRACTOR_MODE`: "simple" or "claimify"
  - `MAX_CLAIMS_PER_ARTICLE`: Claim extraction limit
  - `MAX_EVIDENCE_PER_CLAIM`: Wikipedia results per claim
  - `CLAIMIFY_MAX_SENTENCES`: Sentences to process in Claimify mode

#### [.env.example](.env.example)
- **Template** for environment variables
- **Required**: GEMINI_API_KEY
- **Optional**: GROQ_API_KEY, PERPLEXITY_API_KEY
- **Paths**: Model and database path overrides

### 4. Project Structure

#### [run.py](run.py)
- **Created** simple script to run the Streamlit app
- Usage: `python run.py`
- Checks for dependencies and provides helpful error messages

## Project Structure

```
Explainable-AI-for-News-Integrity/
├── app/                                    # Web application
│   ├── __init__.py
│   └── app.py                              # Streamlit UI with full pipeline
├── src/                                    # Core modules
│   ├── __init__.py
│   ├── classifier.py                       # FakeNewsDetector (RoBERTa)
│   ├── explainer.py                        # LLMExplainer (Gemini)
│   ├── extractor.py                        # ClaimExtractor (Simple mode)
│   ├── extractor_claimify.py              # ClaimifyExtractor (3-stage)
│   ├── retriever.py                        # WikiRetriever (ChromaDB)
│   ├── retriever_pg.py                     # WikiRetrieverPG (PostgreSQL)
│   └── perplexity_fact_checker.py         # PerplexityFactChecker
├── config/
│   └── config.py                           # Environment-aware configuration
├── notebooks/                              # Development & analysis
│   ├── Big_data_WikiDB.ipynb              # Wikipedia ETL pipeline
│   ├── fake_news_classification.ipynb     # Model training
│   └── EDA_and_preprocessing.ipynb        # Data exploration
├── data/                                   # Local data (gitignored)
│   └── chroma_db_wiki/                     # ChromaDB vector store
├── models/                                 # Trained models (gitignored)
│   └── checkpoint_roberta/                 # Fine-tuned RoBERTa
├── .env.example                            # Environment template
├── run.py                                  # Application launcher
└── requirements.txt                        # Python dependencies
```

## How to Run

### 1. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API keys and paths
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
# Option 1: Using the run script
python run.py

# Option 2: Direct streamlit command
streamlit run app/app.py
```

## Cloud Deployment Architecture

### Environment Detection
The system automatically detects its runtime environment using the `K_SERVICE` environment variable:
- **Cloud Run**: Uses GCS-mounted paths (`/mnt/gcs/models/`)
- **Local**: Uses relative paths (`./models/`, `./data/`)

### Cloud Components

#### Google Cloud Run
- **Container**: Dockerized Streamlit application
- **Scaling**: Automatic based on traffic
- **Environment**: Sets `K_SERVICE` for runtime detection
- **Volume**: GCS FUSE mount for model storage

#### Cloud SQL (PostgreSQL + pgvector)
- **Database**: Fully managed PostgreSQL instance
- **Extension**: pgvector for vector similarity search
- **Access**: Private IP with VPC peering or public with SSL
- **Connection**: Uses `retriever_pg.WikiRetrieverPG`

#### Cloud Storage (GCS)
- **Models**: Stores large model files (RoBERTa, Sentence Transformers)
- **Mount**: FUSE volume mount to Cloud Run container
- **Benefits**: Avoids HuggingFace rate limits, faster cold starts

### Deployment Flow
```
Developer → GitHub → Cloud Build → Container Registry → Cloud Run
                                            ↓
                                    GCS FUSE Mount
                                            ↓
                                Cloud SQL (PostgreSQL)
```

## Benefits of This Architecture

1. **Separation of Concerns**: Business logic (src/) separate from UI (app/) and config
2. **Modularity**: Each component (classification, extraction, retrieval, fact-check, explanation) is independent
3. **Reusability**: Source modules can be imported in notebooks, scripts, or other applications
4. **Maintainability**: Easy to update individual components without affecting others
5. **Configuration Management**: Single source of truth with environment-aware defaults
6. **Scalability**: Cloud-native architecture supports both local and cloud deployment
7. **Flexibility**: Multiple options for each component (Simple vs Claimify, ChromaDB vs PostgreSQL)
8. **Explainability**: Multi-stage pipeline with transparent reasoning at each step

## Testing Individual Modules

Each module can be tested independently:

```bash
# Test fake news classifier
python src/classifier.py

# Test LLM explainer
python src/explainer.py

# Test simple claim extractor
python src/extractor.py

# Test Claimify extractor
python src/test_claimify_detailed.py

# Test ChromaDB retriever
python src/retriever.py

# Test PostgreSQL retriever
python src/retriever_pg.py

# Test Perplexity fact checker
python src/perplexity_fact_checker.py
```

## API Integration Details

### Google Gemini API
- **Purpose**: Generate comprehensive explanations
- **Model**: gemini-2.5-flash
- **Input**: Full context (article, claims, evidence, fact-checks)
- **Output**: Structured JSON with verdict, explanation, key flags, claim analysis
- **Rate Limits**: Generous free tier, handles long context

### Groq API
- **Purpose**: Claim extraction with Llama models
- **Model**: llama-3.1-8b-instant
- **Modes**: Single prompt (Simple) or 3-stage pipeline (Claimify)
- **Speed**: Very fast inference (~200 tokens/sec)
- **Rate Limits**: Free tier available

### Perplexity API
- **Purpose**: Real-time web search fact-checking
- **Model**: llama-3.1-sonar-small-128k-online
- **Features**: Web search integration, source citations
- **Output**: Verdict with evidence snippets and URLs
- **Rate Limits**: Pay-per-use pricing

## Data Flow

```
User Input (Title + Text)
    ↓
FakeNewsDetector.classify()
    ↓
ClaimExtractor.extract() or ClaimifyExtractor.extract()
    ↓
WikiRetriever.search_claims() or WikiRetrieverPG.search_claims()
    ↓
PerplexityFactChecker.check_claims() [Optional]
    ↓
LLMExplainer.generate_explanation()
    ↓
Structured Result (Verdict + Explanation + Evidence)
```

## Team Contributions

### Hung's Modules
- Wikipedia database setup (ChromaDB & PostgreSQL ETL)
- Evidence retrieval system (retriever.py, retriever_pg.py)
- Claim extraction implementation (extractor.py, extractor_claimify.py)
- Cloud SQL deployment and configuration

### Jack's Modules
- Fake news classification (classifier.py, model training)
- LLM explanation generation (explainer.py)
- Perplexity fact-checking integration (perplexity_fact_checker.py)
- Cloud Run deployment with GCS volume mounting
- System integration and orchestration (app.py)
