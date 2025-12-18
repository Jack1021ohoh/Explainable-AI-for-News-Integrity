# Explainable AI for News Integrity

An automated fact-checking system that detects fake news, extracts claims from articles, retrieves evidence from Wikipedia, and provides explainable verdicts using AI.

## 🎯 Project Overview

This system implements a complete fact-checking pipeline with a user-friendly Streamlit interface:
```
News Article → Classification → Claim Extraction → Evidence Retrieval → Fact-Checking → AI Explanation
```

The pipeline analyzes news articles through multiple AI-powered stages:
1. **Classification**: RoBERTa model detects fake/real news with confidence scores
2. **Claim Extraction**: Extracts verifiable claims using LLM (Simple or Claimify modes)
3. **Evidence Retrieval**: Searches Wikipedia knowledge base for relevant context
4. **Fact-Checking**: Perplexity AI verifies claims with web search and citations
5. **Explanation**: Gemini generates comprehensive, human-readable analysis with verdicts

| Component | Description | Status |
|-----------|-------------|--------|
| **Fake News Classifier** | RoBERTa-based model for fake news detection | ✅ Implemented |
| **LLM Explainer** | Generates human-readable explanations using Gemini API | ✅ Implemented |
| **Claim Extractor** | Two modes: Simple (fast) and Claimify (3-stage pipeline) | ✅ Implemented |
| **Wikipedia Retriever** | ChromaDB (local) or PostgreSQL + pgvector (cloud) | ✅ Implemented |
| **Perplexity Fact Checker** | AI-powered fact-checking with web search and sources | ✅ Implemented |
| **Streamlit Web App** | Interactive web interface for news analysis | ✅ Implemented |
| **Cloud Deployment** | Google Cloud Run with GCS volume mounting | ✅ Implemented |

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Web Framework**: [Streamlit](https://streamlit.io/)
- **ML Framework**: PyTorch, Transformers (Hugging Face)
- **Classifier**: RoBERTa-base fine-tuned for fake news detection
- **LLMs**:
  - [Google Gemini API](https://ai.google.dev/) (gemini-2.5-flash) for explanations
  - [Groq API](https://groq.com/) (Llama 3.1) for claim extraction
  - [Perplexity API](https://www.perplexity.ai/) for fact-checking with web search
- **Vector Database**:
  - [ChromaDB](https://www.trychroma.com/) (local development)
  - PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) (cloud deployment)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Cloud Platform**: Google Cloud Run with GCS FUSE volume mounting

## 📁 Project Structure
```
Explainable-AI-for-News-Integrity/
├── app/                        # Streamlit web application
│   ├── __init__.py
│   └── app.py                  # Main Streamlit UI
├── src/                        # Core business logic modules
│   ├── __init__.py
│   ├── classifier.py           # FakeNewsDetector (RoBERTa-based)
│   ├── explainer.py            # LLMExplainer (Gemini API)
│   ├── extractor.py            # ClaimExtractor (simple mode)
│   ├── extractor_claimify.py   # ClaimifyExtractor (3-stage pipeline)
│   ├── retriever.py            # WikiRetriever (ChromaDB)
│   ├── retriever_pg.py         # WikiRetrieverPG (PostgreSQL + pgvector)
│   └── perplexity_fact_checker.py  # PerplexityFactChecker (AI-powered)
├── config/                     # Configuration management
│   └── config.py               # Centralized configuration
├── notebooks/                  # Jupyter notebooks
│   ├── Big_data_WikiDB.ipynb   # Wikipedia ETL pipeline
│   ├── fake_news_classification.ipynb  # Model training
│   └── EDA_and_preprocessing.ipynb     # Data exploration
├── data/                       # Data files (not in git)
│   └── chroma_db_wiki/         # Vector database
├── models/                     # Trained models (not in git)
│   └── checkpoint_roberta/     # Fine-tuned RoBERTa model
├── .env.example                # Environment variables template
├── run.py                      # Application launcher script
└── requirements.txt            # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- **API Keys** (all free tier available):
  - [Google Gemini API](https://aistudio.google.com/app/apikey) - for explanations (required)
  - [Groq API](https://console.groq.com/keys) - for claim extraction (optional, improves quality)
  - [Perplexity API](https://www.perplexity.ai/settings/api) - for fact-checking (optional)
- Pre-trained RoBERTa model for fake news detection
- Wikipedia vector database (ChromaDB or PostgreSQL)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Jack1021ohoh/Explainable-AI-for-News-Integrity.git
cd Explainable-AI-for-News-Integrity

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
#   - GEMINI_API_KEY (required)
#   - GROQ_API_KEY (optional)
#   - PERPLEXITY_API_KEY (optional)
```

### Setup Required Data

#### 1. Wikipedia Vector Database
```bash
# Run the ETL notebook to build the vector database
# This creates data/chroma_db_wiki/ (not tracked in git)
jupyter notebook notebooks/Big_data_WikiDB.ipynb
```

#### 2. Fake News Classification Model
```bash
# Train the model or download pre-trained checkpoint
# This creates models/checkpoint_roberta/ (not tracked in git)
jupyter notebook notebooks/fake_news_classification.ipynb
```

### Running the Application

```bash
# Option 1: Using the run script (recommended)
python run.py

# Option 2: Direct streamlit command
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

### Testing Individual Modules

```bash
# Test fake news classifier
python src/classifier.py

# Test LLM explainer
python src/explainer.py

# Test claim extractor
python src/extractor.py

# Test Wikipedia retriever
python src/retriever.py
```

## 📖 Usage

### Web Application

1. **Launch the app**: `python run.py`
2. **Configure settings** (in sidebar):
   - Enter API keys (Gemini, Groq, Perplexity)
   - Choose claim extractor mode (Simple or Claimify)
   - Enable/disable Perplexity fact-checking
   - Adjust evidence retrieval settings
3. **Enter article** title and text
4. **Click "Analyze Article"**
5. **View comprehensive results**:
   - Overall verdict (False, Misleading, Unverified, Partially Verified, Verified, etc.)
   - AI-generated explanation with reasoning
   - Key indicators and flags
   - Extracted claims with status badges (Supported/Contradicted/Unverified)
   - Wikipedia evidence for each claim
   - Perplexity fact-check results with sources

### Programmatic Usage

#### Fake News Classification
```python
from src.classifier import FakeNewsDetector

detector = FakeNewsDetector()
label, confidence = detector.classify("""
    Breaking: Scientists discover new planet identical to Earth.
    The planet is located just 10 light years away...
""")

print(f"Classification: {label} ({confidence:.2%} confidence)")
# Output: Classification: FAKE (85.3% confidence)
```

#### AI Explanation Generation
```python
from src.explainer import LLMExplainer

explainer = LLMExplainer(api_key="your-gemini-api-key")
result = explainer.generate_explanation(
    title="Article Title",
    text="Article content...",
    classification="FAKE",
    confidence=0.85
)

print(result['display_status'])
print(result['explanation'])
# Output: Structured explanation with key flags
```

#### Evidence Retrieval
```python
# Using ChromaDB (local)
from src.retriever import WikiRetriever

retriever = WikiRetriever()
evidence = retriever.search("Climate change statistics", top_k=5)

for doc in evidence:
    print(f"[{doc['source']}] {doc['text'][:100]}...")

# Using PostgreSQL + pgvector (cloud)
from src.retriever_pg import WikiRetrieverPG

retriever_pg = WikiRetrieverPG()
evidence = retriever_pg.search("Climate change statistics", top_k=5)
```

#### Claim Extraction
```python
# Simple extractor (fast, single prompt)
from src.extractor import ClaimExtractor

extractor = ClaimExtractor(api_key="your-groq-api-key")
claims = extractor.extract("""
    Tesla reported record revenue. The company also announced
    new product launches for next quarter.
""", max_claims=5)

for claim in claims:
    print(f"- {claim.text}")

# Claimify extractor (3-stage pipeline, higher quality)
from src.extractor_claimify import ClaimifyExtractor

claimify = ClaimifyExtractor(api_key="your-groq-api-key")
result = claimify.extract(
    text="Article text here...",
    max_claims=5,
    max_sentences=5,
    use_prefilter=True
)

for claim in result.claims:
    print(f"- {claim}")
```

#### Fact-Checking with Perplexity
```python
from src.perplexity_fact_checker import PerplexityFactChecker

checker = PerplexityFactChecker(api_key="your-perplexity-api-key")
results = checker.check_claims([
    "Water boils at 100 degrees Celsius at sea level.",
    "The Great Wall of China is visible from the moon."
])

for result in results:
    print(f"Claim: {result['claim']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Sources: {result['sources']}\n")
```

## 🚀 Cloud Deployment

This project supports deployment to Google Cloud Run with PostgreSQL and GCS volume mounting.

### Architecture
- **Cloud Run**: Serverless container deployment
- **Cloud SQL (PostgreSQL)**: Vector database with pgvector extension
- **Cloud Storage (GCS)**: Model storage with FUSE volume mounting
- **Sentence Transformer**: Loaded from GCS-mounted volume to avoid HuggingFace rate limits

### Key Features
- Auto-detects Cloud Run environment (`K_SERVICE` env var)
- Automatically uses appropriate paths for models and databases
- Falls back to ChromaDB for local development
- Supports both local and cloud configurations in single codebase

See recent commits for deployment configuration details.

## 👥 Team

| Member | Responsibilities |
|--------|-----------------|
| **Hung** | Wikipedia Database Setup, Evidence Retrieval (ChromaDB & PostgreSQL), Cloud SQL Deployment, Claim Extraction (Simple & Claimify) |
| **Jack** | Fake News Classification Model, LLM Explainer, Perplexity Fact-Checking, Cloud Run Deployment with GCS, System Integration |

## 📄 License

This project is for educational purposes.

## 🔗 Links

- [GitHub Repository](https://github.com/Jack1021ohoh/Explainable-AI-for-News-Integrity)
- **API Documentation**:
  - [Google Gemini API](https://aistudio.google.com/app/apikey)
  - [Groq API](https://console.groq.com/docs/quickstart)
  - [Perplexity API](https://docs.perplexity.ai/)
- **Framework & Tools**:
  - [ChromaDB Documentation](https://docs.trychroma.com/)
  - [pgvector for PostgreSQL](https://github.com/pgvector/pgvector)
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
  - [Streamlit Documentation](https://docs.streamlit.io/)
  - [Google Cloud Run](https://cloud.google.com/run/docs)

## 📚 Additional Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and integration details
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines and workflow
