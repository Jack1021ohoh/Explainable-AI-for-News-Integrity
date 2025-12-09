# Explainable AI for News Integrity

An automated fact-checking system that detects fake news, extracts claims from articles, retrieves evidence from Wikipedia, and provides explainable verdicts using AI.

## 🎯 Project Overview

This system implements a complete fact-checking pipeline with a user-friendly Streamlit interface:
```
News Article → Classification → Claim Extraction → Evidence Retrieval → AI Explanation
```

| Component | Description | Status |
|-----------|-------------|--------|
| **Fake News Classifier** | RoBERTa-based model for fake news detection | ✅ Implemented |
| **LLM Explainer** | Generates human-readable explanations using Gemini API | ✅ Implemented |
| **Claim Extractor** | Extracts verifiable claims from articles | ✅ Implemented |
| **Wikipedia Retriever** | Retrieves relevant evidence from Wikipedia via ChromaDB | ✅ Implemented |
| **Streamlit Web App** | Interactive web interface for news analysis | ✅ Implemented |
| **Fact Check API** | Integrates Google Fact Check Tools API | 📋 Planned |

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Web Framework**: [Streamlit](https://streamlit.io/)
- **ML Framework**: PyTorch, Transformers (Hugging Face)
- **Classifier**: RoBERTa-base fine-tuned for fake news detection
- **LLM**: [Google Gemini API](https://ai.google.dev/) (gemini-2.0-flash-exp)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)

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
│   ├── extractor.py            # ClaimExtractor
│   └── retriever.py            # WiliRetriever (Wikipedia + ChromaDB)
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
- Google Gemini API key (free at https://aistudio.google.com/app/apikey)
- Pre-trained RoBERTa model for fake news detection
- Wikipedia vector database (ChromaDB)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Jack1021ohoh/Explainable-AI-for-News-Integrity.git
cd Explainable-AI-for-News-Integrity

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY and paths
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

1. Launch the app: `python run.py`
2. Enter article title and text
3. Click "Analyze Article"
4. View results:
   - Classification (FAKE/REAL) with confidence score
   - AI-generated explanation
   - Key indicators and flags

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
from src.retriever import WiliRetriever

retriever = WiliRetriever()
evidence = retriever.search("Climate change statistics", top_k=5)

for doc in evidence:
    print(f"[{doc['source']}] {doc['text'][:100]}...")
```

#### Claim Extraction
```python
from src.extractor import ClaimExtractor

extractor = ClaimExtractor()
claims = extractor.extract_claims("""
    Tesla reported record revenue. The company also announced
    new product launches for next quarter.
""")

for claim in claims:
    print(f"- {claim}")
```

## 👥 Team

| Member | Responsibilities |
|--------|-----------------|
| **Hung** | Wikipedia Database Setup, Evidence Retrieval, Fact Check API Integration |
| **Jack** | Fake News Classification Model, LLM Explainer, System Integration |

## 📄 License

This project is for educational purposes.

## 🔗 Links

- [GitHub Repository](https://github.com/Jack1021ohoh/Explainable-AI-for-News-Integrity)
- [Google Gemini API](https://aistudio.google.com/app/apikey)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📚 Additional Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and integration details
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines and workflow
