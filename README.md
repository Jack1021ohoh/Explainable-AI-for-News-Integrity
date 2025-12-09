# Explainable AI for News Integrity

An automated fact-checking system that extracts claims from news articles, retrieves evidence from Wikipedia, and provides explainable verdicts.

## 🎯 Project Overview

This system implements a complete fact-checking pipeline:
```
News Article → Claim Extraction → Evidence Retrieval → Classification → Explanation
```

| Component | Description | Status |
|-----------|-------------|--------|
| **Claim Extractor** | Extracts verifiable claims from articles using LLM | ✅ Implemented |
| **Wikipedia Retriever** | Retrieves relevant evidence from Wikipedia via ChromaDB | ✅ Implemented |
| **Fact Check API** | Integrates Google Fact Check Tools API | 🚧 In Progress |
| **Classifier** | Determines claim veracity (True/False/Unverifiable) | 📋 Planned |
| **Explainer** | Generates human-readable explanations | 📋 Planned |

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Package Manager**: [uv](https://docs.astral.sh/uv/)
- **LLM Inference**: [Groq API](https://console.groq.com/) (Llama 3)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: Sentence Transformers

## 📁 Project Structure
```
Explainable-AI-for-News-Integrity/
├── src/                    # Core modules
│   ├── extractor.py        # Claim extraction (Simple + Claimify)
│   ├── retriever.py        # Wikipedia evidence retrieval
│   ├── classifier.py       # Veracity classification (TODO)
│   └── explainer.py        # Explanation generation (TODO)
├── app/                    # Web interface
│   └── main.py             # API/UI entry point
├── notebooks/              # Data processing
│   └── Big_data_WikiDB.ipynb  # Wikipedia ETL pipeline
├── data/                   # Data files (not in git)
│   └── chroma_db_wiki/     # Vector database
└── pyproject.toml          # Dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Groq API key (free at https://console.groq.com/)

### Installation
```bash
# Clone the repository
git clone https://github.com/Jack1021ohoh/Explainable-AI-for-News-Integrity.git
cd Explainable-AI-for-News-Integrity

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Build Wikipedia Database
```bash
# Run the ETL notebook to build the vector database
# This creates data/chroma_db_wiki/ (not tracked in git)
jupyter notebook notebooks/Big_data_WikiDB.ipynb
```

### Run Tests
```bash
# Test claim extractor
uv run python src/extractor.py

# Test Wikipedia retriever
uv run python src/retriever.py
```

## 📖 Usage

### Claim Extraction
```python
from src.extractor import ClaimExtractor

extractor = ClaimExtractor()
claims = extractor.extract("""
    Tesla reported record quarterly revenue of $25.5 billion in Q3 2024,
    representing a 7% increase from the same period last year.
""")

for claim in claims:
    print(f"- {claim.text} (confidence: {claim.confidence})")
```

### Evidence Retrieval
```python
from src.retriever import WikiRetriever

retriever = WikiRetriever()
evidence = retriever.retrieve("Tesla Q3 2024 revenue", top_k=5)

for doc in evidence:
    print(f"- {doc['title']}: {doc['text'][:100]}...")
```

## 👥 Team

| Member | Responsibilities |
|--------|-----------------|
| **Hung** | Claim Extractor, WikiDB Setup, Fact Check API |
| **Jack** | Classification Model, Explainer |

## 📄 License

This project is for educational purposes.

## 🔗 Links

- [GitHub Repository](https://github.com/Jack1021ohoh/Explainable-AI-for-News-Integrity)
- [Groq Console](https://console.groq.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
