# System Architecture

## Overview
This document describes the architecture of the Explainable AI for News Integrity system, including the modular design and how components work together.

## Architecture Components

### 1. Source Modules (Core Business Logic)

#### [src/classifier.py](src/classifier.py)
- **Created** `FakeNewsDetector` class
- Extracted from the original `app.py`
- Uses RoBERTa-based model for fake news classification
- Imports configuration from `config.config`
- Returns classification label ("FAKE" or "REAL") with confidence score

#### [src/explainer.py](src/explainer.py)
- **Created** `LLMExplainer` class
- Extracted from the original `app.py`
- Uses Google Gemini API for generating explanations
- Falls back to simple rule-based explanations if API is unavailable
- Returns structured JSON with display_status, explanation, and key_flags

#### [src/extractor.py](src/extractor.py)
- **Created** `ClaimExtractor` class
- Placeholder implementation for claim extraction
- Can be enhanced with NLP models in the future

#### [src/retriever.py](src/retriever.py)
- **Updated** to use centralized configuration
- Now imports `CHROMA_DB_PATH` and `SENTENCE_TRANSFORMER_MODEL` from config
- Maintains existing Wikipedia retrieval functionality

#### [src/__init__.py](src/__init__.py)
- **Updated** to export all source modules
- Enables clean imports: `from src import FakeNewsDetector, LLMExplainer, etc.`

### 2. Application Layer

#### [app/app.py](app/app.py)
- **Refactored** to import from `src` modules instead of defining classes inline
- Removed duplicate class definitions (FakeNewsDetector, LLMExplainer, etc.)
- Added model caching with `@st.cache_resource` decorator
- Cleaner, more maintainable code structure
- All business logic now separated into `src/` modules

#### [app/__init__.py](app/__init__.py)
- **Created** to mark app as a Python package

#### app/main.py
- **Deleted** as it was not needed (as requested)

### 3. Configuration

#### [config/config.py](config/config.py)
- **Updated** with environment variable support
- Changed Gemini model to `gemini-2.0-flash-exp`
- Added `SENTENCE_TRANSFORMER_MODEL` configuration
- Model paths now use environment variables with defaults:
  - `DETECTOR_MODEL_PATH`: defaults to `./models/checkpoint_roberta`
  - `CHROMA_DB_PATH`: defaults to `./data/chroma_db_wiki`

#### [.env.example](.env.example)
- **Updated** to reflect new configuration structure
- Removed GROQ_API_KEY (not used)
- Added GEMINI_API_KEY
- Added path configurations

### 4. Project Structure

#### [run.py](run.py)
- **Created** simple script to run the Streamlit app
- Usage: `python run.py`
- Checks for dependencies and provides helpful error messages

## New Project Structure

```
Explainable-AI-for-News-Integrity/
├── app/
│   ├── __init__.py          # Package marker
│   └── app.py               # Streamlit UI (refactored)
├── src/
│   ├── __init__.py          # Module exports
│   ├── classifier.py        # FakeNewsDetector class
│   ├── explainer.py         # LLMExplainer class
│   ├── extractor.py         # ClaimExtractor class
│   └── retriever.py         # WiliRetriever class (updated)
├── config/
│   └── config.py            # Centralized configuration
├── .env.example             # Environment variable template
└── run.py                   # Startup script
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

## Benefits of This Architecture

1. **Separation of Concerns**: Business logic (src/) separate from UI (app/)
2. **Modularity**: Each component is in its own file and can be tested independently
3. **Reusability**: Source modules can be imported and used in other scripts
4. **Maintainability**: Easier to update and debug individual components
5. **Configuration Management**: Centralized config with environment variable support
6. **Scalability**: Easy to add new features without modifying existing code

## Testing Individual Modules

Each module can be tested independently:

```bash
# Test classifier
python src/classifier.py

# Test explainer
python src/explainer.py

# Test extractor
python src/extractor.py

# Test retriever
python src/retriever.py
```

## Next Steps

1. Add unit tests for each module in `tests/` directory
2. Enhance `ClaimExtractor` with actual NLP implementation
3. Add logging throughout the application
4. Create a requirements.txt if not already present
5. Add error handling and validation
6. Consider adding a CLI interface in addition to Streamlit
