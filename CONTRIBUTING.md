# Contributing Guidelines

This document outlines the Git workflow and coding standards for our team.

## 🌿 Branch Naming Convention

Use the following prefixes:

| Prefix | Use Case | Example |
|--------|----------|---------|
| `feature/` | New functionality | `feature/claim-extractor` |
| `fix/` | Bug fixes | `fix/retriever-null-check` |
| `chore/` | Maintenance tasks | `chore/update-dependencies` |
| `docs/` | Documentation | `docs/update-readme` |
| `refactor/` | Code restructuring | `refactor/extractor-cleanup` |

### Examples
```bash
# Create a new feature branch
git checkout -b feature/google-factcheck-api

# Create a bug fix branch
git checkout -b fix/extractor-json-parsing
```

## 📝 Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):
```
<type>: <short description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Formatting (no code change) |
| `refactor` | Code restructuring |
| `test` | Adding tests |
| `chore` | Maintenance tasks |

### Examples
```bash
# Good commits ✅
git commit -m "feat: add ClaimExtractor with Groq API integration"
git commit -m "fix: handle empty response in retriever"
git commit -m "docs: update installation instructions"
git commit -m "chore: add python-dotenv dependency"

# Bad commits ❌
git commit -m "update"
git commit -m "fix stuff"
git commit -m "WIP"
```

### Multi-line Commit
```bash
git commit -m "feat: implement 3-stage claim extraction pipeline

- Add Selection stage to filter non-verifiable content
- Add Disambiguation stage to resolve ambiguous references
- Add Decomposition stage to extract atomic claims

Implements #3"
```

## 🔄 Git Workflow

### 1. Start New Work
```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes
```bash
# Stage changes
git add <files>

# Commit with conventional message
git commit -m "feat: description"
```

### 3. Push and Create PR
```bash
# Push to remote
git push origin feature/your-feature-name

# Then create Pull Request on GitHub
```

### 4. After PR Merged
```bash
# Switch back to main
git checkout main
git pull origin main

# Delete local feature branch
git branch -d feature/your-feature-name
```

## 📋 Issue Guidelines

### Creating Issues

Use clear titles with prefixes:
```
[Feature] Add Google Fact Check API integration
[Bug] Extractor returns empty list for valid articles
[Docs] Update README with usage examples
```

### Issue Template
```markdown
## Description
Brief description of the issue or feature.

## Tasks
- [ ] Task 1
- [ ] Task 2

## Acceptance Criteria
- What defines "done"?

## Related
- Links to related issues or PRs
```

## 🔀 Pull Request Guidelines

### PR Title Format

Same as commit messages:
```
feat: add Google Fact Check API integration
```

### PR Description Template
```markdown
## Summary
Brief description of changes.

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Code follows project style
- [ ] Self-reviewed the code
- [ ] Tested locally
- [ ] Updated documentation if needed

## Related Issues
Closes #<issue-number>
```

## 🏗️ Development Setup
```bash
# Clone repo
git clone https://github.com/Jack1021ohoh/Explainable-AI-for-News-Integrity.git
cd Explainable-AI-for-News-Integrity

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your GEMINI_API_KEY and other configurations to .env

# Build required data (first time only)
# 1. Run Wikipedia database notebook
jupyter notebook notebooks/Big_data_WikiDB.ipynb

# 2. Train/setup classification model
jupyter notebook notebooks/fake_news_classification.ipynb

# Test individual modules
python src/classifier.py
python src/explainer.py
python src/extractor.py
python src/retriever.py

# Run the application
python run.py
```

## 📂 Code Organization

### Module Structure

```
Explainable-AI-for-News-Integrity/
├── app/                    # Streamlit web interface
│   ├── __init__.py
│   └── app.py             # Main UI (imports from src/)
├── src/                   # Core business logic
│   ├── __init__.py        # Module exports
│   ├── classifier.py      # FakeNewsDetector class
│   ├── explainer.py       # LLMExplainer class
│   ├── extractor.py       # ClaimExtractor class
│   └── retriever.py       # WiliRetriever class
├── config/                # Configuration
│   └── config.py          # Centralized settings
├── notebooks/             # Jupyter notebooks
│   ├── Big_data_WikiDB.ipynb
│   ├── fake_news_classification.ipynb
│   └── EDA_and_preprocessing.ipynb
└── run.py                 # Application launcher
```

### Module Ownership

| Module | Purpose | Owner |
|--------|---------|-------|
| `src/classifier.py` | Fake news classification (RoBERTa) | Jack |
| `src/explainer.py` | AI explanation generation (Gemini) | Jack |
| `src/extractor.py` | Claim extraction from articles | Hung |
| `src/retriever.py` | Wikipedia evidence retrieval | Hung |
| `app/app.py` | Streamlit web interface | Jack |
| `config/config.py` | Configuration management | Shared |
| `notebooks/` | Data processing & model training | Shared |

### Design Principles

1. **Separation of Concerns**: Business logic (src/) separate from UI (app/)
2. **Modularity**: Each component is independently testable
3. **Configuration**: Centralized in config/ with environment variable support
4. **Reusability**: All src/ modules can be imported and used standalone

## 🔧 Adding New Features

### Adding a New Source Module

1. Create your module in `src/your_module.py`
2. Import config values from `config.config`
3. Add your class/functions with proper docstrings
4. Export it in `src/__init__.py`
5. Add tests by running the module directly
6. Import and use in `app/app.py` if needed

Example structure:
```python
# src/your_module.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import YOUR_CONFIG_VALUE

class YourClass:
    def __init__(self, param=None):
        self.param = param or YOUR_CONFIG_VALUE

    def your_method(self, input):
        """Your method with clear docstring"""
        # Implementation
        return result

if __name__ == "__main__":
    # Test code
    obj = YourClass()
    result = obj.your_method("test")
    print(result)
```

### Modifying the Streamlit App

The app is in `app/app.py` and follows this pattern:
1. Import from `src/` modules (not inline classes)
2. Use `@st.cache_resource` for model loading
3. Keep UI logic separate from business logic
4. Test changes with `python run.py`

### Updating Configuration

Add new config values to `config/config.py`:
```python
# config/config.py
import os

NEW_CONFIG = os.getenv("NEW_CONFIG", "default_value")
```

Then update `.env.example`:
```bash
# .env.example
NEW_CONFIG=your-value-here
```

## ❓ Questions?

If unsure about anything, ask in the PR or create a discussion issue!
