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
uv sync

# Set up environment
cp .env.example .env
# Add your API keys to .env

# Run tests
uv run python src/extractor.py
uv run python src/retriever.py
```

## 📂 Code Organization

| Directory | Purpose | Owner |
|-----------|---------|-------|
| `src/extractor.py` | Claim extraction | Hung |
| `src/retriever.py` | Wikipedia retrieval | Hung |
| `src/classifier.py` | Veracity classification | Jack |
| `src/explainer.py` | Explanation generation | Jack |
| `app/` | User interface | TBD |
| `notebooks/` | Data processing | Shared |

## ❓ Questions?

If unsure about anything, ask in the PR or create a discussion issue!
