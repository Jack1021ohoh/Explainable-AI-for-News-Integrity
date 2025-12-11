"""
Source modules for the Explainable AI News Integrity System

This package contains the core business logic modules:
- FakeNewsDetector: RoBERTa-based fake news classification
- LLMExplainer: Gemini-powered explanation generation
- ClaimExtractor: Simple single-prompt claim extraction
- ClaimifyExtractor: Three-stage pipeline claim extraction (higher quality)
- WikiRetriever: Wikipedia evidence retrieval via ChromaDB
- FactChecker: Google Fact Check API integration
"""

from .classifier import FakeNewsDetector
from .explainer import LLMExplainer
from .extractor import ClaimExtractor
from .extractor_claimify import ClaimifyExtractor
from .retriever import WikiRetriever
# from .factchecker import FactChecker

__all__ = [
    # Classification
    'FakeNewsDetector',
    
    # Explanation
    'LLMExplainer',
    
    # Claim Extraction
    'ClaimExtractor',
    'ClaimifyExtractor',
    
    # Evidence Retrieval
    'WikiRetriever',
    
    # Fact Checking
    'FactChecker',
]