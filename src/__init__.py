"""
Source modules for the Fake News Detection System
"""
from .classifier import FakeNewsDetector
from .explainer import LLMExplainer
from .extractor import ClaimExtractor
from .retriever import WiliRetriever

__all__ = [
    'FakeNewsDetector',
    'LLMExplainer',
    'ClaimExtractor',
    'WiliRetriever',
]
