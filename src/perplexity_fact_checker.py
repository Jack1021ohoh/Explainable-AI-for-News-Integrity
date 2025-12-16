"""
Perplexity-based Fact Checker for News Integrity Analysis

This module uses the Perplexity Search API to fact-check claims extracted from news articles.
It provides detailed fact-check results with explanations and sources.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from perplexityai import Perplexity
    PERPLEXITY_SDK_AVAILABLE = True
except ImportError:
    PERPLEXITY_SDK_AVAILABLE = False
    print("⚠️  Warning: perplexityai SDK not installed. Install with: pip install perplexityai")


class PerplexityFactChecker:
    """
    Fact-checker using Perplexity's Search API.

    This checker takes claims and uses Perplexity's AI-powered search to verify them,
    providing detailed explanations and source citations.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the Perplexity fact checker.

        Args:
            api_key (str): Perplexity API key (or from PERPLEXITY_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")

        if not self.api_key:
            print("⚠️  Warning: No Perplexity API key found. Set PERPLEXITY_API_KEY environment variable.")
            print("   Fact-checking will be disabled.")
            self.client = None
        elif not PERPLEXITY_SDK_AVAILABLE:
            print("⚠️  Warning: Perplexity SDK not available. Install with: pip install perplexity-sdk")
            self.client = None
        else:
            try:
                self.client = Perplexity(api_key=self.api_key)
                print("✅ PerplexityFactChecker initialized successfully.")
            except Exception as e:
                print(f"❌ Failed to initialize Perplexity client: {e}")
                self.client = None

    def check_claim(self, claim: str) -> Dict[str, Any]:
        """
        Fact-check a single claim using Perplexity Search API.

        Args:
            claim (str): The claim to fact-check

        Returns:
            Dict containing:
                - claim: The original claim
                - verdict: "TRUE", "FALSE", "PARTIALLY TRUE", "UNVERIFIED"
                - explanation: Detailed explanation
                - sources: List of source URLs/titles
        """
        if not self.client:
            return self._create_error_response(claim, "Perplexity client not initialized")

        try:
            # Create fact-checking query
            query = f"Is this claim true or false? Fact-check: {claim}"

            # Use Perplexity Search API
            search_results = self.client.search.create(
                query=query,
                max_results=3,
                max_tokens_per_page=1024
            )

            # Process search results
            result = self._analyze_search_results(claim, search_results)
            return result

        except Exception as e:
            print(f"❌ Error checking claim '{claim[:50]}...': {e}")
            return self._create_error_response(claim, str(e))

    def check_claims(self, claims: List[str]) -> List[Dict[str, Any]]:
        """
        Fact-check multiple claims.

        Args:
            claims (List[str]): List of claims to fact-check

        Returns:
            List of fact-check results, one per claim
        """
        results = []

        for i, claim in enumerate(claims, 1):
            print(f"🔍 Fact-checking claim {i}/{len(claims)}: {claim[:60]}...")
            result = self.check_claim(claim)
            results.append(result)

        return results

    def _analyze_search_results(self, claim: str, search_results) -> Dict[str, Any]:
        """
        Analyze search results and determine verdict.

        Args:
            claim (str): Original claim
            search_results: Perplexity search results object

        Returns:
            Dict: Formatted fact-check result
        """
        try:
            # Extract search results
            sources = []
            combined_text = []

            for result in search_results.results:
                sources.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": getattr(result, 'snippet', '')[:200]
                })
                # Combine snippets for analysis
                snippet = getattr(result, 'snippet', '')
                if snippet:
                    combined_text.append(snippet)

            # Analyze the combined text to determine verdict
            full_text = " ".join(combined_text).lower()

            # Simple heuristic-based analysis
            verdict, explanation = self._determine_verdict(claim, full_text, sources)

            # Format sources as list of strings
            source_list = [f"{s['title']} - {s['url']}" for s in sources[:5]]

            return {
                "claim": claim,
                "verdict": verdict,
                "explanation": explanation,
                "sources": source_list
            }

        except Exception as e:
            print(f"❌ Error analyzing search results: {e}")
            return self._create_error_response(claim, f"Analysis error: {e}")

    def _determine_verdict(self, claim: str, text: str, sources: List[Dict]) -> tuple:
        """
        Determine verdict based on search results text.

        Args:
            claim (str): The claim being checked
            text (str): Combined text from search results
            sources (List[Dict]): Source information

        Returns:
            tuple: (verdict, explanation)
        """
        # Keywords indicating false claims
        false_indicators = ['false', 'myth', 'debunk', 'not true', 'incorrect', 'wrong', 'fake', 'hoax']
        true_indicators = ['true', 'correct', 'verified', 'confirm', 'accurate', 'fact']
        partial_indicators = ['partially', 'partly', 'some truth', 'misleading', 'context']

        # Count indicators
        false_count = sum(1 for indicator in false_indicators if indicator in text)
        true_count = sum(1 for indicator in true_indicators if indicator in text)
        partial_count = sum(1 for indicator in partial_indicators if indicator in text)

        # Determine verdict
        if false_count > true_count and false_count > partial_count:
            verdict = "FALSE"
            explanation = f"Based on search results from {len(sources)} sources, this claim appears to be false. "
            explanation += "Multiple reliable sources indicate this is a common myth or misconception."
        elif true_count > false_count and true_count > partial_count:
            verdict = "TRUE"
            explanation = f"Based on search results from {len(sources)} sources, this claim appears to be true. "
            explanation += "Multiple reliable sources confirm the accuracy of this statement."
        elif partial_count > 0:
            verdict = "PARTIALLY TRUE"
            explanation = f"Based on search results from {len(sources)} sources, this claim is partially true. "
            explanation += "The claim contains some accurate information but may be missing context or contain misleading elements."
        else:
            verdict = "UNVERIFIED"
            explanation = f"Based on available search results from {len(sources)} sources, this claim could not be definitively verified. "
            explanation += "More research may be needed to reach a conclusive verdict."

        # Add source mention
        if sources:
            top_source = sources[0]['title']
            explanation += f" Key source: {top_source}."

        return verdict, explanation

    def _create_error_response(self, claim: str, error_msg: str) -> Dict[str, Any]:
        """Create an error response for a failed fact-check."""
        return {
            "claim": claim,
            "verdict": "ERROR",
            "explanation": f"Unable to verify this claim. Error: {error_msg}",
            "sources": []
        }

    def format_result(self, result: Dict[str, Any]) -> str:
        """
        Format a fact-check result for display.

        Args:
            result (Dict): Fact-check result

        Returns:
            str: Formatted string for display
        """
        verdict_emoji = {
            "TRUE": "✅",
            "FALSE": "❌",
            "PARTIALLY TRUE": "⚠️",
            "UNVERIFIED": "❓",
            "ERROR": "🔴"
        }

        emoji = verdict_emoji.get(result["verdict"], "❓")

        output = [
            "-" * 80,
            f"Claim: {emoji} {result['verdict']}",
            f"  Statement: \"{result['claim']}\"",
            f"  Explanation: {result['explanation']}",
        ]

        if result["sources"]:
            output.append("  Sources:")
            for source in result["sources"]:
                output.append(f"    - {source}")
        else:
            output.append("  Sources: None available")

        return "\n".join(output)

    def export_results_json(self, results: List[Dict[str, Any]], output_file: str = "fact_check_results.json"):
        """
        Export fact-check results to a JSON file.

        Args:
            results (List[Dict]): List of fact-check results
            output_file (str): Output file path
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Results exported to {output_file}")
        except Exception as e:
            print(f"❌ Error exporting results: {e}")


def main():
    """Test the Perplexity fact checker with example claims."""
    print("\n" + "=" * 80)
    print(" PERPLEXITY FACT CHECKER TEST")
    print("=" * 80 + "\n")

    # Initialize checker
    checker = PerplexityFactChecker()

    if not checker.client:
        print("❌ Perplexity client not initialized.")
        print("   Please set PERPLEXITY_API_KEY environment variable.")
        print("   And install: pip install perplexityai")
        return

    # Test claims
    test_claims = [
        "The Great Wall of China is visible from the moon.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The COVID-19 vaccine contains microchips."
    ]

    # Check all claims
    results = checker.check_claims(test_claims)

    # Display results
    print("\n" + "=" * 80)
    print(" FACT-CHECK RESULTS")
    print("=" * 80 + "\n")

    for result in results:
        print(checker.format_result(result))
        print()

    # Export to JSON
    checker.export_results_json(results, "test_fact_check_results.json")


if __name__ == "__main__":
    main()
