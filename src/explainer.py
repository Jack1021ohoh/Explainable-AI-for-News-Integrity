"""
LLM Explainer for News Integrity Analysis

This module generates human-readable explanations for news classification results,
incorporating extracted claims, Wikipedia evidence, and fact-check results.
"""

import google.generativeai as genai
import json
from typing import List, Dict, Optional, Any
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import GEMINI_MODEL_NAME, GEMINI_API_KEY


class LLMExplainer:
    """
    Generate comprehensive explanations using LLM (Gemini API).
    
    This explainer takes classification results along with supporting evidence
    (claims, Wikipedia facts, fact-check results) to produce detailed,
    explainable verdicts for news articles.
    """

    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize the LLM explainer.

        Args:
            api_key (str): Google Gemini API key
            model_name (str): Gemini model name (default from config)
        """
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or GEMINI_MODEL_NAME
        self.model = None

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                print("LLMExplainer initialized successfully with Gemini API.")
            except Exception as e:
                print(f"Warning: Failed to initialize Gemini model: {e}")
                print("Falling back to simple explanations.")
        else:
            print("LLMExplainer initialized without API key. Using simple explanations.")

    def generate_explanation(
        self,
        title: str,
        text: str,
        classification: str,
        confidence: float,
        claims: List[str] = None,
        wikipedia_evidence: Dict[str, List[Dict]] = None,
        fact_check_results: Dict[str, List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation using all available evidence.

        Args:
            title: Article title.
            text: Article text.
            classification: Classification result ("FAKE" or "REAL").
            confidence: Confidence score (0-1).
            claims: List of extracted claims from the article.
            wikipedia_evidence: Dict mapping claims to Wikipedia evidence.
            fact_check_results: Dict mapping claims to fact-check results.

        Returns:
            Dict containing:
                - display_status: Brief headline
                - explanation: Detailed explanation
                - key_flags: List of key indicators
                - claim_analysis: Per-claim analysis (if claims provided)
        """
        claims = claims or []
        wikipedia_evidence = wikipedia_evidence or {}
        fact_check_results = fact_check_results or {}
        # # Convert classification to detector label format (0 for FAKE, 1 for REAL)
        # detector_label = 0 if classification == "FAKE" else 1

        if not self.model:
            return self._generate_simple_explanation(
                classification, confidence, claims, wikipedia_evidence, fact_check_results
            )
        try:
            result = self._explain_with_evidence(title, text, classification, confidence, claims, wikipedia_evidence, fact_check_results)
            return result
        
        except Exception as e:
            print(f"Error generating explanation with Gemini: {e}")
            return self._generate_simple_explanation(
                classification, confidence, claims,
                wikipedia_evidence, fact_check_results
            )

    def _explain_with_evidence(
        self,
        title: str,
        text: str,
        classification: str,
        confidence: float,
        claims: List[str],
        wikipedia_evidence: Dict[str, List[Dict]],
        fact_check_results: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """Generate explanation using Gemini with all available evidence."""
        evidence_section = self._format_evidence_for_prompt(
            claims, wikipedia_evidence, fact_check_results
        )

        prompt = f"""You are a professional Fact-Checking Assistant helping users understand news credibility.

    ## Task
    Analyze the news article below and provide a clear, educational explanation of its credibility.

    ## Article Information
    **Title:** {title}
    **Classification:** {classification} (Confidence: {confidence:.1%})

    **Article Text:**
    {text[:2000]}{"..." if len(text) > 2000 else ""}

    ## Evidence Gathered
    {evidence_section}

    ## Instructions
    Based on the classification result and the evidence gathered:

    1. **Evaluate** the article's credibility considering:
    - The AI classifier's assessment
    - Whether extracted claims are supported by Wikipedia
    - Whether any claims have been fact-checked by independent organizations

    2. **Explain** your reasoning in a way that helps the reader understand:
    - Why this article might be classified as {classification}
    - Which specific claims are problematic or well-supported
    - What sources support or contradict the claims

    3. **Provide** actionable guidance for the reader.

    ## Output Format
    Respond in this exact JSON format (no markdown code blocks):
    {{
        "display_status": "Brief 2-5 word headline (e.g., 'Likely Misinformation' or 'Appears Credible')",
        "explanation": "2-4 sentences explaining the overall assessment. Be specific about which claims are problematic and why. Reference the evidence when possible.",
        "key_flags": [
            "Specific indicator 1 (e.g., 'Claim about X contradicts Wikipedia source')",
            "Specific indicator 2 (e.g., 'No fact-checks found for main claims')",
            "Specific indicator 3 (e.g., 'Statistics not verifiable')"
        ],
        "claim_analysis": [
            {{
                "claim": "The specific claim text",
                "status": "supported / contradicted / unverified / partially_true",
                "evidence_summary": "Brief explanation of what evidence shows"
            }}
        ]
    }}
    """

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            # Clean up response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            result = json.loads(response_text)
            return result

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return {
                "display_status": f"Analysis Complete - {classification}",
                "explanation": response.text if 'response' in locals() else "Unable to generate explanation.",
                "key_flags": ["Analysis completed but response parsing failed"],
                "claim_analysis": []
            }
        
    def _format_evidence_for_prompt(
        self,
        claims: List[str],
        wikipedia_evidence: Dict[str, List[Dict]],
        fact_check_results: Dict[str, List[Dict]]
    ) -> str:
        """Format all evidence into a readable section for the prompt."""
        
        if not claims:
            return "No claims were extracted from this article."
        
        sections = []
        sections.append(f"**Extracted Claims:** {len(claims)} claims found\n")
        
        for i, claim in enumerate(claims, 1):
            claim_section = f"### Claim {i}: \"{claim}\"\n"
            
            # Wikipedia evidence
            wiki_evidence = wikipedia_evidence.get(claim, [])
            if wiki_evidence:
                claim_section += "**Wikipedia Evidence:**\n"
                for j, ev in enumerate(wiki_evidence[:2], 1):  # Limit to 2 per claim
                    source = ev.get('source', 'Unknown')
                    text = ev.get('text', '')[:200]
                    claim_section += f"  {j}. [{source}]: {text}...\n"
            else:
                claim_section += "**Wikipedia Evidence:** No relevant articles found\n"
            
            # Fact-check results
            fc_results = fact_check_results.get(claim, [])
            if fc_results:
                claim_section += "**Fact-Check Results:**\n"
                for j, fc in enumerate(fc_results[:2], 1):  # Limit to 2 per claim
                    rating = fc.get('rating', 'Unknown')
                    publisher = fc.get('publisher', 'Unknown')
                    claim_section += f"  {j}. {publisher} rated: \"{rating}\"\n"
            else:
                claim_section += "**Fact-Check Results:** No existing fact-checks found\n"
            
            sections.append(claim_section)
        
        return "\n".join(sections)


    def _generate_simple_explanation(
        self,
        classification: str,
        confidence: float,
        claims: List[str],
        wikipedia_evidence: Dict[str, List[Dict]],
        fact_check_results: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """Generate simple rule-based explanation when API is not available."""
        
        # Analyze evidence
        claims_with_wiki = sum(1 for c in claims if wikipedia_evidence.get(c))
        claims_with_fc = sum(1 for c in claims if fact_check_results.get(c))
        total_claims = len(claims)
        
        # Build key flags based on classification and evidence
        key_flags = []
        
        if classification == "FAKE":
            key_flags.append(f"AI classifier detected patterns consistent with misinformation ({confidence:.0%} confidence)")
            if total_claims > 0 and claims_with_wiki < total_claims / 2:
                key_flags.append(f"Only {claims_with_wiki}/{total_claims} claims found supporting Wikipedia evidence")
            if claims_with_fc > 0:
                key_flags.append("Some claims have been previously fact-checked")
            else:
                key_flags.append("No existing fact-checks found for verification")
        else:
            key_flags.append(f"AI classifier found patterns consistent with credible news ({confidence:.0%} confidence)")
            if claims_with_wiki > 0:
                key_flags.append(f"{claims_with_wiki}/{total_claims} claims have relevant Wikipedia sources")
            if claims_with_fc > 0:
                key_flags.append("Some claims verified by fact-checking organizations")
        
        # Build explanation
        if classification == "FAKE":
            explanation = (
                f"This article has been classified as potentially unreliable with {confidence:.0%} confidence. "
                f"We extracted {total_claims} claims from the article. "
                f"Please verify the information with trusted sources before sharing."
            )
            display_status = "Potential Misinformation Detected"
        else:
            explanation = (
                f"This article appears to be credible based on our analysis ({confidence:.0%} confidence). "
                f"We found {total_claims} verifiable claims. "
                f"As always, cross-reference important information with multiple sources."
            )
            display_status = "Appears Credible"
        
        # Build claim analysis
        claim_analysis = []
        for claim in claims[:5]:  # Limit to 5 claims
            wiki_found = len(wikipedia_evidence.get(claim, [])) > 0
            fc_found = len(fact_check_results.get(claim, [])) > 0
            
            if wiki_found and fc_found:
                status = "verified"
                summary = "Found supporting Wikipedia evidence and fact-checks"
            elif wiki_found:
                status = "partially_verified"
                summary = "Found relevant Wikipedia content"
            elif fc_found:
                status = "fact_checked"
                summary = "Previously fact-checked by independent organizations"
            else:
                status = "unverified"
                summary = "No supporting evidence found"
            
            claim_analysis.append({
                "claim": claim,
                "status": status,
                "evidence_summary": summary
            })
        
        return {
            "display_status": display_status,
            "explanation": explanation,
            "key_flags": key_flags,
            "claim_analysis": claim_analysis
        }


if __name__ == "__main__":
    # Test the explainer
    print("\n" + "=" * 60)
    print(" LLM EXPLAINER TEST")
    print("=" * 60)
    
    explainer = LLMExplainer()
    
    # Test with mock data
    test_claims = [
        "Tesla reported record revenue of $25 billion",
        "Electric vehicles are dangerous"
    ]
    
    test_wiki_evidence = {
        "Tesla reported record revenue of $25 billion": [
            {"source": "Tesla, Inc.", "text": "Tesla is an American electric vehicle company..."}
        ],
        "Electric vehicles are dangerous": []
    }
    
    test_fc_results = {
        "Tesla reported record revenue of $25 billion": [],
        "Electric vehicles are dangerous": [
            {"rating": "False", "publisher": "Snopes", "title": "Are EVs dangerous?"}
        ]
    }
    
    result = explainer.generate_explanation(
        title="Tesla Q3 2024 Earnings Report",
        text="Tesla reported record quarterly revenue...",
        classification="REAL",
        confidence=0.85,
        claims=test_claims,
        wikipedia_evidence=test_wiki_evidence,
        fact_check_results=test_fc_results
    )
    
    print("\nExplanation Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
