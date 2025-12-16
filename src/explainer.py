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

        # Determine detector label string
        detector_label_str = "Potentially Fake" if classification == "FAKE" else "Likely Real"

        prompt = f"""Role: You are a professional Fact-Checker.
Your goal is to determine the factual accuracy of the news article below for a confused reader.

Input Data:
1. Automated Alert: **{detector_label_str}** (Confidence: {confidence:.2f}).
   (Treat this as a lead, but verify it against the evidence.)
2. Independent Evidence is provided below.

---
ARTICLE TITLE: {title}

ARTICLE TEXT:
\"\"\"
{text}
\"\"\"
---
{evidence_section}
---

INSTRUCTIONS:
1. **Analyze First**: Look at the "Claims" vs. "Independent Evidence".
2. **Determine Validity**: Does the evidence support or contradict the text?
3. **Formulate Output**: Write the response for the user.

OUTPUT FORMAT (JSON):
Strictly output a JSON object with this structure. Ensure "thought_process" is the FIRST key.
{{
    "thought_process": "Step-by-step reasoning. 1. Checked claim X against evidence Y. 2. Noted contradiction...",
    "display_status": "Short Verdict (e.g., 'False', 'Verified', 'Satire', 'Opinion')",
    "explanation": "2-3 clear sentences for the user (ignoring the thought process). Quote the evidence directly.",
    "key_flags": [
        "Bullet 1: Specific contradiction or confirmation",
        "Bullet 2: Tone analysis or other flag"
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

        # Try with JSON mode first
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            response_text = response.text.strip()
            print(f"[DEBUG] Received response from Gemini ({len(response_text)} chars)")

        except Exception as api_error:
            print(f"❌ GEMINI API ERROR: {type(api_error).__name__}: {api_error}")
            print("Attempting without JSON mode constraint...")

            # Try without JSON mode as fallback
            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                print(f"[DEBUG] Received response without JSON mode ({len(response_text)} chars)")
            except Exception as fallback_error:
                print(f"❌ GEMINI API FAILED COMPLETELY: {fallback_error}")
                return self._generate_simple_explanation(
                    classification, confidence, claims,
                    wikipedia_evidence, fact_check_results
                )

        # Extract and parse JSON from response
        try:
            # Clean potential markdown artifacts and extra text
            cleaned_text = response_text.strip()

            # Remove markdown code blocks
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]

            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]

            cleaned_text = cleaned_text.strip()

            # Try to extract JSON if there's extra text
            import re
            json_match = re.search(r'\{[\s\S]*\}', cleaned_text)
            if json_match:
                cleaned_text = json_match.group(0)

            # Parse JSON
            result = json.loads(cleaned_text)
            print(f"[DEBUG] Successfully parsed JSON. Keys: {list(result.keys())}")

            # Validate that we got the expected structure
            if 'explanation' not in result or 'key_flags' not in result:
                print(f"⚠️ Response missing expected fields. Got keys: {list(result.keys())}")
                print(f"Attempting to construct valid response from partial data...")

                # Try to salvage what we can
                result = self._fix_incomplete_response(result, classification, confidence)

            # Remove thought_process from output (it's internal reasoning)
            if 'thought_process' in result:
                del result['thought_process']

            return result

        except json.JSONDecodeError as e:
            print(f"❌ JSON PARSING ERROR: {e}")
            print(f"Raw response (first 1000 chars):")
            print(response_text[:1000] if 'response_text' in locals() else 'N/A')
            print("-" * 60)

            # Fall back to simple explanation
            return self._generate_simple_explanation(
                classification, confidence, claims,
                wikipedia_evidence, fact_check_results
            )
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
            if 'response_text' in locals():
                print(f"Response: {response_text[:500]}")
            print("-" * 60)

            return self._generate_simple_explanation(
                classification, confidence, claims,
                wikipedia_evidence, fact_check_results
            )
        
    def _fix_incomplete_response(
        self,
        partial_result: Dict[str, Any],
        classification: str,
        confidence: float
    ) -> Dict[str, Any]:
        """Attempt to fix incomplete response from Gemini."""
        fixed = {
            "display_status": partial_result.get("display_status", "Unverified"),
            "explanation": partial_result.get("explanation", f"Analysis completed with {confidence:.0%} confidence."),
            "key_flags": partial_result.get("key_flags", []),
            "claim_analysis": partial_result.get("claim_analysis", [])
        }

        # Ensure key_flags is a list
        if not isinstance(fixed["key_flags"], list):
            fixed["key_flags"] = [str(fixed["key_flags"])]

        # Ensure claim_analysis is a list
        if not isinstance(fixed["claim_analysis"], list):
            fixed["claim_analysis"] = []

        return fixed

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
                    explanation = fc.get('title', '')
                    sources = fc.get('sources', [])

                    claim_section += f"  {j}. {publisher} verdict: \"{rating}\"\n"
                    if explanation:
                        # Truncate long explanations
                        expl_short = explanation[:200] + "..." if len(explanation) > 200 else explanation
                        claim_section += f"     Explanation: {expl_short}\n"
                    if sources:
                        claim_section += f"     Sources: {', '.join(sources[:3])}\n"
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
