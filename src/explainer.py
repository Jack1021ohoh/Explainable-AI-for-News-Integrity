import google.generativeai as genai
import json
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import GEMINI_MODEL_NAME


class LLMExplainer:
    """Generate explanation using LLM (Gemini API)"""

    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize the LLM explainer.

        Args:
            api_key (str): Google Gemini API key
            model_name (str): Gemini model name (default from config)
        """
        self.api_key = api_key
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
        wikipedia_facts: Dict[str, List[Dict]] = None,
        fact_check_results: Dict[str, List[Dict]] = None
    ) -> Dict:
        """
        Generate comprehensive explanation using Gemini.

        Args:
            title (str): Article title
            text (str): Article text
            classification (str): Classification result ("FAKE" or "REAL")
            confidence (float): Confidence score
            claims (List[str]): Extracted claims (optional)
            wikipedia_facts (Dict): Wikipedia verification results (optional)
            fact_check_results (Dict): Fact check API results (optional)

        Returns:
            Dict: Explanation with display_status, explanation, and key_flags
        """
        # Convert classification to detector label format (0 for FAKE, 1 for REAL)
        detector_label = 0 if classification == "FAKE" else 1

        if not self.model:
            return self._generate_simple_explanation(
                classification, confidence, claims, wikipedia_facts, fact_check_results
            )

        try:
            # Use the Gemini-based explanation prompt
            explanation_json = self._explain_with_gemini(text, detector_label)
            return explanation_json

        except Exception as e:
            print(f"Error generating explanation with Gemini: {e}")
            return self._generate_simple_explanation(
                classification, confidence, claims, wikipedia_facts, fact_check_results
            )

    def _explain_with_gemini(self, article_text: str, detector_label: int) -> Dict:
        """
        Generate explanation using Gemini API with structured JSON output.

        Args:
            article_text (str): The news article text
            detector_label (int): 0 for FAKE, 1 for REAL

        Returns:
            Dict: Structured explanation
        """
        detector_label_str = "FAKE" if detector_label == 0 else "REAL"

        prompt = f"""
Role: You are a professional Fact-Checking Assistant helping a user understand news credibility.

Task: Analyze the news article below. The system has flagged it as "{detector_label_str}". You can refer to it, but you must verify this yourself.

Target Audience: A general news reader who is confused about whether this story is true.

---
ARTICLE TEXT:
\"\"\"
{article_text}
\"\"\"
---

INSTRUCTIONS:
1. Determine the final credibility status (e.g., specific misinformation, satire, verified news, or opinion).
2. Write a clear, helpful explanation addressing the content directly.
3. Output strictly in this JSON format (no markdown code blocks):
{{
    "display_status": "Brief Headline (e.g., 'Misinformation Detected' or 'Verified News')",
    "explanation": "2-3 sentences explaining WHY. Be specific. Mention if the sources are missing, the tone is sensational, or if the facts contradict known reality.",
    "key_flags": ["Bullet point 1 (e.g., No sources cited)", "Bullet point 2 (e.g., Emotional language)"]
}}
"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            # Clean up response text (remove markdown code blocks if present)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Parse JSON response
            result = json.loads(response_text)
            return result

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return {
                "display_status": f"Analysis Complete - {detector_label_str}",
                "explanation": response.text if 'response' in locals() else "Unable to generate explanation.",
                "key_flags": ["Unable to parse structured response"]
            }
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")

    def _generate_simple_explanation(
        self,
        classification: str,
        confidence: float,
        claims: List[str] = None,
        wikipedia_facts: Dict[str, List[Dict]] = None,
        fact_check_results: Dict[str, List[Dict]] = None
    ) -> Dict:
        """
        Generate simple rule-based explanation when API is not available.

        Args:
            classification (str): "FAKE" or "REAL"
            confidence (float): Confidence score
            claims (List[str]): Extracted claims (optional)
            wikipedia_facts (Dict): Wikipedia verification results (optional)
            fact_check_results (Dict): Fact check API results (optional)

        Returns:
            Dict: Simple explanation
        """
        claims = claims or []
        wikipedia_facts = wikipedia_facts or {}
        fact_check_results = fact_check_results or {}

        # Build key flags
        key_flags = []
        if classification == "FAKE":
            key_flags = [
                "Low consistency with verified sources",
                "Claims lack corroboration from reliable fact-checkers",
                "Language patterns typical of misinformation"
            ]
            recommendation = "Exercise caution with this article. Cross-reference claims with established news sources before sharing."
        else:
            key_flags = [
                "Claims align well with verified information",
                "Consistent with fact-checked sources",
                "Language and presentation appear credible"
            ]
            recommendation = "This article appears legitimate, but always verify important claims independently."

        explanation_text = f"The article has been classified as {classification} with a confidence level of {confidence:.1%}. {recommendation}"

        return {
            "display_status": f"{classification} News Detected",
            "explanation": explanation_text,
            "key_flags": key_flags
        }


if __name__ == "__main__":
    # Test the explainer
    try:
        explainer = LLMExplainer()
        test_text = "This is a sample news article for testing purposes."
        result = explainer.generate_explanation(
            title="Test Article",
            text=test_text,
            classification="FAKE",
            confidence=0.85
        )
        print("\nExplanation Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
