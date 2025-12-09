import os
import json
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Claim:
    """
    Represents an extracted claim from a news article.
    
    Attributes:
        text: The claim statement itself.
        claim_type: Category of claim (statistical, event, quote, causal, other).
        checkable: Whether this claim can be fact-checked.
        confidence: Model's confidence in extraction quality (0.0-1.0).
        source_sentence: Original sentence from which claim was extracted.
    """
    text: str
    claim_type: str
    checkable: bool
    confidence: float
    source_sentence: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

class ClaimExtractor:
    """
    Extracts verifiable claims from news articles using Llama 3 via Groq API.
    
    This extractor identifies factual statements that can be verified against
    external sources, filtering out opinions and subjective statements.
    """
    
    # System prompt for claim extraction
    SYSTEM_PROMPT = """You are a precise claim extraction system for fact-checking.
    Your task is to extract VERIFIABLE FACTUAL CLAIMS from news articles.

    ## What counts as a checkable claim:
    - Statistical facts (numbers, percentages, rankings)
    - Historical events (dates, occurrences, outcomes)  
    - Direct quotes attributed to specific people
    - Causal relationships presented as facts
    - Scientific or technical assertions

    ## What to EXCLUDE:
    - Opinions or subjective statements ("X is the best...")
    - Predictions or future events ("X will happen...")
    - Vague statements without specific details
    - Common knowledge that doesn't need verification

    ## Output Format:
    Return a JSON array of claims. Each claim object must have:
    {
        "text": "The exact claim statement, rephrased for clarity if needed",
        "claim_type": "statistical|event|quote|causal|other",
        "checkable": true/false,
        "confidence": 0.0-1.0,
        "source_sentence": "Original sentence from the article"
    }

    Only return the JSON array, no other text."""

    USER_PROMPT_TEMPLATE = """Extract all verifiable factual claims from this news article:
    ---
    {article_text}
    ---
    Return only checkable claims as a JSON object with a "claims" array."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "llama-3.1-8b-instant"):
        """
        Initialize the ClaimExtractor with Groq API key.
        
        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
            model_name: Groq model to use. Options:
                - "llama-3.1-8b-instant" (fast, good for most cases)
                - "llama-3.1-70b-versatile" (better quality, slower)
                - "llama-3.3-70b-versatile" (latest, best quality)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter. Get free key at: https://console.groq.com/"
            )

        self.model_name = model_name
        self.client = Groq(api_key=self.api_key)
        print(f"ClaimExtractor initialized with model: {model_name}")

    def extract(self, article_text: str, max_claims: int = 10, min_confidence: float = 0.5) -> List[Claim]:
        """
        Extract verifiable claims from a news article.
        
        Args:
            article_text: The full text of the news article.
            max_claims: Maximum number of claims to return.
            min_confidence: Minimum confidence threshold for claims.
            
        Returns:
            List[Claim]: Extracted claims sorted by confidence (descending).
        """
        if not article_text or not article_text.strip():
            return []
        
        truncated_text = article_text[:8000] if len(article_text) > 8000 else article_text

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(
                        article_text=truncated_text
                    )}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            raw_content = response.choices[0].message.content
            claims = self._parse_response(raw_content)

            filtered_claims = [
                claim for claim in claims
                if claim.checkable and claim.confidence >= min_confidence
            ]
            filtered_claims.sort(key=lambda c: c.confidence, reverse=True)

            return filtered_claims[:max_claims]
        
        except Exception as e:
            print(f"Error extracting claims: {e}")
            return []
        
    def _parse_response(self, raw_content: str) -> List[Claim]:
        """Parse LLM response into Claim objects."""
        try:
            data = json.loads(raw_content)
            
            if isinstance(data, list):
                claims_data = data
            elif isinstance(data, dict):
                claims_data = data.get("claims", data.get("results", []))
                if not isinstance(claims_data, list):
                    claims_data = [data]
            else:
                return []
            
            claims = []
            for item in claims_data:
                if not isinstance(item, dict):
                    continue

                try:
                    claim = Claim(
                        text=str(item.get("text", "")).strip(),
                        claim_type=str(item.get("claim_type", "other")).lower(),
                        checkable=bool(item.get("checkable", True)),
                        confidence=float(item.get("confidence", 0.5)),
                        source_sentence=str(item.get("source_sentence", "")).strip()
                    )

                    if claim.text:
                        claims.append(claim)
                except (ValueError, TypeError) as e:
                    print(f"Skipping malformed claim: {e}")
                    continue
            return claims
        
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_content)

            if json_match:
                return self._parse_response(json_match.group(1))
            return []
        
    def extract_batch(self, articles: List[str], max_claims_per_article: int = 10, max_workers: int = 10) -> List[List[Claim]]:
        """Extract claims from multiple articles."""
        results = [None] * len(articles)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.extract, article, max_claims_per_article): i 
                for i, article in enumerate(articles)
        }

            for future in tqdm(as_completed(future_to_index), total=len(articles), desc="Parallel Extraction"):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"Error extracting article {index}: {e}")
                    results[index] = []

        return results
    

if __name__ == "__main__":
    TEST_ARTICLE = """
    Tesla reported record quarterly revenue of $25.5 billion in Q3 2024, 
    representing a 7% increase from the same period last year. CEO Elon Musk 
    stated during the earnings call that "we expect to deliver 2 million 
    vehicles this year." 
    
    The company's stock rose 12% following the announcement, making it the 
    best single-day gain since January 2023. Analysts at Morgan Stanley 
    upgraded their price target from $250 to $310, citing strong demand 
    in China where Tesla sold 150,000 vehicles in September alone.
    
    However, some investors remain skeptical about Tesla's ability to 
    maintain growth amid increasing competition from BYD and other Chinese 
    manufacturers. The electric vehicle market is expected to grow 
    significantly in the coming years.
    """

    try:
        extractor = ClaimExtractor()
        
        print("\n" + "="*60)
        print("SIMPLE CLAIM EXTRACTOR TEST")
        print("="*60)
        
        claims = extractor.extract(TEST_ARTICLE)
        
        print(f"\nExtracted {len(claims)} verifiable claims:\n")
        
        for i, claim in enumerate(claims, 1):
            print(f"[{i}] {claim.text}")
            print(f"    Type: {claim.claim_type} | Confidence: {claim.confidence:.2f}")
            print(f"    Source: {claim.source_sentence[:80]}...")
            print()
            
    except ValueError as e:
        print(f"\nSetup Error: {e}")
        print("\nTo test this module:")
        print("1. Get free API key at: https://console.groq.com/")
        print("2. Add to .env: GROQ_API_KEY=your-key-here")
        print("3. Run: uv run python src/extractor.py")