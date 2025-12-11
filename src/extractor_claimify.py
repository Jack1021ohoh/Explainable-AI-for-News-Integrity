"""
Claimify-based Claim Extractor Module

Based on the paper "Towards Effective Extraction and Evaluation of Factual Claims"
by Metropolitansky & Larson (2025).

This implementation follows the three-stage pipeline:
1. Selection: Filter sentences to retain only verifiable content
2. Disambiguation: Identify and resolve ambiguity, or mark as unresolvable  
3. Decomposition: Extract decontextualized factual claims

Reference: arXiv:2502.10855v2
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from dotenv import load_dotenv
from groq import Groq
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

load_dotenv()

class SentenceStatus(Enum):
    """Status of a sentence after processing through Claimify stages."""
    PENDING = "pending"
    NO_VERIFIABLE_CLAIMS = "no_verifiable_claims"
    CANNOT_BE_DISAMBIGUATED = "cannot_be_disambiguated"
    PROCESSED = "processed"

@dataclass
class ClaimifySentence:
    """Represents a sentence being processed through the Claimify pipeline."""
    original: str
    index: int
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    
    # Stage outputs
    selection_output: Optional[str] = None
    disambiguation_output: Optional[str] = None
    
    # Status tracking
    status: SentenceStatus = SentenceStatus.PENDING
    
    # Final claims
    claims: List[str] = field(default_factory=list)


@dataclass
class ClaimifyResult:
    """Result from the Claimify extraction process."""
    claims: List[str]
    claim_details: List[Dict[str, Any]]
    sentences_total: int
    sentences_processed: int
    sentences_with_claims: int
    sentences_no_verifiable: int
    sentences_ambiguous: int
    sentences_filtered: int


    
class ClaimifyExtractor:
    """
    Extracts verifiable claims using the Claimify three-stage pipeline.
    
    Based on: "Towards Effective Extraction and Evaluation of Factual Claims"
    (arXiv:2502.10855v2)
    
    Stages:
        1. Selection: Identify sentences with verifiable content
        2. Disambiguation: Resolve referential and structural ambiguity
        3. Decomposition: Extract decontextualized factual claims
    """
    
    # ============================================================
    # STAGE 1: SELECTION PROMPTS
    # ============================================================
    
    SELECTION_SYSTEM_PROMPT = """You are an assistant to a fact-checker. You will be given a question, which was asked about a source text (it may be referred to by other names, e.g., a dataset). You will also be given an excerpt from a response to the question. If it contains "[...]", this means that you are NOT seeing all sentences in the response. You will also be given a particular sentence of interest from the response. Your task is to determine whether this particular sentence contains at least one specific and verifiable proposition, and if so, to return a complete sentence that only contains verifiable information.

    Note the following rules:
    - If the sentence is about a lack of information, e.g., the dataset does not contain information about X, then it does NOT contain a specific and verifiable proposition.
    - It does NOT matter whether the proposition is true or false.
    - It does NOT matter whether the proposition is relevant to the question.
    - It does NOT matter whether the proposition contains ambiguous terms, e.g., a pronoun without a clear antecedent. Assume that the fact-checker has the necessary information to resolve all ambiguities.
    - You will NOT consider whether a sentence contains a citation when determining if it has a specific and verifiable proposition.

    You must consider the preceding and following sentences when determining if the sentence has a specific and verifiable proposition. For example:
    - if preceding sentence = "Who is the CEO of Company X?" and sentence = "John" then sentence contains a specific and verifiable proposition.
    - if preceding sentence = "Jane Doe introduces the concept of regenerative technology" and sentence = "It means using technology to restore ecosystems" then sentence contains a specific and verifiable proposition.
    - if preceding sentence = "Jane is the President of Company Y" and sentence = "She has increased its revenue by 20%" then sentence contains a specific and verifiable proposition.
    - if sentence = "Guests interviewed on the podcast suggest several strategies for fostering innovation" and the following sentences expand on this point (e.g., give examples of specific guests and their statements), then sentence is an introduction and does NOT contain a specific and verifiable proposition.
    - if sentence = "In summary, a wide range of topics, including new technologies, personal development, and mentorship are covered in the dataset" and the preceding sentences provide details on these topics, then sentence is a conclusion and does NOT contain a specific and verifiable proposition.

    Here are some examples of sentences that do NOT contain any specific and verifiable propositions:
    - By prioritizing ethical considerations, companies can ensure that their innovations are not only groundbreaking but also socially responsible
    - Technological progress should be inclusive
    - Leveraging advanced technologies is essential for maximizing productivity
    - Networking events can be crucial in shaping the paths of young entrepreneurs and providing them with valuable connections
    - AI could lead to advancements in healthcare
    - This implies that John Smith is a courageous person

    Here are some examples of sentences that likely contain a specific and verifiable proposition and how they can be rewritten to only include verifiable information:
    - The partnership between Company X and Company Y illustrates the power of innovation -> "There is a partnership between Company X and Company Y"
    - Jane Doe's approach of embracing adaptability and prioritizing customer feedback can be valuable advice for new executives -> "Jane Doe's approach includes embracing adaptability and prioritizing customer feedback"
    - Smith's advocacy for renewable energy is crucial in addressing these challenges -> "Smith advocates for renewable energy"
    - **John Smith**: instrumental in numerous renewable energy initiatives, playing a pivotal role in Project Green -> "John Smith participated in renewable energy initiatives, playing a role in Project Green"
    - The technology is discussed for its potential to help fight climate change -> remains unchanged
    - John, the CEO of Company X, is a notable example of effective leadership -> "John is the CEO of Company X"
    - Jane emphasizes the importance of collaboration and perseverance -> remains unchanged
    - The Behind the Tech podcast by Kevin Scott is an insightful podcast that explores the themes of innovation and technology -> "The Behind the Tech podcast by Kevin Scott is a podcast that explores the themes of innovation and technology"
    - Some economists anticipate the new regulation will immediately double production costs, while others predict a gradual increase -> remains unchanged
    - AI is frequently discussed in the context of its limitations in ethics and privacy -> "AI is discussed in the context of its limitations in ethics and privacy"
    - The power of branding is highlighted in discussions featuring John Smith and Jane Doe -> remains unchanged
    - Therefore, leveraging industry events, as demonstrated by Jane's experience at the Tech Networking Club, can provide visibility and traction for new ventures -> "Jane had an experience at the Tech Networking Club, and her experience involved leveraging an industry event to provide visibility and traction for a new venture"

    Your output must be a JSON object with the following structure:
    {
        "reasoning": "Your step-by-step reasoning about whether the sentence contains verifiable content",
        "contains_verifiable": true or false,
        "modified_sentence": "The sentence with only verifiable information, or null if no verifiable content"
    }"""

    SELECTION_USER_TEMPLATE = """Question:
    {question}

    Excerpt:
    {excerpt}

    Sentence:
    {sentence}"""

    # ============================================================
    # STAGE 2: DISAMBIGUATION PROMPTS
    # ============================================================
    
    DISAMBIGUATION_SYSTEM_PROMPT = """You are an assistant to a fact-checker. You will be given a question, which was asked about a source text. You will also be given an excerpt from a response to the question. You will also be given a particular sentence from the response. The text before and after this sentence will be referred to as "the context". Your task is to "decontextualize" the sentence, which means:
    1. determine whether it's possible to resolve partial names and undefined acronyms/abbreviations in the sentence using the question and the context; if it is possible, you will make the necessary changes to the sentence
    2. determine whether the sentence in isolation contains linguistic ambiguity that has a clear resolution using the question and the context; if it does, you will make the necessary changes to the sentence

    Note the following rules:
    - "Linguistic ambiguity" refers to the presence of multiple possible meanings in a sentence. Vagueness and generality are NOT linguistic ambiguity. Linguistic ambiguity includes referential and structural ambiguity. Temporal ambiguity is a type of referential ambiguity.
    - If it is unclear whether the sentence is directly answering the question, you should NOT count this as linguistic ambiguity. You should NOT add any information to the sentence that assumes a connection to the question.
    - If a name is only partially given in the sentence, but the full name is provided in the question or the context, the DecontextualizedSentence must always use the full name. The same rule applies to definitions for acronyms and abbreviations. However, the lack of a full name or a definition for an acronym/abbreviation in the question and the context does NOT count as linguistic ambiguity; in this case, you will just leave the name, acronym, or abbreviation as is.
    - Do NOT include any citations in the DecontextualizedSentence.
    - Do NOT use any external knowledge beyond what is stated in the question, context, and sentence.

    Types of ambiguity to identify:
    1. Referential ambiguity: when it is unclear what a word or phrase refers to (e.g., "They", "the policy", "next year")
    2. Structural ambiguity: when grammatical structure allows for multiple interpretations

    The standard for resolution is whether a group of readers would likely agree on the correct interpretation.

    Your output must be a JSON object with the following structure:
    {
        "reasoning": "Your analysis of ambiguities and whether they can be resolved",
        "has_unresolvable_ambiguity": true or false,
        "ambiguity_details": "Description of any unresolvable ambiguities, or null",
        "decontextualized_sentence": "The clarified sentence, or null if cannot be disambiguated"
    }"""

    DISAMBIGUATION_USER_TEMPLATE = """Question:
    {question}

    Excerpt:
    {excerpt}

    Sentence:
    {sentence}"""

    # ============================================================
    # STAGE 3: DECOMPOSITION PROMPTS
    # ============================================================
    
    DECOMPOSITION_SYSTEM_PROMPT = """You are an assistant for a group of fact-checkers. You will be given a question, which was asked about a source text. You will also be given an excerpt from a response to the question. You will also be given a particular sentence from the response. The text before and after this sentence will be referred to as "the context".

    Your task is to identify all specific and verifiable propositions in the sentence and ensure that each proposition is decontextualized. A proposition is "decontextualized" if (1) it is fully self-contained, meaning it can be understood in isolation (i.e., without the question, the context, and the other propositions), AND (2) its meaning in isolation matches its meaning when interpreted alongside the question, the context, and the other propositions. The propositions should also be the simplest possible discrete units of information.

    Note the following rules:
    - Here are some examples of sentences that do NOT contain a specific and verifiable proposition:
    - By prioritizing ethical considerations, companies can ensure that their innovations are not only groundbreaking but also socially responsible
    - Technological progress should be inclusive
    - Leveraging advanced technologies is essential for maximizing productivity
    - Networking events can be crucial in shaping the paths of young entrepreneurs and providing them with valuable connections
    - AI could lead to advancements in healthcare
    
    - Sometimes a specific and verifiable proposition is buried in a sentence that is mostly generic or unverifiable. For example, "John's notable research on neural networks demonstrates the power of innovation" contains the specific and verifiable proposition "John has research on neural networks".

    - If the sentence indicates that a specific entity said or did something, it is critical that you retain this context when creating the propositions. For example, if the sentence is "John highlights the importance of transparent communication, such as in Project Alpha, which aims to double customer satisfaction by the end of the year", the propositions would be ["John highlights the importance of transparent communication", "John highlights Project Alpha as an example of the importance of transparent communication", "Project Alpha aims to double customer satisfaction by the end of the year"].

    - Do NOT include any citations in the propositions.
    - Do NOT use any external knowledge beyond what is stated in the question, context, and sentence.

    Extracted claims may include text in brackets [...], which typically represents information implied by the question or context but not explicitly stated in the source sentence. For example, given the question "Provide an overview of celebrities' stances on the Middle East," and the sentence "John has called for peace," you may return the claim "John [a celebrity] has called for peace [in the Middle East]."

    Your output must be a JSON object with the following structure:
    {
        "reasoning": "Your analysis of the propositions in the sentence",
        "claims": ["claim 1", "claim 2", ...]
    }

    If no verifiable claims can be extracted, return an empty claims array."""

    DECOMPOSITION_USER_TEMPLATE = """Question:
    {question}

    Excerpt:
    {excerpt}

    Sentence:
    {sentence}"""

    SKIP_PATTERNS = [
        r'^(However,|Therefore,|In summary,|Overall,|Thus,|In conclusion,)',
        r'^(Here are|The following|Below are|Let me)', 
        r'(should|could|would|might|may)\s+(be|have|lead|help|improve)', 
        r'^(I think|In my opinion|It seems|It appears)',
        r'(is expected to|are expected to|will likely|is likely to)',
        r'^(Note that|Please note|Keep in mind)',
        r'(important to|essential to|crucial to|necessary to)\s+(note|remember|consider)',  
    ]
    
    PRIORITY_PATTERNS = [
        (r'\d+(\.\d+)?%', 3),           
        (r'\$[\d,]+(\.\d+)?', 3),       
        (r'\b(19|20)\d{2}\b', 2),       
        (r'\b[A-Z][a-z]+\s[A-Z][a-z]+', 2),  
        (r'\b\d+\s*(million|billion|thousand)\b', 3),  
        (r'(said|stated|announced|reported|according to)', 2),  
        (r'(CEO|CFO|CTO|President|Director|Chairman)', 2),  
        (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', 1), 
        (r'(increased|decreased|rose|fell|grew|dropped)\s+(\d|by)', 2),  
        (r'(Q[1-4]|first quarter|second quarter|third quarter|fourth quarter)', 2),  
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.1-8b-instant",
        max_preceding_sentences: int = 5,
        max_following_sentences: int = 5,
        temperature: float = 0.1,
        completions: int = 1,
        min_successes: int = 1
    ):
        """
        Initialize the ClaimifyExtractor.
        
        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
            model_name: Groq model to use.
            max_preceding_sentences: Number of preceding sentences for context.
            max_following_sentences: Number of following sentences for context.
            temperature: Temperature for LLM calls.
            completions: Number of completions per stage (for voting).
            min_successes: Minimum successful completions to proceed.
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter. Get free key at: https://console.groq.com/"
            )
        
        self.model_name = model_name
        self.client = Groq(api_key=self.api_key)
        
        # Context configuration
        self.max_preceding = max_preceding_sentences
        self.max_following = max_following_sentences
        
        # LLM configuration
        self.temperature = temperature
        self.completions = completions
        self.min_successes = min_successes
        
        print(f"ClaimifyExtractor initialized with model: {model_name}")
        print(f"Context window: {max_preceding_sentences} preceding, {max_following_sentences} following sentences")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        paragraphs = text.split('\n')
        sentences = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_sentences = sent_tokenize(para)
            sentences.extend([s.strip() for s in para_sentences if s.strip()])
        
        return sentences
    
    def _prefilter_sentences(
            self,
            sentences: List[str],
            max_sentences: int
    ) -> List[Tuple[int, str]]:
        """
        Filter sentences that might contain verifiable claims using simple heuristics.
        """
        scored_sentences = []
        for idx, sent in enumerate(sentences):
            if len(sent.split()) < 5:
                continue
            
            # 檢查是否符合跳過模式
            should_skip = False
            for pattern in self.SKIP_PATTERNS:
                if re.search(pattern, sent, re.IGNORECASE):
                    should_skip = True
                    break
            
            if should_skip:
                continue
            
            score = 0
            for pattern, weight in self.PRIORITY_PATTERNS:
                if re.search(pattern, sent, re.IGNORECASE):
                    score += weight
            
            word_count = len(sent.split())
            if 10 <= word_count <= 40:
                score += 1
            
            scored_sentences.append((idx, sent, score))
        
        scored_sentences.sort(key=lambda x: (-x[2], x[0]))
        
        selected = scored_sentences[:max_sentences]
        selected.sort(key=lambda x: x[0])
        
        return [(idx, sent) for idx, sent, _ in selected]

    def _create_excerpt(self, sentences: List[str], current_index: int, preceding: int, following: int) -> str:
        """Create an excerpt with context around the current sentence."""
        start_idx = max(0, current_index - preceding)
        end_idx = min(len(sentences), current_index + following + 1)

        excerpt_parts = []

        # Add "[...]" if we're not starting from the beginning
        if start_idx > 0:
            excerpt_parts.append("[...]")
        
        # Add sentences
        excerpt_parts.extend(sentences[start_idx:end_idx])
        
        # Add "[...]" if we're not ending at the end
        if end_idx < len(sentences):
            excerpt_parts.append("[...]")
        
        return " ".join(excerpt_parts)
    
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        expect_json: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Make an LLM call and parse the response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000,
                response_format={"type": "json_object"} if expect_json else None
            )
            
            content = response.choices[0].message.content
            
            if expect_json:
                return json.loads(content)
            return {"content": content}
        
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            # Try to extract JSON from the response
            try:
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
            return None
        except Exception as e:
            print(f"LLM call error: {e}")
            return None
        
    def _stage_selection(
        self,
        sentence: str,
        excerpt: str,
        question: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Stage 1: Selection - Determine if sentence contains verifiable content.
        
        Returns:
            Tuple of (contains_verifiable, modified_sentence)
        """
        user_prompt = self.SELECTION_USER_TEMPLATE.format(
            question=question,
            excerpt=excerpt,
            sentence=sentence
        )
        
        result = self._call_llm(self.SELECTION_SYSTEM_PROMPT, user_prompt)
        
        if result is None:
            return False, None
        
        contains_verifiable = result.get("contains_verifiable", False)
        modified_sentence = result.get("modified_sentence")
        
        return contains_verifiable, modified_sentence
    
    def _stage_disambiguation(
        self,
        sentence: str,
        excerpt: str,
        question: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Stage 2: Disambiguation - Resolve ambiguity in the sentence.
        
        Returns:
            Tuple of (can_be_disambiguated, decontextualized_sentence)
        """
        user_prompt = self.DISAMBIGUATION_USER_TEMPLATE.format(
            question=question,
            excerpt=excerpt,
            sentence=sentence
        )
        
        result = self._call_llm(self.DISAMBIGUATION_SYSTEM_PROMPT, user_prompt)
        
        if result is None:
            return False, None
        
        has_unresolvable = result.get("has_unresolvable_ambiguity", True)
        decontextualized = result.get("decontextualized_sentence")
        
        return not has_unresolvable, decontextualized
    
    def _stage_decomposition(
        self,
        sentence: str,
        excerpt: str,
        question: str
    ) -> List[str]:
        """
        Stage 3: Decomposition - Extract factual claims from the sentence.
        
        Returns:
            List of extracted claims
        """
        user_prompt = self.DECOMPOSITION_USER_TEMPLATE.format(
            question=question,
            excerpt=excerpt,
            sentence=sentence
        )
        
        result = self._call_llm(self.DECOMPOSITION_SYSTEM_PROMPT, user_prompt)
        
        if result is None:
            return []
        
        claims = result.get("claims", [])
        
        # Ensure we have a list of strings
        if isinstance(claims, list):
            return [str(c) for c in claims if c]
        
        return []
    
    def _process_single_sentence(
        self,
        original_idx: int,
        sentence: str,
        all_sentences: List[str],
        question: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single sentence through all three Claimify stages.
        Returns:
            Dictionary with processing results.
        """
        result = {
            "original_idx": original_idx,
            "sentence": sentence,
            "status": None,
            "claims": [],
            "modified_sentence": None,
            "decontextualized_sentence": None
        }

        excerpt = self._create_excerpt(
            all_sentences, original_idx,
            self.max_preceding,
            self.max_following
        )

        # Stage 1: Selection
        contains_verifiable, modified_sentence = self._stage_selection(
            sentence, excerpt, question
        )
        
        if not contains_verifiable:
            result["status"] = "no_verifiable"
            if verbose:
                print(f"  -> No verifiable content")
            return result
        
        working_sentence = modified_sentence or sentence
        result["modified_sentence"] = modified_sentence
        
        if verbose:
            print(f"  -> Selection passed: {working_sentence[:60]}...")
        
        # Stage 2: Disambiguation
        can_disambiguate, decontextualized = self._stage_disambiguation(
            working_sentence, excerpt, question
        )
        
        if not can_disambiguate:
            result["status"] = "ambiguous"
            if verbose:
                print(f"  -> Cannot be disambiguated")
            return result
        
        final_sentence = decontextualized or working_sentence
        result["decontextualized_sentence"] = decontextualized
        
        if verbose:
            print(f"  -> Disambiguation passed: {final_sentence[:60]}...")
        
        # Stage 3: Decomposition
        claims = self._stage_decomposition(final_sentence, excerpt, question)
        
        if claims:
            result["status"] = "success"
            result["claims"] = claims
            if verbose:
                print(f"  -> Extracted {len(claims)} claims")
                for c in claims:
                    print(f"     • {c}")
        else:
            result["status"] = "no_claims"
            if verbose:
                print(f"  -> No claims extracted in decomposition")
        
        return result

    def extract(
        self,
        article_text: str,
        question: Optional[str] = None,
        max_claims: int = 50,
        max_sentences: int = 10,
        use_prefilter: bool = True,
        max_workers: int = 5,
        verbose: bool = False
    ) -> ClaimifyResult:
        """
        Extract verifiable claims from text using the Claimify pipeline.
        
        Args:
            article_text: The text to extract claims from.
            question: Optional question that the text answers.
            max_claims: Maximum number of claims to return.
            max_sentences: Maximum number of sentences to process.
            use_prefilter: Whether to use smart prefiltering (recommended).
            max_workers: Number of parallel workers for processing.
            verbose: Whether to print progress information.
            
        Returns:
            ClaimifyResult with extracted claims and statistics.
        """
        if not article_text or not article_text.strip():
            return ClaimifyResult(
                claims=[],
                claim_details=[],
                sentences_total=0,
                sentences_processed=0,
                sentences_with_claims=0,
                sentences_no_verifiable=0,
                sentences_ambiguous=0,
                sentences_filtered=0
            )
        
        # Default question if not provided
        if question is None:
            question = "What information does this text contain?"
        
        # Split into sentences
        all_sentences = self._split_into_sentences(article_text)
        total_sentences = len(all_sentences)
        
        if verbose:
            print(f"Total sentences found: {total_sentences}")
        
        # Apply prefilter if enabled
        if use_prefilter and total_sentences > max_sentences:
            selected_sentences = self._prefilter_sentences(all_sentences, max_sentences)
            filtered_count = total_sentences - len(selected_sentences)
            if verbose:
                print(f"After smart filtering: {len(selected_sentences)} sentences (filtered {filtered_count})")
        else:
            selected_sentences = [(i, s) for i, s in enumerate(all_sentences)][:max_sentences]
            filtered_count = max(0, total_sentences - max_sentences)
            if verbose and filtered_count > 0:
                print(f"Limited to first {max_sentences} sentences (skipped {filtered_count})")
        
        if verbose:
            print(f"Processing {len(selected_sentences)} sentences with {max_workers} workers...\n")
        
        # Process sentences (can be parallelized)
        all_results = []
        
        if max_workers > 1 and len(selected_sentences) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_single_sentence,
                        orig_idx, sentence, all_sentences, question, False
                    ): (orig_idx, sentence)
                    for orig_idx, sentence in selected_sentences
                }
                
                for future in tqdm(
                    as_completed(futures),
                    total=len(selected_sentences),
                    desc="Extracting claims",
                    disable=not verbose
                ):
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        orig_idx, sentence = futures[future]
                        print(f"Error processing sentence {orig_idx}: {e}")
        else:
            # Sequential processing (with verbose output)
            for i, (orig_idx, sentence) in enumerate(selected_sentences):
                if verbose:
                    print(f"--- Sentence {i+1}/{len(selected_sentences)} (idx: {orig_idx}) ---")
                    print(f"Original: {sentence[:80]}...")
                
                result = self._process_single_sentence(
                    orig_idx, sentence, all_sentences, question, verbose
                )
                all_results.append(result)
                
                if verbose:
                    print()
        
        # Sort by original index
        all_results.sort(key=lambda x: x["original_idx"])
        
        # Aggregate results
        all_claims = []
        claim_details = []
        stats = {
            "no_verifiable": 0,
            "ambiguous": 0,
            "with_claims": 0,
            "no_claims": 0
        }
        
        for r in all_results:
            status = r["status"]
            if status == "no_verifiable":
                stats["no_verifiable"] += 1
            elif status == "ambiguous":
                stats["ambiguous"] += 1
            elif status == "success":
                stats["with_claims"] += 1
                for claim in r["claims"]:
                    all_claims.append(claim)
                    claim_details.append({
                        "claim": claim,
                        "sentence_index": r["original_idx"],
                        "source_sentence": r["sentence"],
                        "modified_sentence": r["modified_sentence"],
                        "decontextualized_sentence": r["decontextualized_sentence"]
                    })
            elif status == "no_claims":
                stats["no_claims"] += 1
        
        return ClaimifyResult(
            claims=all_claims[:max_claims],
            claim_details=claim_details[:max_claims],
            sentences_total=total_sentences,
            sentences_processed=len(selected_sentences),
            sentences_with_claims=stats["with_claims"],
            sentences_no_verifiable=stats["no_verifiable"] + stats["no_claims"],
            sentences_ambiguous=stats["ambiguous"],
            sentences_filtered=filtered_count
        )
    
    def extract_claims(self, text: str, max_sentences: int = 10) -> List[str]:
        """
        Wrapper method for compatibility with existing architecture.
        Returns a list of claim texts (strings).
        """
        result = self.extract(text, max_sentences=max_sentences)
        return result.claims

    def extract_batch(
        self,
        articles: List[str],
        questions: Optional[List[str]] = None,
        max_claims_per_article: int = 20,
        max_sentences_per_article: int = 10,
        max_workers: int = 3
    ) -> List[ClaimifyResult]:
        """
        Extract claims from multiple articles.
        
        Args:
            articles: List of article texts.
            questions: Optional list of questions (one per article).
            max_claims_per_article: Maximum claims per article.
            max_sentences_per_article: Maximum sentences to process per article.
            max_workers: Number of parallel workers per article.
            
        Returns:
            List of ClaimifyResult objects.
        """
        if questions is None:
            questions = [None] * len(articles)
        
        results = []
        for i, (article, question) in enumerate(zip(articles, questions)):
            print(f"\n{'='*50}")
            print(f"Processing article {i+1}/{len(articles)}")
            print(f"{'='*50}")
            
            result = self.extract(
                article,
                question=question,
                max_claims=max_claims_per_article,
                max_sentences=max_sentences_per_article,
                max_workers=max_workers,
                verbose=True
            )
            results.append(result)
        
        return results


# ============================================================
# MAIN - Test the extractor
# ============================================================

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
    
    TEST_QUESTION = "What are the key highlights from Tesla's Q3 2024 earnings?"

    try:
        extractor = ClaimifyExtractor()
        
        print("\n" + "=" * 60)
        print("CLAIMIFY EXTRACTOR TEST")
        print("=" * 60)
        print(f"\nQuestion: {TEST_QUESTION}")
        print(f"\nSettings: max_sentences=5, use_prefilter=True")
        
        result = extractor.extract(
            TEST_ARTICLE,
            question=TEST_QUESTION,
            max_sentences=5,      
            use_prefilter=True,   
            max_workers=1,   
            verbose=True
        )
        
        print("\n" + "=" * 60)
        print("EXTRACTION RESULTS")
        print("=" * 60)
        print(f"\nStatistics:")
        print(f"  - Total sentences in article: {result.sentences_total}")
        print(f"  - Sentences filtered (skipped): {result.sentences_filtered}")
        print(f"  - Sentences processed: {result.sentences_processed}")
        print(f"  - Sentences with claims: {result.sentences_with_claims}")
        print(f"  - Sentences (no verifiable): {result.sentences_no_verifiable}")
        print(f"  - Sentences (ambiguous): {result.sentences_ambiguous}")
        
        print(f"\nExtracted {len(result.claims)} claims:\n")
        for i, detail in enumerate(result.claim_details, 1):
            print(f"[{i}] {detail['claim']}")
            print(f"    Source (idx {detail['sentence_index']}): {detail['source_sentence'][:50]}...")
            print()
            
    except ValueError as e:
        print(f"\nSetup Error: {e}")
        print("\nTo test this module:")
        print("1. Get free API key at: https://console.groq.com/")
        print("2. Add to .env: GROQ_API_KEY=your-key-here")
        print("3. Run: uv run python src/extractor_claimify.py")

    







    
    