"""
Explainable AI for News Integrity - Streamlit Application

This application provides a complete pipeline for analyzing news articles:
1. Classification - Detect if article is fake or real
2. Claim Extraction - Extract verifiable claims
3. Evidence Retrieval - Find supporting Wikipedia content
4. Fact Checking - Query existing fact-checks
5. Explanation - Generate comprehensive analysis
"""

import streamlit as st
import os
import sys
import time
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from src.classifier import FakeNewsDetector
from src.explainer import LLMExplainer
from src.extractor import ClaimExtractor
from src.retriever import WikiRetriever
# from src.factchecker import FactChecker
FactChecker = None

# Try to import Claimify extractor
try:
    from src.extractor_claimify import ClaimifyExtractor
    CLAIMIFY_AVAILABLE = True
except ImportError:
    CLAIMIFY_AVAILABLE = False
    print("Warning: ClaimifyExtractor not available")

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="News Integrity Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.3rem;
    }
    .result-fake {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .result-real {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .claim-card {
        background-color: #fff8e1;
        border-left: 3px solid #ff9800;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .evidence-card {
        background-color: #e3f2fd;
        border-left: 3px solid #2196f3;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    .factcheck-card {
        background-color: #f3e5f5;
        border-left: 3px solid #9c27b0;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 5px;
    }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .badge-supported { background-color: #c8e6c9; color: #2e7d32; }
    .badge-contradicted { background-color: #ffcdd2; color: #c62828; }
    .badge-unverified { background-color: #fff9c4; color: #f57f17; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Cached Model Loading
# =============================================================================

@st.cache_resource
def load_detector():
    """Load and cache the fake news detector."""
    try:
        return FakeNewsDetector()
    except Exception as e:
        st.error(f"Failed to load detector: {e}")
        return None

@st.cache_resource
def load_retriever():
    """Load and cache the Wikipedia retriever."""
    try:
        return WikiRetriever()
    except FileNotFoundError as e:
        st.warning(f"WikiDB not found: {e}")
        return None
    except Exception as e:
        st.warning(f"Failed to load retriever: {e}")
        return None

@st.cache_resource
def load_simple_extractor(api_key: str = None):
    """Load and cache the simple claim extractor."""
    try:
        if api_key:
            return ClaimExtractor(api_key=api_key)
        return ClaimExtractor()
    except Exception as e:
        st.warning(f"Failed to load simple extractor: {e}")
        return None

@st.cache_resource
def load_claimify_extractor(api_key: str = None):
    """Load and cache the Claimify claim extractor."""
    if not CLAIMIFY_AVAILABLE:
        return None
    try:
        if api_key:
            return ClaimifyExtractor(api_key=api_key)
        return ClaimifyExtractor()
    except Exception as e:
        st.warning(f"Failed to load Claimify extractor: {e}")
        return None

# =============================================================================
# Main Application
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 News Integrity Analyzer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Analyze news articles using AI classification, claim extraction, 
        Wikipedia evidence, and fact-checking APIs.
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # Sidebar Configuration
    # ==========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys Section
        st.subheader("API Keys")

        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="For AI-powered explanations"
        )

        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="For claim extraction using Llama models"
        )

        factcheck_key = st.text_input(
            "Google Fact Check API Key",
            type="password",
            value=os.getenv("GOOGLE_FACTCHECK_API_KEY", ""),
            help="For querying existing fact-checks"
        )
        
        # Extractor Selection
        st.subheader("Claim Extraction")
        
        extractor_mode = st.radio(
            "Extractor Mode",
            options=["simple", "claimify"] if CLAIMIFY_AVAILABLE else ["simple"],
            format_func=lambda x: {
                "simple": "⚡ Simple (Fast)",
                "claimify": "🎯 Claimify (High Quality)"
            }.get(x, x),
            help="Simple: Single prompt, fast\nClaimify: 3-stage pipeline, more accurate"
        )
        
        if extractor_mode == "claimify":
            max_sentences = st.slider(
                "Max Sentences to Process",
                min_value=3,
                max_value=15,
                value=5,
                help="More sentences = more claims but slower"
            )
        else:
            max_sentences = 10  # Not used for simple mode
        
        # Evidence Settings
        st.subheader("Evidence Settings")
        
        max_evidence = st.slider(
            "Max Evidence per Claim",
            min_value=1,
            max_value=5,
            value=3,
            help="Wikipedia results per claim"
        )
        
        enable_factcheck = st.checkbox(
            "Enable Fact Check API",
            value=bool(factcheck_key),
            help="Query Google Fact Check for existing fact-checks"
        )
        
        # System Status
        st.subheader("System Status")
        
        detector = load_detector()
        retriever = load_retriever()
        
        status_items = [
            ("Classifier", detector is not None),
            ("WikiDB", retriever is not None),
            ("Gemini API", bool(gemini_key)),
            ("Groq API", bool(groq_key)),
            ("Fact Check API", bool(factcheck_key)),
            ("Claimify", CLAIMIFY_AVAILABLE)
        ]
        
        for name, available in status_items:
            icon = "✅" if available else "❌"
            st.write(f"{icon} {name}")

    # ==========================================================================
    # Main Content - Article Input
    # ==========================================================================
    
    st.markdown('<div class="section-header">📰 Article Input</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        title = st.text_input(
            "Article Title",
            placeholder="Enter the news article title..."
        )
    
    with col2:
        text = st.text_area(
            "Article Text",
            placeholder="Paste the full article text here...",
            height=150
        )
    
    # Analyze Button
    analyze_button = st.button(
        "🔎 Analyze Article",
        type="primary",
        use_container_width=True,
        disabled=not (title and text)
    )
    
    if not (title and text):
        st.info("Please enter both title and article text to begin analysis.")
        return
    
    if not analyze_button:
        return

    # ==========================================================================
    # Analysis Pipeline
    # ==========================================================================
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    
    try:
        # ----------------------------------------------------------------------
        # Step 1: Classification
        # ----------------------------------------------------------------------
        status_text.text("🔄 Step 1/5: Classifying article...")
        progress_bar.progress(10)
        
        if detector:
            classification, confidence = detector.classify(text)
            results['classification'] = classification
            results['confidence'] = confidence
        else:
            st.error("Classifier not available. Cannot proceed.")
            return
        
        progress_bar.progress(20)
        
        # ----------------------------------------------------------------------
        # Step 2: Claim Extraction
        # ----------------------------------------------------------------------
        status_text.text("🔄 Step 2/5: Extracting claims...")

        claims = []

        if extractor_mode == "simple":
            # Load cached extractor (with or without API key)
            extractor = load_simple_extractor(api_key=groq_key if groq_key else None)
            if extractor:
                # Use extract() method with max_claims limit
                rich_claims = extractor.extract(text, max_claims=5)
                claims = [c.text for c in rich_claims]
        else:
            # Load cached Claimify extractor (with or without API key)
            extractor = load_claimify_extractor(api_key=groq_key if groq_key else None)
            if extractor:
                result = extractor.extract(
                    text,
                    max_claims=5,
                    max_sentences=max_sentences,
                    use_prefilter=True,
                    verbose=False
                )
                claims = result.claims

        results['claims'] = claims
        progress_bar.progress(40)
        
        # ----------------------------------------------------------------------
        # Step 3: Wikipedia Evidence Retrieval
        # ----------------------------------------------------------------------
        status_text.text("🔄 Step 3/5: Retrieving Wikipedia evidence...")
        
        wikipedia_evidence = {}
        
        if retriever and claims:
            wikipedia_evidence = retriever.search_claims(claims, top_k=max_evidence)
        
        results['wikipedia_evidence'] = wikipedia_evidence
        progress_bar.progress(60)
        
        # ----------------------------------------------------------------------
        # Step 4: Fact Check API
        # ----------------------------------------------------------------------
        status_text.text("🔄 Step 4/5: Checking fact-check databases...")
        
        fact_check_results = {}
        
        if enable_factcheck and factcheck_key and claims and FactChecker:
            try:
                fact_checker = FactChecker(api_key=factcheck_key)
                fact_check_results = fact_checker.check_claims_for_article(
                    claims,
                    max_results_per_claim=3
                )
            except Exception as e:
                st.warning(f"Fact Check API error: {e}")
        elif enable_factcheck and not FactChecker:
            # st.warning("FactChecker module is currently disabled.")
            pass
        
        results['fact_check_results'] = fact_check_results
        progress_bar.progress(80)
        
        # ----------------------------------------------------------------------
        # Step 5: Generate Explanation
        # ----------------------------------------------------------------------
        status_text.text("🔄 Step 5/5: Generating explanation...")
        
        explainer = LLMExplainer(api_key=gemini_key if gemini_key else None)
        
        explanation = explainer.generate_explanation(
            title=title,
            text=text,
            classification=classification,
            confidence=confidence,
            claims=claims,
            wikipedia_evidence=wikipedia_evidence,
            fact_check_results=fact_check_results
        )
        
        results['explanation'] = explanation
        progress_bar.progress(100)
        
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        progress_bar.empty()
        status_text.empty()
        return

    # ==========================================================================
    # Display Results
    # ==========================================================================

    # Analysis Result (from Explainer)
    st.markdown('<div class="section-header">📊 Analysis Result</div>', unsafe_allow_html=True)

    # Ensure explanation is a dict (handle edge cases where it might be a string)
    if isinstance(explanation, str):
        try:
            explanation = json.loads(explanation)
        except:
            explanation = {
                "display_status": "Error",
                "explanation": explanation,
                "key_flags": []
            }

    # Extract fields (excluding thought_process which is internal)
    display_status = explanation.get('display_status', 'Analysis Complete')
    explanation_text = explanation.get('explanation', '')
    key_flags = explanation.get('key_flags', [])

    # Try to infer if it's fake/real from display_status for styling
    is_likely_fake = any(word in display_status.lower() for word in ['false', 'fake', 'misinformation', 'misleading', 'unreliable', 'unsubstantiated', 'alarmist', 'satire'])
    is_likely_real = any(word in display_status.lower() for word in ['verified', 'credible', 'true', 'accurate', 'confirmed'])

    # Determine styling based on content
    if is_likely_fake:
        result_class = "result-fake"
        result_icon = "⚠️"
    elif is_likely_real:
        result_class = "result-real"
        result_icon = "✅"
    else:
        result_class = "result-fake"  # Default to warning for uncertain/unverified
        result_icon = "❓"

    st.markdown(f"""
    <div class="{result_class}">
        <h3>{result_icon} {display_status}</h3>
        <p>{explanation_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Flags
    if key_flags:
        st.markdown("**Key Indicators:**")
        for flag in key_flags:
            st.markdown(f"• {flag}")
    
    # ----------------------------------------------------------------------
    # Extracted Claims & Evidence
    # ----------------------------------------------------------------------
    
    if claims:
        st.markdown('<div class="section-header">📝 Extracted Claims & Evidence</div>', unsafe_allow_html=True)
        
        st.info(f"Found **{len(claims)}** verifiable claims using **{extractor_mode}** extractor")
        
        for i, claim in enumerate(claims, 1):
            with st.expander(f"Claim {i}: {claim[:80]}{'...' if len(claim) > 80 else ''}", expanded=(i <= 3)):
                st.markdown(f'<div class="claim-card"><strong>Claim:</strong> {claim}</div>', unsafe_allow_html=True)
                
                # Claim Analysis (if available)
                claim_analysis = explanation.get('claim_analysis', [])
                matching_analysis = next((ca for ca in claim_analysis if ca.get('claim') == claim), None)
                
                if matching_analysis:
                    status = matching_analysis.get('status', 'unknown')
                    badge_class = {
                        'supported': 'badge-supported',
                        'verified': 'badge-supported',
                        'contradicted': 'badge-contradicted',
                        'unverified': 'badge-unverified',
                        'partially_verified': 'badge-unverified'
                    }.get(status, 'badge-unverified')
                    
                    st.markdown(f"""
                    <span class="status-badge {badge_class}">{status.upper()}</span>
                    <span style="margin-left: 10px; color: #666;">{matching_analysis.get('evidence_summary', '')}</span>
                    """, unsafe_allow_html=True)
                
                # Wikipedia Evidence
                wiki_ev = wikipedia_evidence.get(claim, [])
                if wiki_ev:
                    st.markdown("**📚 Wikipedia Evidence:**")
                    for ev in wiki_ev[:3]:
                        evidence_text = ev.get('text', '')
                        source = ev.get('source', 'Unknown')

                        # Show full text in expandable section
                        with st.expander(f"📄 {source}", expanded=False):
                            st.markdown(f"""
                            <div class="evidence-card">
                                {evidence_text}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.caption("No relevant Wikipedia articles found")
                
                # Fact Check Results
                fc_results = fact_check_results.get(claim, [])
                if fc_results:
                    st.markdown("**🔍 Fact-Check Results:**")
                    for fc in fc_results[:2]:
                        rating = fc.get('rating', 'Unknown')
                        publisher = fc.get('publisher', 'Unknown')
                        url = fc.get('url', '')
                        
                        st.markdown(f"""
                        <div class="factcheck-card">
                            <strong>{publisher}</strong>: {rating}<br>
                            <a href="{url}" target="_blank">View full fact-check →</a>
                        </div>
                        """, unsafe_allow_html=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()