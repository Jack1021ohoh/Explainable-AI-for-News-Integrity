import streamlit as st
import os
import sys
import time

# Add parent and src directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import FakeNewsDetector
from src.explainer import LLMExplainer

# Set page config
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .fake-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .real-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .claim-box {
        background-color: #fff3e0;
        border-left: 3px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .evidence-box {
        background-color: #e3f2fd;
        border-left: 3px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


# Cache the detector model to avoid reloading on every run
@st.cache_resource
def load_detector():
    """Load and cache the detector model"""
    return FakeNewsDetector()


# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Main app
def main():
    st.markdown('<h1 class="main-header">Fake News Detection System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This system analyzes news articles using multiple verification methods:
    - **Classification**: BERT-based fake news detection
    - **Claim Extraction**: Identifies key claims in the article
    - **Wikipedia Verification**: Cross-references with knowledge base
    - **Fact-Checking**: Uses Google Fact Check Tools API
    - **LLM Analysis**: Provides comprehensive explanation
    """)

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        gemini_api_key = st.text_input(
            "Gemini API Key (optional)",
            type="password",
            help="For AI-powered explanations using Google Gemini"
        )

        st.info("The system works with basic explanations if API key is not provided.")
        st.info("Note: Only the detector and LLM explainer are currently active. Other features are placeholders for testing.")
    
    # Input section
    st.markdown('<div class="section-header">Article Input</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        title = st.text_input("Article Title", placeholder="Enter the news article title...")
    
    with col2:
        text = st.text_area(
            "Article Text",
            placeholder="Paste the full article text here...",
            height=200
        )
    
    # Process button
    if st.button("🔎 Analyze Article", type="primary", use_container_width=True):
        if not title or not text:
            st.error("Please provide both title and article text.")
            return

        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Load models (with caching, this is fast after first load)
        status_text.text("Loading models...")
        progress_bar.progress(10)
        detector = load_detector()
        explainer = LLMExplainer(gemini_api_key if gemini_api_key else None)

        # Step 2: Classification
        status_text.text("Analyzing article with AI detector...")
        progress_bar.progress(30)
        classification, confidence = detector.classify(text)

        # Step 3: Generate explanation
        status_text.text("Generating detailed explanation...")
        progress_bar.progress(60)
        explanation = explainer.generate_explanation(
            title, text, classification, confidence
        )

        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        # Display results
        st.markdown('<div class="section-header">Analysis Result</div>', unsafe_allow_html=True)

        # Display the explanation in a structured way
        if isinstance(explanation, dict):
            st.markdown(f"### {explanation.get('display_status', 'Analysis Complete')}")
            st.write(explanation.get('explanation', 'No explanation available.'))

            if explanation.get('key_flags'):
                st.markdown("#### Key Indicators:")
                for flag in explanation['key_flags']:
                    st.markdown(f"- {flag}")
        else:
            st.markdown(explanation)

        st.success("✅ Analysis complete!")

if __name__ == "__main__":
    main()
