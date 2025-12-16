# Perplexity Fact Checker Integration

The Perplexity Search API has been integrated into your News Integrity application for fact-checking claims.

## What Changed

### ✅ Files Modified

1. **[app/app.py](app/app.py)** - Streamlit application
   - Replaced Google Fact Check API with Perplexity Search API
   - Added Perplexity API key input in sidebar
   - Updated fact-check display to show verdicts, explanations, and sources

2. **[src/explainer.py](src/explainer.py)** - LLM Explainer
   - Enhanced to include Perplexity's detailed explanations and sources in evidence prompts

3. **[config/config.py](config/config.py)** - Configuration
   - Added `PERPLEXITY_API_KEY` environment variable

### ✅ Files Created

1. **[src/perplexity_fact_checker.py](src/perplexity_fact_checker.py)** - Core fact checker using Perplexity Search API

## Setup Instructions

### 1. Install Perplexity SDK

```bash
pip install perplexity-sdk
```

### 2. Get Perplexity API Key

1. Sign up at [https://www.perplexity.ai/](https://www.perplexity.ai/)
2. Navigate to API settings
3. Generate an API key

### 3. Configure Environment Variable

Add to your `.env` file in the project root:

```bash
PERPLEXITY_API_KEY=your-api-key-here
```

Or set it temporarily:

```bash
# Windows
set PERPLEXITY_API_KEY=your-api-key-here

# Linux/Mac
export PERPLEXITY_API_KEY=your-api-key-here
```

### 4. Run the Application

```bash
streamlit run app/app.py
```

## How It Works

The fact checker uses the **Perplexity Search API** (not the chat API):

```python
from perplexity import Perplexity

client = Perplexity(api_key="your-api-key")

# Search for fact-check information
search_results = client.search.create(
    query=f"Is this claim true or false? Fact-check: {claim}",
    max_results=5,
    max_tokens_per_page=1024
)

# Analyze search results to determine verdict
for result in search_results.results:
    print(f"{result.title}: {result.url}")
```

## Usage

1. **Enter API Key** - In the sidebar, enter your Perplexity API key (or it will auto-load from `.env`)

2. **Enable Fact Checking** - Check the "Enable Perplexity Fact Checking" checkbox

3. **Analyze Article** - Enter title and text, then click "Analyze Article"

4. **View Results** - Each claim will show:
   - ✅/❌/⚠️/❓ Verdict icon (TRUE/FALSE/PARTIALLY TRUE/UNVERIFIED)
   - Detailed explanation based on search results
   - Source citations with URLs

## Output Format

For each claim, the fact checker returns:

```json
{
  "claim": "The statement being checked",
  "verdict": "TRUE | FALSE | PARTIALLY TRUE | UNVERIFIED",
  "explanation": "Analysis based on search results",
  "sources": [
    "Source Title - https://source-url.com",
    "Another Source - https://another-url.com"
  ]
}
```

Example display:

```
Claim: ❌ FALSE
  Statement: "The Great Wall of China is visible from the moon."
  Explanation: Based on search results from 5 sources, this claim appears
  to be false. Multiple reliable sources indicate this is a common myth or
  misconception. Key source: NASA - Myths About the Moon.
  Sources:
    - NASA - Myths About the Moon - https://nasa.gov/...
    - Scientific American - https://scientificamerican.com/...
    - National Geographic - https://nationalgeographic.com/...
```

## How Verdicts Are Determined

The fact checker:

1. **Searches** - Uses Perplexity Search API to find relevant web pages
2. **Extracts** - Gets titles, URLs, and snippets from top 5 results
3. **Analyzes** - Examines text for keywords indicating truth/falsehood:
   - **FALSE indicators**: "false", "myth", "debunk", "not true", "incorrect", "hoax"
   - **TRUE indicators**: "true", "correct", "verified", "confirm", "accurate"
   - **PARTIAL indicators**: "partially", "misleading", "some truth", "context"
4. **Determines verdict** - Based on keyword frequency and source credibility
5. **Generates explanation** - Summarizes findings with source attribution

## Benefits Over Google Fact Check API

- ✅ **Real-time search** - Finds current information, not limited to pre-existing fact-checks
- ✅ **Broader coverage** - Can check any claim using web search
- ✅ **Source diversity** - Returns multiple sources with URLs
- ✅ **No database limitations** - Not restricted to fact-checking databases
- ✅ **Official SDK** - Uses Perplexity's official Python SDK

## API Configuration

The fact checker uses:

- **API**: Perplexity Search API
- **Max results**: 5 search results per claim
- **Max tokens per page**: 1024 (snippet length)
- **SDK**: `perplexity-sdk` Python package

## Troubleshooting

### "perplexity SDK not installed"
```bash
pip install perplexity-sdk
```

### "No API key found"
- Make sure `PERPLEXITY_API_KEY` is in your `.env` file or entered in the sidebar

### "API request failed"
- Check your API key is valid
- Verify you have API credits
- Check network connectivity
- Ensure you installed `perplexity-sdk` (not `perplexity`)

### Claims not being fact-checked
- Ensure "Enable Perplexity Fact Checking" is checked
- Verify API key is entered
- Check that claims were successfully extracted

## Code Structure

```
src/
  perplexity_fact_checker.py  # Uses Perplexity Search API
  explainer.py                 # Updated to use Perplexity results
app/
  app.py                       # Streamlit UI with Perplexity integration
config/
  config.py                    # Added PERPLEXITY_API_KEY
```

## Testing the Integration

Test the fact checker standalone:

```python
from src.perplexity_fact_checker import PerplexityFactChecker

# Initialize
checker = PerplexityFactChecker()

# Check a claim
result = checker.check_claim("Water boils at 100°C at sea level")

# Display result
print(checker.format_result(result))
```

Or run the test script:

```bash
python src/perplexity_fact_checker.py
```

## Next Steps

1. Install SDK: `pip install perplexity-sdk`
2. Set your `PERPLEXITY_API_KEY` in `.env`
3. Run the app: `streamlit run app/app.py`
4. Test with a sample article

That's it! Your application now uses the official Perplexity Search API for intelligent fact-checking with source citations.
