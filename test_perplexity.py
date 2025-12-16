"""
Quick test to check Perplexity package installation and initialization
"""
import os

print("=" * 60)
print("Testing Perplexity Package")
print("=" * 60)

# Test 1: Check if package is installed
print("\n1. Checking if perplexity package is installed...")
try:
    from perplexity import Perplexity
    print("   ✅ perplexity package is installed")
except ImportError as e:
    print(f"   ❌ perplexity package NOT installed: {e}")
    print("   Install with: pip install perplexity")
    exit(1)

# Test 2: Check if API key is set
print("\n2. Checking for API key...")
api_key = os.getenv("PERPLEXITY_API_KEY")
if api_key:
    print(f"   ✅ API key found: {api_key[:10]}...")
else:
    print("   ❌ PERPLEXITY_API_KEY not set in environment")
    print("   Please set it with:")
    print("      Windows CMD: set PERPLEXITY_API_KEY=your_key")
    print("      PowerShell: $env:PERPLEXITY_API_KEY='your_key'")

    # Ask for manual input for testing
    api_key = input("\n   Enter API key manually for testing (or press Enter to skip): ").strip()
    if not api_key:
        print("   Skipping client initialization test")
        exit(1)

# Test 3: Try to initialize client
print("\n3. Attempting to initialize Perplexity client...")
try:
    client = Perplexity(api_key=api_key)
    print("   ✅ Client initialized successfully")
except Exception as e:
    print(f"   ❌ Client initialization failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    exit(1)

# Test 4: Try a simple search
print("\n4. Testing search functionality...")
try:
    search_results = client.search.create(
        query="What is the capital of France?",
        max_results=2,
        max_tokens_per_page=1024
    )
    print("   ✅ Search executed successfully")
    print(f"\n   Results found: {len(search_results.results)}")

    for i, result in enumerate(search_results.results, 1):
        print(f"\n   Result {i}:")
        print(f"      Title: {result.title}")
        print(f"      URL: {result.url}")

except Exception as e:
    print(f"   ❌ Search failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
