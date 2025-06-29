import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv()

# Check API key
api_key = os.getenv('GEMINI_API_KEY')
print(f"1. API Key exists: {bool(api_key)}")
print(f"2. API Key length: {len(api_key) if api_key else 0}")

# Test Gemini connection
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello")
        print("3. Gemini API works: ✅")
        print(f"4. Response: {response.text[:50]}...")
    except Exception as e:
        print(f"3. Gemini API error: ❌ {e}")
else:
    print("3. Cannot test - no API key")

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

print("Available models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"- {model.name}")