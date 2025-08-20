import os
import google.generativeai as genai

key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not key:
    raise SystemExit("Set GEMINI_API_KEY in your environment first.")

genai.configure(api_key=key)

models = genai.list_models()
for m in models:
    print(m)
