import requests
import json
import re
import os
from typing import Dict
from config.settings import settings

class LLMPreprocessor:
    """Lightweight wrapper to call a Grok-like LLM HTTP API to clean text and return candidate entities.

    Expected response JSON format:
    {
      "clean_text": "...",
      "candidates": [ {"text": "John Doe", "type": "PERSON"}, ... ]
    }

    If GROK API config is missing, this class returns the original text and empty candidates.
    """

    def __init__(self, api_url: str = None, api_key: str = None, timeout: int = 10):
        # support multiple env var names (GROK / GROQ) and allow omitting explicit URL
        env_api_key = os.getenv('GROK_API_KEY') or os.getenv('GROQ_API_KEY')
        env_api_url = os.getenv('GROK_API_URL') or os.getenv('GROQ_API_URL')

        self.api_key = api_key or env_api_key or settings.GROK_API_KEY
        self.api_url = api_url or env_api_url or settings.GROK_API_URL
        # If no explicit URL provided but API key exists, use a reasonable default endpoint.
        if not self.api_url and self.api_key:
            # default to Groq hosted endpoint (OpenAI-compatible)
            self.api_url = 'https://api.groq.com/openai/v1/chat/completions'
        self.timeout = timeout

    def format_text(self, text: str) -> Dict:
        # fallback when no API configured
        if not self.api_url or not self.api_key:
            return {"clean_text": text, "candidates": []}

        prompt = (
            "You are a text normalizer and entity extractor.\n"
            "Input is a possibly mixed Chinese/English technical or descriptive text.\n"
            "Return a JSON object with exactly two fields:\n"
            "1. 'clean_text': a single cleaned string (NOT an array)\n"
            "2. 'candidates': array of entities with 'text' and 'type'\n"
            "For 'type', use the most appropriate category (e.g., PERSON, ORG, GPE, TECHNOLOGY, CONCEPT, EVENT, DATE, etc.).\n"
            "Only return one valid JSON object. Do not repeat or add extra commentary.\n\n"
            "Text:\n" + text
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "llama-3.1-8b-instant",  # Groq's current model
            "messages": [{"role": "system", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0
        }

        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            resp_json = resp.json()
            
            # Extract content from OpenAI-compatible response
            text_resp = resp_json.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

            # optional debug output
            if os.getenv('GROK_DEBUG') == '1':
                print('--- GROK REQUEST ---')
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                print('--- GROK RESPONSE RAW ---')
                print(text_resp)

            # try to extract JSON substring if the model returned extra text
            m = re.search(r"\{[\s\S]*\}", text_resp)
            if m:
                json_text = m.group()
            else:
                json_text = text_resp

            data = json.loads(json_text)

            # ensure keys
            clean = data.get("clean_text", "")
            candidates = data.get("candidates", [])
            if not isinstance(candidates, list):
                candidates = []

            return {"clean_text": clean, "candidates": candidates}

        except Exception as e:
            if os.getenv('GROK_DEBUG') == '1':
                print('Grok call failed:', repr(e))
                if hasattr(e, 'response'):
                    print('Response text:', e.response.text)
            # on any error, gracefully fallback
            return {"clean_text": text, "candidates": []}
