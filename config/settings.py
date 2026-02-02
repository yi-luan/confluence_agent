# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
    CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_TOKEN")
    CONFLUENCE_SPACE = os.getenv("CONFLUENCE_SPACE")
    VECTOR_STORE_PATH = "confluence_vectorstore"
    MAX_CONVERSATION_AGE_HOURS = 24
    MODEL_NAME = "BAAI/bge-m3"
    LLM_MODEL = "gemini-pro"
    LLM_TEMPERATURE = 0.3
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "codellama"  # 或 "llama2"
    # Grok LLM API (可選)
    GROK_API_URL: str = os.getenv("GROK_API_URL")
    GROK_API_KEY: str = os.getenv("GROK_API_KEY")

settings = Settings()