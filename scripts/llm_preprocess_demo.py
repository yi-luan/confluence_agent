from services.llm_preprocessor import LLMPreprocessor
from dotenv import load_dotenv
import os

load_dotenv()  # load .env if exists

SAMPLE = "John Doe 是一名軟體工程師，現居於台北。他在 Google 工作。"

if __name__ == '__main__':
    # ensure services package is importable when running from scripts/
    os.environ.setdefault('PYTHONPATH', '.')

    p = LLMPreprocessor()
    out = p.format_text(SAMPLE)
    print("--- LLM Preprocess Output ---")
    print(out)
    print("----------------------------")
