"""LLM / Embedding provider 工廠。

業務程式只依賴 LangChain 的 BaseChatModel / Embeddings 介面，
要換 OpenAI、Azure OpenAI、Gemini 或本機 Ollama 只需要改環境變數。
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from app.config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
            stream_usage=True,
        )
    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_chat_deployment,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
            timeout=settings.llm_timeout_seconds,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url,
        )
    if provider == "fake":
        from app.llm.offline import ScriptedChatModel

        return ScriptedChatModel()
    raise ValueError(f"未知的 LLM provider: {provider}")


def get_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embedding_provider
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            dimensions=settings.embedding_dim,
        )
    if provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_embedding_deployment,
            dimensions=settings.embedding_dim,
        )
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model, google_api_key=settings.google_api_key
        )
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)
    if provider == "fake":
        return HashingEmbeddings(dim=settings.embedding_dim)
    raise ValueError(f"未知的 embedding provider: {provider}")


class HashingEmbeddings(Embeddings):
    """不需要模型的決定性 embedding（字元 1-gram + 2-gram 雜湊）。

    用途：CI 測試與檢索評估的基準線。對中文有基本的字面比對能力，
    但沒有語意理解，正式環境請改用 bge-m3 或 text-embedding-3。
    """

    _token_re = re.compile(r"[a-z0-9]+|[一-鿿]")

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        units = self._token_re.findall(text.lower())
        return units + [a + b for a, b in zip(units, units[1:], strict=False)]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "big")
            vec[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
