"""
Shared Gemini embedding helpers for ChromaDB integrations.
"""

from typing import List
import time
from config import EMBEDDING_MODEL, client
from chromadb.utils.embedding_functions import EmbeddingFunction
from token_logger import log_token_usage, extract_token_counts

def embed_text(text: str) -> List[float]:
    """Embed text using Gemini embeddings."""
    response = None
    start = time.time()
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[{"parts": [{"text": text}]}],
        )
        return list(response.embeddings[0].values)
    except Exception as e:
        raise RuntimeError(f"Failed to get embedding: {e}") from e

class GeminiEmbeddingFunction(EmbeddingFunction):
    """Chroma-compatible embedding function backed by Gemini embeddings."""
    def __call__(self, texts: List[str]) -> List[List[float]]:
        return [embed_text(t) for t in texts]

def get_gemini_embedding_function() -> EmbeddingFunction:
    """Convenience helper for wiring the embedding function into Chroma."""
    return GeminiEmbeddingFunction()
