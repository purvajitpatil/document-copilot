"""Query embedding for live retrieval (OpenAI-compatible /v1/embeddings)."""

from __future__ import annotations

from app.config import settings
from app.llm.gateway import embedding_kwargs, get_embedding_client


def embed_query(text: str) -> list[float]:
    response = get_embedding_client().embeddings.create(
        input=[text],
        model=settings.openai_embedding_model,
        **embedding_kwargs(),
    )
    embedding = response.data[0].embedding
    expected_dims = settings.openai_embedding_dimensions
    if len(embedding) != expected_dims:
        raise ValueError(
            f"Expected embedding dimension {expected_dims}, got {len(embedding)}. "
            "The pgvector column is fixed-width; changing the embedding model "
            "requires a matching dimension and a full re-embed."
        )
    return embedding
