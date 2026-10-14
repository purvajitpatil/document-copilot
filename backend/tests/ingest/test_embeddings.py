from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from ingest import embeddings


def _response(count: int, dims: int, *, shuffled: bool = False) -> SimpleNamespace:
    items = [SimpleNamespace(index=i, embedding=[float(i)] * dims) for i in range(count)]
    if shuffled:
        items.reverse()
    return SimpleNamespace(data=items)


def test_embed_texts_batches_and_preserves_order() -> None:
    dims = settings.openai_embedding_dimensions
    client = MagicMock()
    client.embeddings.create.side_effect = [
        _response(2, dims, shuffled=True),
        _response(1, dims),
    ]
    with patch.object(embeddings, "get_embedding_client", return_value=client):
        vectors = embeddings.embed_texts(["a", "b", "c"], batch_size=2)

    assert len(vectors) == 3
    assert vectors[0][0] == 0.0 and vectors[1][0] == 1.0  # index order restored
    first_call = client.embeddings.create.call_args_list[0].kwargs
    assert first_call["input"] == ["a", "b"]
    assert first_call["dimensions"] == dims


def test_embed_texts_rejects_wrong_dimension() -> None:
    client = MagicMock()
    client.embeddings.create.return_value = _response(1, 7)
    with patch.object(embeddings, "get_embedding_client", return_value=client):
        with pytest.raises(ValueError, match="dimension"):
            embeddings.embed_texts(["a"])


def test_embed_texts_empty_input_makes_no_calls() -> None:
    client = MagicMock()
    with patch.object(embeddings, "get_embedding_client", return_value=client):
        assert embeddings.embed_texts([]) == []
    client.embeddings.create.assert_not_called()
