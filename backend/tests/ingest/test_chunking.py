from __future__ import annotations

from pathlib import Path

import pytest

from ingest.chunking import (
    CHUNK_MAX_TOKENS,
    ChunkRecord,
    build_hierarchical_chunker,
    build_hybrid_chunker,
    chunk_document,
    convert_markdown_to_document,
    map_chunk_record,
)

FILING_METADATA = {
    "ticker": "TEST",
    "cik": "0000000000",
    "company_name": "Test Corp",
    "form": "10-K",
    "filing_date": "2021-01-01",
    "report_date": "2020-12-31",
    "fiscal_year": 2020,
    "accession_number": "0000000000-21-000001",
    "primary_document": "test.md",
    "source_url": "https://example.com/test.md",
}

SAMPLE_MARKDOWN = """\
# UNITED STATES SECURITIES AND EXCHANGE COMMISSION

# Item 1. Business

We design and sell consumer electronics products worldwide.

# Item 1A. Risk Factors

Our business depends on global supply chains and may be affected by shortages.

## Products and Services Performance

The following table shows net sales by category for 2025, 2024 and 2023 (dollars in millions):

| Category | 2025 | Change | 2024 | Change | 2023 |
| --- | --- | --- | --- | --- | --- |
| iPhone | 209,586 | 4% | 201,183 | - | 200,583 |
| Services (1) | 109,158 | 14% | 96,169 | 13% | 85,200 |

(1) Services net sales include amortization of deferred value.

# Item 2. Properties

We operate offices, data centers, and retail stores worldwide after the table.
"""


@pytest.fixture(scope="module")
def sample_markdown_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    fixture_dir = tmp_path_factory.mktemp("fixtures")
    md_path = fixture_dir / "sample_filing.md"
    md_path.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    return md_path


def test_hierarchical_chunker_produces_chunks(sample_markdown_path: Path) -> None:
    doc = convert_markdown_to_document(sample_markdown_path)
    chunker = build_hierarchical_chunker()
    chunks = list(chunker.chunk(dl_doc=doc))
    assert chunks
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "consumer electronics" in combined.lower()


def test_hybrid_chunker_respects_token_limit(sample_markdown_path: Path) -> None:
    records = chunk_document(sample_markdown_path, FILING_METADATA)
    assert records
    assert all(isinstance(record, ChunkRecord) for record in records)
    assert all(record.token_count <= CHUNK_MAX_TOKENS for record in records)


def test_map_chunk_record_extracts_metadata(sample_markdown_path: Path) -> None:
    doc = convert_markdown_to_document(sample_markdown_path)
    chunker = build_hybrid_chunker()
    chunk = next(chunker.chunk(dl_doc=doc))
    record = map_chunk_record(
        chunk_index=0,
        chunk=chunk,
        chunker=chunker,
        filing_metadata=FILING_METADATA,
    )
    assert record.chunk_index == 0
    assert record.text
    assert record.token_count > 0
    assert record.chunk_metadata["chunk_kind"] == "narrative"
    assert record.chunk_metadata["ticker"] == "TEST"
    assert record.chunk_metadata["raw_text"]
    assert record.chunk_metadata["accession_number"] == FILING_METADATA["accession_number"]
    assert record.chunk_metadata["source_url"] == FILING_METADATA["source_url"]


def test_chunk_document_produces_only_narrative_chunks(sample_markdown_path: Path) -> None:
    records = chunk_document(sample_markdown_path, FILING_METADATA)

    kinds = {record.chunk_metadata.get("chunk_kind") for record in records}

    assert kinds == {"narrative"}
    assert all(record.chunk_metadata["ticker"] == "TEST" for record in records)
    assert [record.chunk_index for record in records] == list(range(len(records)))
