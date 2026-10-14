"""Docling-based chunking for SEC Markdown filings."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    HierarchicalChunker,
)
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)

CHUNK_MAX_TOKENS = 512
MARKDOWN_DIR = Path(__file__).resolve().parents[2] / "data" / "markdown"
MANIFEST_PATH = MARKDOWN_DIR / "manifest.json"

_ITEM_SECTION_RE = re.compile(r"\bItem\s+[\dA-Z.]+\b", re.IGNORECASE)


class PatchedOpenAITokenizer(OpenAITokenizer):
    """Allow tiktoken special tokens that appear in SEC filing text."""

    def count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer.encode(
                text=text,
                allowed_special=set(),
                disallowed_special=(),
            )
        )


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Serialize tables as Markdown for 10-K financial tables."""

    def get_serializer(self, doc: Any) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        )


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_index: int
    text: str
    page: str | None
    section: str | None
    token_count: int
    chunk_metadata: dict[str, Any]


def load_manifest_markdown_paths() -> dict[str, str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    paths: dict[str, str] = {}
    for filing in manifest.get("filings", []):
        accession = filing["accession_number"]
        md_path = filing.get("local_path")
        if not md_path:
            raise KeyError(f"No local_path for {accession} in manifest")
        paths[accession] = md_path
    return paths


def markdown_path_for_accession(accession_number: str) -> Path:
    paths = load_manifest_markdown_paths()
    if accession_number not in paths:
        raise KeyError(f"Accession {accession_number} not found in {MANIFEST_PATH}")
    md_path = MARKDOWN_DIR / paths[accession_number]
    if not md_path.is_file():
        raise FileNotFoundError(f"Missing Markdown file: {md_path}")
    return md_path


def build_tokenizer(max_tokens: int = CHUNK_MAX_TOKENS) -> PatchedOpenAITokenizer:
    return PatchedOpenAITokenizer(
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        max_tokens=max_tokens,
    )


def build_hybrid_chunker(
    max_tokens: int = CHUNK_MAX_TOKENS,
) -> HybridChunker:
    # HybridChunker applies token-aware splits on top of HierarchicalChunker output.
    return HybridChunker(
        tokenizer=build_tokenizer(max_tokens=max_tokens),
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=MarkdownTableSerializerProvider(),
    )


def build_hierarchical_chunker() -> HierarchicalChunker:
    return HierarchicalChunker(
        serializer_provider=MarkdownTableSerializerProvider(),
    )


def convert_markdown_to_document(markdown_path: Path) -> Any:
    return DocumentConverter(allowed_formats=[InputFormat.MD]).convert(markdown_path).document


def _page_from_chunk_meta(meta: Any) -> str | None:
    origin = getattr(meta, "origin", None)
    if origin is not None:
        page_no = getattr(origin, "page_no", None)
        if page_no is not None:
            return str(page_no)

    for item in getattr(meta, "doc_items", []):
        prov = getattr(item, "prov", None) or []
        for entry in prov:
            page_no = getattr(entry, "page_no", None)
            if page_no is not None:
                return str(page_no)
    return None


def _section_from_chunk(meta: Any, text: str) -> str | None:
    headings = getattr(meta, "headings", None) or []
    if headings:
        return " > ".join(headings)

    match = _ITEM_SECTION_RE.search(text)
    if match:
        return match.group(0)
    return None


def map_chunk_record(
    *,
    chunk_index: int,
    chunk: Any,
    chunker: HybridChunker,
    filing_metadata: dict[str, Any],
) -> ChunkRecord:
    contextualized = chunker.contextualize(chunk=chunk)
    meta = chunk.meta
    tokenizer = chunker.tokenizer

    return ChunkRecord(
        chunk_index=chunk_index,
        text=contextualized,
        page=_page_from_chunk_meta(meta),
        section=_section_from_chunk(meta, contextualized),
        token_count=tokenizer.count_tokens(contextualized),
        chunk_metadata={
            **_base_chunk_metadata(filing_metadata),
            "chunk_kind": "narrative",
            "raw_text": chunk.text,
            "docling_meta": meta.export_json_dict(),
        },
    )


def chunk_document(
    markdown_path: Path,
    filing_metadata: dict[str, Any],
    *,
    max_chunks: int | None = None,
) -> list[ChunkRecord]:
    doc = convert_markdown_to_document(markdown_path)
    chunker = build_hybrid_chunker()
    records: list[ChunkRecord] = []

    for index, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        if max_chunks is not None and index >= max_chunks:
            break
        records.append(
            map_chunk_record(
                chunk_index=len(records),
                chunk=chunk,
                chunker=chunker,
                filing_metadata=filing_metadata,
            )
        )
    return records


def _base_chunk_metadata(filing_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": filing_metadata.get("ticker"),
        "cik": filing_metadata.get("cik"),
        "company_name": filing_metadata.get("company_name"),
        "form": filing_metadata.get("form"),
        "filing_date": filing_metadata.get("filing_date"),
        "report_date": filing_metadata.get("report_date"),
        "fiscal_year": filing_metadata.get("fiscal_year"),
        "accession_number": filing_metadata.get("accession_number"),
        "primary_document": filing_metadata.get("primary_document"),
        "source_url": filing_metadata.get("source_url"),
    }


def iter_all_markdown_paths() -> Iterator[tuple[str, Path]]:
    for accession, relative_path in load_manifest_markdown_paths().items():
        yield accession, MARKDOWN_DIR / relative_path
