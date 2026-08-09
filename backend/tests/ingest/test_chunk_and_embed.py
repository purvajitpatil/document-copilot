from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import Text

from app.database.models import DocumentChunk, DocumentTable, MessageCitation
from ingest.chunk_and_embed import _delete_chunks, _document_tables_from_markdown


def test_document_table_title_allows_full_sec_captions() -> None:
    assert isinstance(DocumentTable.__table__.c.title.type, Text)


def test_document_chunk_section_allows_full_sec_captions() -> None:
    assert isinstance(DocumentChunk.__table__.c.section.type, Text)


def test_message_citation_section_allows_full_sec_captions() -> None:
    assert isinstance(MessageCitation.__table__.c.section.type, Text)


def test_delete_chunks_removes_dependent_citations_before_chunks() -> None:
    session = MagicMock()
    document_id = uuid.uuid4()

    _delete_chunks(session, document_id)

    deleted_tables = [call.args[0].table.name for call in session.execute.call_args_list]
    assert deleted_tables == [
        "message_citations",
        "document_chunks",
        "document_tables",
    ]


def test_document_tables_from_markdown_reads_tables_json(tmp_path: Path) -> None:
    document_id = uuid.uuid4()
    md_path = tmp_path / "sample.md"
    md_path.write_text("# Heading", encoding="utf-8")
    (md_path.with_suffix(".tables.json")).write_text(
        json.dumps(
            [
                {
                    "table_index": 2,
                    "title": "Products and Services Performance",
                    "units": "dollars in millions",
                    "markdown": "| Category | 2025 Sales |\n| --- | --- |\n| iPhone | $209,586 |",
                    "source_html_hash": "abc123",
                }
            ]
        ),
        encoding="utf-8",
    )

    tables = _document_tables_from_markdown(document_id, md_path)

    assert len(tables) == 1
    assert tables[0].document_id == document_id
    assert tables[0].table_index == 2
    assert tables[0].title == "Products and Services Performance"
    assert tables[0].units == "dollars in millions"
    assert tables[0].markdown.startswith("| Category |")
    assert tables[0].source_html_hash == "abc123"
    assert tables[0].table_data["table_index"] == 2


def test_document_tables_from_markdown_returns_empty_without_tables_file(
    tmp_path: Path,
) -> None:
    md_path = tmp_path / "sample.md"
    md_path.write_text("# Heading", encoding="utf-8")

    tables = _document_tables_from_markdown(uuid.uuid4(), md_path)

    assert tables == []
