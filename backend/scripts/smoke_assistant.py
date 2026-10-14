"""Run one assistant smoke query. Edit the QUERY below, then: uv run python scripts/smoke_assistant.py"""

from __future__ import annotations

import asyncio
import sys
import uuid

import nest_asyncio

from app.assistant.agent import run_document_agent
from app.assistant.deps import DocumentAgentDeps, TurnRegistry
from app.assistant.progress import (
    add_progress_listener,
    clear_progress_listeners,
    elapsed_seconds,
    report_progress,
    reset_progress_clock,
)
from app.config import settings
from app.grounding.validator import GroundingValidator, prune_unreferenced_citations
from app.retrieval.retriever import DocumentRetriever

nest_asyncio.apply()

QUERY = (
    "Across Apple's 2021-2025 10-Ks, how did the revenue mix between iPhone, "
    "Services, Mac, iPad, and Wearables change?"
)


def _print_progress(message: str) -> None:
    print(f"[{elapsed_seconds():7.2f}s] {message}", flush=True)


def setup_progress_logging() -> None:
    clear_progress_listeners()
    reset_progress_clock()
    add_progress_listener(_print_progress)


def main(query: str) -> None:
    setup_progress_logging()

    registry = TurnRegistry()
    deps = DocumentAgentDeps(
        retriever=DocumentRetriever(),
        registry=registry,
        thread_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
    )

    print(f"Model: {settings.openai_chat_model}", flush=True)
    print(f"Query: {query}\n", flush=True)

    answer = prune_unreferenced_citations(run_document_agent(query, deps))
    validation = asyncio.run(GroundingValidator().validate(answer, registry))

    report_progress(
        f"grounding validation ok={validation.ok} "
        f"insufficient_evidence={answer.insufficient_evidence} "
        f"citations={len(answer.citations)}"
    )
    if validation.error:
        report_progress(f"grounding validation error: {validation.error}")

    print(f"\ninsufficient_evidence: {answer.insufficient_evidence}", flush=True)
    print(f"validation_ok: {validation.ok}", flush=True)
    if validation.error:
        print(f"validation_error: {validation.error}", flush=True)
    print(f"\n{answer.answer}\n", flush=True)

    for c in answer.citations:
        p = registry.passages_by_chunk_id.get(c.chunk_id)
        meta = f"{p.ticker} {p.form} p.{p.page}" if p else ""
        print(f"[{c.citation_index}] {meta}\n  {c.excerpt[:200]}", flush=True)


if __name__ == "__main__":
    try:
        main(QUERY)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr, flush=True)
        raise SystemExit(130)
