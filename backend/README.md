# Backend

FastAPI service for Document Copilot (Python 3.12+, managed with `uv`).

## Setup

```bash
cp .env.example .env   # fill in Supabase, Postgres, OpenAI values
uv sync                # install deps
uv sync --extra ingest # + ingestion deps (docling)
```

## Run

```bash
uv run uvicorn app.main:app --reload
```

- Health: `http://127.0.0.1:8000/health`
- Docs: `http://127.0.0.1:8000/docs`

## Manage

| Task | Command |
| --- | --- |
| Start dev server | `uv run uvicorn app.main:app --reload` |
| Add dependency | `uv add <package>` |
| Migrations (apply) | `uv run alembic upgrade head` |
| Migrations (new) | `uv run alembic revision --autogenerate -m "<desc>"` |
| Lint | `uv run ruff check .` |
| Tests (fast, no network) | `uv run pytest -m "not integration"` |
| Ingest corpus | `uv run python -m ingest.load_source_documents` then `uv run python -m ingest.chunk_and_embed --all` |

Env is loaded from `.env` by `app.config.settings` (fail-fast). Never read env vars directly in app code.

Full guide: [docs/guides/backend-setup.md](../docs/guides/backend-setup.md).
