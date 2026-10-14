# Document Copilot

An internal AI chatbot that lets analysts query a corpus of documents in plain English and get sourced, citable answers.

## The client

**Driftwood Capital** — fictional independent investment research firm. Their analysts spend half their week reading 10-Ks and 10-Qs before they can produce any original analysis. Document Copilot eats that intake work so they can skip straight to insight.

Full brief: [docs/client-brief.md](docs/client-brief.md)

## Stack

| Layer              | Choice                                               |
| ------------------ | ---------------------------------------------------- |
| Backend            | Python + FastAPI                                     |
| Frontend           | Vite + React SPA + TypeScript                        |
| Database           | Supabase Postgres (users, chats, documents, chunks)  |
| Migrations         | SQLAlchemy models + Alembic                          |
| Retrieval          | Supabase `pgvector` + Postgres full-text search      |
| Auth               | Supabase Auth (email only)                           |
| Hosting            | Railway                                              |
| LLM                | Any OpenAI-compatible endpoint; OmniRoute (free models) recommended |
| Embeddings         | OpenAI `text-embedding-3-small` (pluggable)          |

## Repo layout

```text
document-copilot/
├── AGENTS.md           # agent instructions (read first)
├── README.md           # this file
├── data/               # local corpus + download script (payloads gitignored)
├── docs/
│   ├── client-brief.md # the client one-pager
│   └── guides/         # setup, deployment, AI gateway (OmniRoute)
├── backend/            # FastAPI service
├── frontend/           # React SPA (Vite)
└── docker-compose.yml  # OmniRoute + backend + frontend
```

## Prerequisites

Install these before setting up `backend/` or `frontend/`:

| Tool | Version | Used for | Install |
| ---- | ------- | -------- | ------- |
| [Python](https://www.python.org/downloads/) | 3.12+ | Backend runtime | OS package manager or python.org |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | latest | Backend deps + `data/download.py` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Node.js](https://nodejs.org/) | 20+ (LTS) | Frontend toolchain | nodejs.org or `nvm install --lts` |
| [pnpm](https://pnpm.io/installation) | latest | Frontend package manager | `corepack enable && corepack prepare pnpm@latest --activate` |

You also need accounts/keys for external services once the app is wired up. Start with [docs/guides/supabase-setup.md](docs/guides/supabase-setup.md) (account + project), then set up the AI gateway: [OmniRoute](docs/guides/omniroute-setup.md) for free chat models, plus an [OpenAI API key](https://platform.openai.com/api-keys) for embeddings.

## Running locally

Start with [docs/guides/supabase-setup.md](docs/guides/supabase-setup.md), then create env files for both services.

Backend env:

```bash
cd backend
cp .env.example .env
```

Fill these values in `backend/.env`:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `DATABASE_URL` using the direct Supabase Postgres connection, not the transaction pooler
- `LLM_BASE_URL` + `LLM_API_KEY` for your OmniRoute gateway (see [docs/guides/omniroute-setup.md](docs/guides/omniroute-setup.md))
- `EMBEDDING_API_KEY` (OpenAI key; embeddings must match the stored `vector(1536)` index)
- `ALLOWED_ORIGINS=http://localhost:5173`

Frontend env:

```bash
cd frontend
cp .env.example .env
```

Fill these values in `frontend/.env`:

- `VITE_API_BASE_URL=http://localhost:8000`
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY`

Install dependencies and migrate the database:

```bash
cd backend
uv sync
uv run alembic upgrade head

cd ../frontend
pnpm install
```

Run the backend:

```bash
cd backend
uv run uvicorn app.main:app --reload
```

Run the frontend in another terminal:

```bash
cd frontend
pnpm dev
```

Open the Vite URL, usually `http://localhost:5173`.

Or run everything in containers (OmniRoute + backend + frontend):

```bash
cp .env.example .env && cp backend/.env.example backend/.env   # fill both in
docker compose up -d --build
```

Health endpoints: `GET /health` (liveness) and `GET /ready` (database + LLM gateway status).

Useful checks:

```bash
cd backend
uv run pytest -m "not integration"
uv run ruff check .
uv run python scripts/smoke_retrieval.py

cd ../frontend
pnpm lint
pnpm build
```

## Sample SEC data

Use the standalone downloader to fetch a small local 10-K sample from SEC EDGAR. Edit the params at the top of `data/download.py`, especially `USER_AGENT`, then run:

```bash
uv run data/download.py
```

By default this downloads the latest 5 10-K filings for AAPL, MSFT, NVDA, AMZN, and GOOGL into year folders under `data/downloads/` and writes a `manifest.json`.
Downloaded files are gitignored; the `data/` folder itself stays in git for the script and notes.

Convert downloaded HTML filings to Markdown:

```bash
uv run data/convert_to_markdown.py
```

Load filing metadata into Supabase:

```bash
cd backend
uv sync --extra ingest
uv run python -m ingest.load_source_documents
```

Chunk and embed the full sample corpus:

```bash
cd backend
uv run python -m ingest.chunk_and_embed --all
```

Update an existing corpus:

```bash
# Re-run metadata loading; existing rows are skipped by default.
cd backend
uv run python -m ingest.load_source_documents

# Ingest only new filings; existing chunks are skipped by default.
uv run python -m ingest.chunk_and_embed --all
```

Force-refresh chunks for one filing after changing chunking logic:

```bash
cd backend
uv run python -m ingest.chunk_and_embed --accession 0000000000-00-000000 --force
```
