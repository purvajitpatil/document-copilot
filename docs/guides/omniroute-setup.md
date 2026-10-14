# AI gateway setup: free models via OmniRoute

Document Copilot talks to its LLM through **one OpenAI-compatible endpoint**
(`LLM_BASE_URL`). [OmniRoute](https://github.com/diegosouzapw/OmniRoute) is a self-hosted
gateway that exposes such an endpoint and routes each request across the free
providers you connect, falling back automatically when a quota runs out or a provider
fails. The backend needs no OmniRoute-specific code: it only changes where requests go.

All model traffic is centralised in `backend/app/llm/gateway.py`:

| Backend feature | Config | Notes |
| --- | --- | --- |
| Document agent (answers + tool calls) | `LLM_CHAT_MODEL` | Needs tool calling, or set `LLM_OUTPUT_MODE=prompted` |
| Citation grounding judge | `LLM_GROUNDING_MODEL` | Accuracy-critical; fails closed if it errors |
| Thread titles (AI service) | `LLM_FAST_MODEL` | Best-effort; falls back to the first message |
| Full-text keyword extraction | `RETRIEVAL_FTS_KEYWORD_MODEL` | Best-effort; deterministic fallback |
| Query + ingestion embeddings | `EMBEDDING_*` | See [Free embeddings](#free-embeddings) |

## 1. Run OmniRoute

Local development, one container:

```bash
docker run -d --name omniroute --restart unless-stopped \
  -p 127.0.0.1:20128:20128 \
  -v omniroute-data:/app/data \
  -e INITIAL_PASSWORD='choose-a-long-random-password' \
  -e REQUIRE_API_KEY=true \
  diegosouzapw/omniroute:latest
```

Or run the whole stack (OmniRoute + backend + frontend) with the repo's
`docker-compose.yml`: copy `.env.example` to `.env`, set `OMNIROUTE_INITIAL_PASSWORD`,
then `docker compose up -d --build`.

Open `http://127.0.0.1:20128` and sign in with `INITIAL_PASSWORD`.

## 2. Connect free providers

Dashboard → **Providers** → **Add Provider**. Connect two or three so fallback has
somewhere to go. OmniRoute's own [free-tier guide](https://github.com/diegosouzapw/OmniRoute/blob/main/docs/getting-started/FREE-TIERS-GUIDE.md)
lists current options; representative ones include Gemini (AI Studio key), Groq,
Cloudflare Workers AI, OpenRouter's free models and keyless Pollinations. Free tiers
are rate-limited, change over time, and some require signup, a key or approval.
A fresh OmniRoute install with nothing connected has no usable route.

Things to check before you rely on a provider:

- **Privacy.** Analyst questions leave your infrastructure and go to the third-party
  provider. Some free providers log requests (OmniRoute's guide notes this for
  Logfare) or may use them for training. Do not send confidential client material
  through providers whose terms you have not read. The *corpus* here is public SEC
  filings, but *questions and retrieved passages* are sent upstream.
- **Terms of service.** OmniRoute flags some sources (e.g. OpenCode Free, Kiro) as
  `tos: avoid` and excludes them from automatic routing by default. Leave that default.
- **Capacity.** "Free" means quotas and rate limits, not guaranteed throughput. Expect
  occasional slow or failed turns during pilot usage; see [Operations](#operations).

## 3. Create an API key and point the backend at it

Dashboard → **Endpoints** → copy an API key. In `backend/.env`:

```bash
LLM_BASE_URL=http://localhost:20128/v1     # http://omniroute:20128/v1 inside docker compose
LLM_API_KEY=<key from the dashboard>
LLM_CHAT_MODEL=auto
LLM_GROUNDING_MODEL=auto/smart
LLM_FAST_MODEL=auto/fast
RETRIEVAL_FTS_KEYWORD_MODEL=auto/fast
```

`auto` lets OmniRoute choose among healthy eligible targets; `auto/fast`,
`auto/smart`, `auto/cheap` and others bias the choice. You can also create a **combo**
in the dashboard (an ordered fallback chain of specific models) and use its name as
the model.

Check it end to end:

```bash
curl http://localhost:20128/v1/models -H "Authorization: Bearer $LLM_API_KEY"
cd backend && uv run uvicorn app.main:app --reload
curl localhost:8000/ready          # {"status":"ok","database":"ok","llm":{"status":"ok",...}}
```

## 4. Model capability caveats (important)

Free and varied models do not all support the same features. The gateway layer handles
the two the app depends on:

- **Structured output** (grounding judge, keywords, titles). `LLM_STRUCTURED_MODE`
  starts at strict JSON-schema output and automatically steps down to JSON-object
  mode and then prompt-only parsing if a model returns a 400 or unparseable output.
  Degradations are logged as `llm.structured_output_degraded`.
- **Agent output + tool calling.** The agent calls retrieval tools, so the routed model
  must support function calling. If routed models struggle, set
  `LLM_OUTPUT_MODE=prompted`, which asks for the final answer as JSON in the prompt
  instead of via a tool call (retrieval tools are still used).

The product's trust guarantee is unchanged by model choice: every citation is checked
for a verbatim excerpt in a retrieved chunk and judged by the grounding model, and a
failed check returns "could not verify" instead of an unverified answer. Weaker models
will therefore show up as *more refusals and retries*, not as wrong cited answers.
Evaluate with the example questions in `docs/client-brief.md` before the pilot.

## Free embeddings

Embeddings are configured separately (`EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`,
`OPENAI_EMBEDDING_MODEL`) and **default to OpenAI**. That is deliberate:

- The database column is `vector(1536)` and every stored chunk was embedded with
  `text-embedding-3-small`. Query vectors must come from the same model, so you cannot
  just swap the embedding model at runtime.
- The free embedding models OmniRoute documents return different widths (for example
  1024 and 3072), not 1536.

If you still want free embeddings, it is a re-index, not a config change (this path is
not exercised by the repo's tests):

1. Pick a model and note its dimension *N* (`GET /v1/embeddings` on OmniRoute lists models).
2. Set `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `OPENAI_EMBEDDING_MODEL`,
   `OPENAI_EMBEDDING_DIMENSIONS=N`, and `EMBEDDING_SEND_DIMENSIONS=false` unless the
   provider supports the `dimensions` parameter.
3. Set `EMBEDDING_DIMENSIONS = N` in `backend/app/database/models/constants.py`.
4. Add an Alembic migration that drops `ix_document_chunks_embedding_hnsw`, nulls the
   embedding column, alters it to `vector(N)`, and recreates the HNSW index.
5. Re-embed: `uv run python -m ingest.chunk_and_embed --all --force`.
6. Never mix models in one index; do not serve traffic until step 5 finishes.

## Operations

- **Never expose OmniRoute publicly.** It holds every provider credential. Bind it to
  loopback or a private network only, keep `REQUIRE_API_KEY=true`, set
  `INITIAL_PASSWORD`, and protect the `/app/data` volume (it stores generated secrets).
  On Railway: add the image as a third service with a volume at `/app/data`, no public
  domain, and use its private hostname for `LLM_BASE_URL`
  (`http://<service>.railway.internal:20128/v1`).
- **Timeouts and retries.** `LLM_TIMEOUT_SECONDS` (default 60) and `LLM_MAX_RETRIES`
  (default 2) apply to every call. OmniRoute does its own provider fallback inside a
  single request, so prefer a modest app-level retry count.
- **Rate limiting.** The backend limits each user to `RATE_LIMIT_CHAT_PER_MINUTE`
  turns per minute so one analyst cannot burn the shared free quota. The limiter is
  per-process; use a shared store or edge limiter before running multiple replicas.
- **Monitoring.** `GET /ready` returns 503 only when the database is down; the `llm`
  field reports gateway reachability without failing readiness. Logs are structured JSON
  in production with a `request_id` on every line (also returned as `X-Request-ID`).
