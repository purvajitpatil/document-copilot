FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    APP_ENV=production

# Pinned (not :latest) so builds are reproducible and match uv.lock's resolver.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

# Run as an unprivileged user.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY alembic ./alembic
COPY alembic.ini ./
COPY app ./app
COPY ingest ./ingest
COPY scripts ./scripts
RUN uv sync --frozen --no-dev \
 && chown -R app:app /app

ENV PATH="/app/.venv/bin:$PATH"

USER app

EXPOSE 8000

# Liveness only (/health never touches dependencies). Use /ready for readiness probes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"]

# Access logs come from the app's structured request logger, so uvicorn's are off.
# WEB_CONCURRENCY (if set) controls worker count; note the in-memory rate limiter
# is per-worker (see app/ratelimit.py).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --no-access-log"]
