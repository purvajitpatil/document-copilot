from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.config import settings
from app.database.session import get_engine
from app.llm.gateway import check_gateway
from app.logging_config import configure_logging
from app.middleware import RequestContextMiddleware

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    log.info(
        "app.startup",
        env=settings.app_env,
        llm_base_url=settings.llm_base_url or "https://api.openai.com/v1",
        chat_model=settings.llm_chat_model,
        output_mode=settings.llm_output_mode,
        structured_mode=settings.llm_structured_mode,
        embedding_model=settings.openai_embedding_model,
    )
    yield
    log.info("app.shutdown")


app = FastAPI(
    title="Document Copilot",
    lifespan=lifespan,
    docs_url="/docs" if settings.api_docs_enabled else None,
    redoc_url="/redoc" if settings.api_docs_enabled else None,
    openapi_url="/openapi.json" if settings.api_docs_enabled else None,
)
app.include_router(auth_router)
app.include_router(chat_router)

# Order matters: the last middleware added is the outermost. Request context sits
# inside CORS so that even 500 responses carry CORS headers the browser can read.
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID", "Retry-After"],
    max_age=600,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is up. Never touches dependencies."""
    return {"status": "ok"}


def _ping_database() -> None:
    with get_engine().connect() as connection:
        connection.execute(text("SELECT 1"))


@app.get("/ready")
async def ready(response: Response) -> dict[str, object]:
    """Readiness: the database must answer; the LLM gateway is reported, not required."""
    try:
        await run_in_threadpool(_ping_database)
        database = "ok"
    except Exception as exc:  # noqa: BLE001
        log.warning("ready.database_unreachable", error=type(exc).__name__)
        database = "unreachable"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    llm = await run_in_threadpool(check_gateway)
    overall = "ok" if database == "ok" else "unavailable"
    return {"status": overall, "database": database, "llm": llm}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
