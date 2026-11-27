from __future__ import annotations

import asyncio
import importlib
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from .backend import SearchBackend, SearchOptions
from .schemas import HealthResponse, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _load_factory() -> Callable[[], SearchBackend]:
    """Load `module:callable`, defaulting to the bundled integration hook."""
    dotted = os.getenv(
        "PODCAST_SEARCH_BACKEND_FACTORY",
        "semanticsearch.api.integration:build_backend",
    )
    module_name, separator, attr_name = dotted.partition(":")
    if not separator:
        raise RuntimeError(
            "PODCAST_SEARCH_BACKEND_FACTORY must use 'python.module:callable' syntax"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):
        raise TypeError(f"Configured backend factory is not callable: {dotted}")
    return factory


@asynccontextmanager
async def lifespan(app: FastAPI):
    factory = _load_factory()
    backend = factory()  # expensive models/indexes are loaded once here
    app.state.backend = backend
    # The initial implementation deliberately serializes searches. The lexical
    # connection is lock-protected; ONNX session concurrency has not yet been
    # established for this application.
    app.state.search_lock = threading.Lock()
    try:
        yield
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


app = FastAPI(
    title="Podcast Transcript Search API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/api/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    if not hasattr(request.app.state, "backend"):
        raise HTTPException(status_code=503, detail="Search backend is not initialized")
    return HealthResponse(status="ok")


@app.post("/api/search", response_model=SearchResponse)
async def search(payload: SearchRequest, request: Request) -> SearchResponse:
    backend: SearchBackend = request.app.state.backend
    lock: threading.Lock = request.app.state.search_lock
    options = SearchOptions(
        limit=payload.limit,
        lexical_weight=payload.lexical_weight,
        semantic_weight=payload.semantic_weight,
    )

    started = time.perf_counter()
    try:
        # Run blocking SQLite/ONNX work outside the event loop.
        results = await asyncio.to_thread(_locked_search, lock, backend, payload.query, options)
    except NotImplementedError as exc:
        logger.exception("Search backend integration is incomplete")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Search request failed")
        raise HTTPException(status_code=500, detail="Search failed. Check server logs.") from exc

    elapsed_ms = (time.perf_counter() - started) * 1_000
    return SearchResponse(
        query=payload.query,
        lexical_weight=payload.lexical_weight,
        semantic_weight=payload.semantic_weight,
        limit=payload.limit,
        results=list(results),
        elapsed_ms=round(elapsed_ms, 1),
    )


def _locked_search(lock, backend, query, options):
    with lock:
        return backend.search(query, options)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(FRONTEND_DIR / "index.html")
