import os
from pathlib import Path
import sys

# Ensure both project root and backend dir are in sys.path when running from any CWD
_backend_dir = Path(__file__).resolve().parent
_root_dir = _backend_dir.parent
for _p in [str(_root_dir), str(_backend_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contextlib import asynccontextmanager
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from backend.app.api.health_routes import router as health_router
from backend.app.api.rag_routes import router as rag_router
from backend.app.services.rag_service import RAGService
from backend.app.services.redis_manager import RedisCacheManager
from backend.app.utils.env_utils import check_api_keys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("backend.main")

load_dotenv(verbose=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: verify API credentials and pre-warm Memori Labs and RAG on startup."""
    log.info("Starting GithubChat API service...")
    try:
        keys = check_api_keys()
        if not keys.get("gemini"):
            log.warning("GEMINI_API_KEY is not configured. Set it in .env or environment variables.")
        if not keys.get("groq"):
            log.warning("GROQ_API_KEY is not configured. Set it in .env or environment variables.")
    except Exception as key_err:
        log.warning(f"Could not check API keys on startup: {key_err}")

    # Reconcile orphaned background jobs from prior server crashes/restarts
    try:
        reconciled = RedisCacheManager.get_instance().reconcile_orphaned_jobs()
        if reconciled > 0:
            log.info(f"Startup: Reconciled {reconciled} interrupted background jobs.")
    except Exception as recon_err:
        log.debug(f"Startup job reconciliation skipped: {recon_err}")

    # Pre-warm Memori Labs memory layer and LangGraph RAG on server startup
    try:
        rag_service = RAGService.get_instance()
        rag_service.get_rag()
        log.info("Memori Labs memory layer and LangGraph Agentic RAG pre-warmed successfully.")
    except (RuntimeError, ValueError) as prewarm_err:
        log.info(f"RAG engine pre-warm deferred to first request: {prewarm_err}")
    except Exception as e:
        log.info(f"RAG engine pre-warm deferred to first request: {e}")

    yield
    log.info("Shutting down GithubChat API service.")


# Initialize FastAPI application
app = FastAPI(
    title="GithubChat API",
    description="Production-grade API for querying GitHub repositories using RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# High-performance GZip compression middleware (compresses responses > 1KB by 70-85%)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Global Exception Handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    log.warning(f"Validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Bad Request", "detail": str(exc)},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    log.error(f"Runtime error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Server Error", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


# Mount routers: Health/System endpoints and RAG business operations
app.include_router(health_router)
app.include_router(rag_router)


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

