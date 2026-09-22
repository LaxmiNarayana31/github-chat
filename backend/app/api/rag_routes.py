"""FastAPI routing layer for repository ingestion and RAG querying operations."""

import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Body, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from backend.app.dto.rag_dto import (
    AsyncInitResponse,
    ClearMemoryRequest,
    ClearMemoryResponse,
    InitRequest,
    InitResponse,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
    SetContextRequest,
    SetContextResponse,
)
from backend.app.services.rag_service import RAGService

log = logging.getLogger(__name__)

router = APIRouter(tags=["RAG Operations"])
rag_service = RAGService.get_instance()


@router.post("/init", response_model=InitResponse, summary="Initialize and index repository (synchronous)")
async def init_repository(request: InitRequest) -> InitResponse:
    """Clone, process, and generate vector embeddings for a repository synchronously."""
    try:
        return rag_service.handle_init_sync(request)
    except ValueError as ve:
        log.warning(f"Validation error in /init: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)) from ve
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error in /init: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.post("/init/async", response_model=AsyncInitResponse, summary="Initialize repository in background (asynchronous)")
async def init_repository_async(request: InitRequest) -> AsyncInitResponse:
    """Dispatch repository cloning and indexing as an asynchronous background worker."""
    try:
        return rag_service.handle_init_async(request)
    except ValueError as ve:
        log.warning(f"Validation error in /init/async: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)) from ve
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error in /init/async: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("/status/{job_id}", response_model=JobStatusResponse, summary="Poll status of background indexing job")
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Poll the real-time status and progress of a background repository indexing job."""
    try:
        response = rag_service.get_job_status_response(job_id)
        if not response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' was not found or has expired.",
            )
        return response
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error in /status/{job_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.post("/query", response_model=QueryResponse, summary="Query repository with RAG")
async def query_repository(request: QueryRequest) -> QueryResponse:
    """Perform semantic search and generate LLM answer with multi-turn memory and Redis cache."""
    try:
        return rag_service.execute_query(request)
    except ValueError as ve:
        log.warning(f"Validation error in /query: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)) from ve
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error in /query: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.post("/query/stream", summary="Query repository with real-time SSE streaming")
async def query_repository_stream(request: QueryRequest) -> StreamingResponse:
    """Perform Agentic RAG search and stream live tokens + status updates via Server-Sent Events."""
    try:
        return rag_service.create_streaming_response(request)
    except ValueError as ve:
        log.warning(f"Validation error in /query/stream: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)) from ve
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error in /query/stream: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.post("/clear-memory", response_model=ClearMemoryResponse, summary="Clear conversation memory")
async def clear_memory(
    request: Optional[ClearMemoryRequest] = Body(default=None),
    session_id: Optional[str] = Query(default=None),
) -> ClearMemoryResponse:
    """Clear dialogue history for the active session."""
    try:
        return rag_service.clear_conversation_memory(session_id=session_id, request=request)
    except ValueError as ve:
        log.warning(f"Validation error in /clear-memory: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)) from ve
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error in /clear-memory: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.post("/set-context", response_model=SetContextResponse, summary="Restore conversation context")
async def set_context(
    payload: Union[SetContextRequest, List[Dict[str, Any]]] = Body(...),
    session_id: Optional[str] = Query(default=None),
) -> SetContextResponse:
    """Restore multi-turn conversation context from client messages for a specific session."""
    try:
        return rag_service.restore_context(payload=payload, session_id=session_id)
    except ValueError as ve:
        log.warning(f"Validation error in /set-context: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve)) from ve
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error in /set-context: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e
