from __future__ import annotations

import json
import logging
import threading
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

from fastapi.responses import StreamingResponse

from backend.app.rag.rag import RAG
from backend.app.services.redis_manager import RedisCacheManager
from backend.app.dto.document_dto import Document, DocumentMetadata
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
from backend.app.utils.env_utils import ensure_api_keys_configured
from backend.app.utils.repo_utils import get_repo_slug, normalize_repo_url

log = logging.getLogger(__name__)


class RAGService:
    """Enterprise service managing session isolation, caching, distributed locks, and RAG orchestrations."""

    _instance: Optional['RAGService'] = None
    _lock = threading.Lock()

    def __init__(self):
        self._session_rags: Dict[str, RAG] = {}
        self._session_repos: Dict[str, str] = {}
        self._default_session = "api_user"
        self._init_lock = threading.Lock()
        self.redis_cache = RedisCacheManager.get_instance()

    @classmethod
    def get_instance(cls) -> 'RAGService':
        """Singleton accessor for RAGService."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _resolve_session_id(self, session_id: Optional[str] = None) -> str:
        """Resolve clean session identifier for multi-user isolation."""
        if session_id and str(session_id).strip():
            return str(session_id).strip()
        return self._default_session

    def get_rag(self, session_id: Optional[str] = None) -> RAG:
        """Get or lazily instantiate session-isolated RAG engine with API key validation."""
        sid = self._resolve_session_id(session_id)
        if sid not in self._session_rags:
            with self._init_lock:
                if sid not in self._session_rags:
                    ensure_api_keys_configured()
                    try:
                        rag = RAG(entity_id=sid, process_id="fastapi_backend")
                        self._session_rags[sid] = rag
                        log.info(f"RAGService: Initialized core RAG engine for session '{sid}'.")
                    except Exception as e:
                        log.error(f"RAGService: Failed to instantiate RAG engine for session '{sid}': {e}")
                        raise RuntimeError(f"Failed to initialize RAG engine: {e}") from e
        return self._session_rags[sid]

    def initialize_repository(
        self,
        repo_url: str,
        force_reindex: bool = False,
        token: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Clone repository, generate chunk embeddings, and build Qdrant hybrid index."""
        normalized = normalize_repo_url(repo_url)
        if not normalized:
            raise ValueError("A valid repository URL or local folder path is required.")

        sid = self._resolve_session_id(session_id)
        rag_instance = self.get_rag(session_id=sid)

        # Acquire distributed lock in Redis to prevent concurrent re-indexing of the same repo
        lock_key = f"ingest:{normalized}"
        lock_acquired = self.redis_cache.acquire_lock(lock_key, timeout_seconds=600)
        if not lock_acquired:
            log.info(f"RAGService [{sid}]: Another process holds lock for '{normalized}'. Awaiting or reusing collection...")

        try:
            log.info(f"RAGService [{sid}]: Preparing index for '{normalized}' (force_reindex={force_reindex})")
            rag_instance.prepare_retriever(normalized, force_reindex=force_reindex, token=token)
            self._session_repos[sid] = normalized
            return normalized
        except Exception as e:
            log.error(f"RAGService [{sid}]: Failed to prepare retriever for '{normalized}': {e}")
            raise
        finally:
            if lock_acquired:
                self.redis_cache.release_lock(lock_key)

    def initialize_repository_async(
        self,
        repo_url: str,
        force_reindex: bool = False,
        token: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Dispatch background repository indexing job and return a unique job_id for polling."""
        normalized = normalize_repo_url(repo_url)
        if not normalized:
            raise ValueError("A valid repository URL or local folder path is required.")

        slug = get_repo_slug(normalized)
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        # Initialize job status in Redis
        self.redis_cache.set_job_status(
            job_id,
            {
                "job_id": job_id,
                "status": "pending",
                "progress": 0.05,
                "message": "Repository indexing queued in background worker...",
                "repo_url": normalized,
                "slug": slug,
            },
        )

        def _worker():
            try:
                self.redis_cache.set_job_status(
                    job_id,
                    {
                        "job_id": job_id,
                        "status": "processing",
                        "progress": 0.25,
                        "message": f"Cloning and processing files for repository '{slug}'...",
                        "repo_url": normalized,
                        "slug": slug,
                    },
                )
                self.initialize_repository(
                    repo_url=normalized,
                    force_reindex=force_reindex,
                    token=token,
                    session_id=session_id,
                )
                self.redis_cache.set_job_status(
                    job_id,
                    {
                        "job_id": job_id,
                        "status": "completed",
                        "progress": 1.0,
                        "message": f"Repository '{normalized}' successfully indexed and ready for queries.",
                        "repo_url": normalized,
                        "slug": slug,
                    },
                )
            except Exception as exc:
                log.error(f"RAGService async worker failed for job {job_id}: {exc}")
                self.redis_cache.set_job_status(
                    job_id,
                    {
                        "job_id": job_id,
                        "status": "failed",
                        "progress": 0.0,
                        "message": f"Indexing failed: {str(exc)}",
                        "repo_url": normalized,
                        "slug": slug,
                    },
                )

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve background indexing progress for given job_id."""
        return self.redis_cache.get_job_status(job_id)

    def query(
        self,
        query_str: str,
        repo_url: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> QueryResponse:
        """Execute semantic search and LLM generation for a user query with Redis caching."""
        sid = self._resolve_session_id(session_id)
        rag_instance = self.get_rag(session_id=sid)

        # If repo_url is provided and retriever is not yet ready, initialize it
        current_repo = self._session_repos.get(sid)
        if repo_url and (rag_instance.retriever is None or current_repo != normalize_repo_url(repo_url)):
            self.initialize_repository(repo_url, session_id=sid)

        if rag_instance.retriever is None:
            raise ValueError("No repository has been indexed yet. Please initialize a repository first.")

        cleaned_query = query_str.strip()
        if not cleaned_query:
            raise ValueError("Query string cannot be empty.")

        active_collection = rag_instance.current_collection or "default"

        # Check Redis query cache for fast sub-10ms response
        cached = self.redis_cache.get_query_cache(repo_slug=active_collection, query=cleaned_query)
        if cached:
            cached_contexts = [
                Document(
                    text=c.get("text", ""),
                    meta_data=DocumentMetadata(**c.get("meta_data", {})),
                )
                for c in cached.get("contexts", [])
            ]
            return QueryResponse(
                rationale=cached.get("rationale", "") + " [Cache: Redis HIT]",
                answer=cached.get("answer", ""),
                contexts=cached_contexts,
            )

        try:
            response, retrieved_docs = rag_instance.call(cleaned_query)

            answer = (
                response.answer
                if hasattr(response, "answer") and response.answer
                else getattr(response, "raw_response", str(response))
            )
            rationale = getattr(response, "rationale", "") or ""

            contexts = []
            if retrieved_docs and retrieved_docs[0].documents:
                for doc in retrieved_docs[0].documents:
                    meta = doc.meta_data or {}
                    contexts.append(
                        Document(
                            text=doc.text,
                            meta_data=DocumentMetadata(
                                file_path=meta.get("file_path", ""),
                                type=meta.get("type", ""),
                                is_code=meta.get("is_code", False),
                                is_implementation=meta.get("is_implementation", False),
                                title=meta.get("title", ""),
                            ),
                        )
                    )

            # Store computed answer in Redis cache for subsequent calls
            cache_payload = {
                "rationale": rationale,
                "answer": answer,
                "contexts": [
                    {
                        "text": ctx.text,
                        "meta_data": ctx.meta_data.model_dump()
                        if hasattr(ctx.meta_data, "model_dump")
                        else dict(ctx.meta_data),
                    }
                    for ctx in contexts
                ],
            }
            self.redis_cache.set_query_cache(
                repo_slug=active_collection,
                query=cleaned_query,
                payload=cache_payload,
                ttl=3600,
            )

            return QueryResponse(
                rationale=rationale,
                answer=answer,
                contexts=contexts,
            )
        except Exception as e:
            log.error(f"RAGService [{sid}]: Error during query execution: {e}")
            raise

    def stream_query(
        self,
        query_str: str,
        repo_url: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Iterator[str]:
        """Execute semantic search and stream live tokens via LangGraph Agentic RAG in SSE format."""
        sid = self._resolve_session_id(session_id)
        try:
            rag_instance = self.get_rag(session_id=sid)
            current_repo = self._session_repos.get(sid)

            if repo_url and (rag_instance.retriever is None or current_repo != normalize_repo_url(repo_url)):
                yield f"event: status\ndata: {json.dumps({'stage': 'indexing', 'message': 'Indexing repository...'})}\n\n"
                self.initialize_repository(repo_url, session_id=sid)

            if rag_instance.retriever is None:
                err_payload = {"error": "No repository has been indexed yet. Please initialize a repository first."}
                yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"
                return

            cleaned_query = query_str.strip()
            if not cleaned_query:
                err_payload = {"error": "Query string cannot be empty."}
                yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"
                return

            for event in rag_instance.stream_call(cleaned_query):
                event_type = event.get("type", "status")
                payload = {k: v for k, v in event.items() if k != "type"}
                yield f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

        except Exception as e:
            log.error(f"RAGService [{sid}]: Error during stream_query execution: {e}")
            err_payload = {"error": str(e)}
            yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"

    def clear_memory(self, session_id: Optional[str] = None) -> None:
        """Clear dialog turns in the active conversation memory and Memori storage for a specific session."""
        sid = self._resolve_session_id(session_id)
        try:
            rag_instance = self._session_rags.get(sid)
            if rag_instance is None:
                log.debug(f"RAGService [{sid}]: No active RAG instance; memory is already clean.")
                return
            if hasattr(rag_instance, "memory") and hasattr(rag_instance.memory, "current_conversation"):
                rag_instance.memory.current_conversation.dialog_turns.clear()
            if hasattr(rag_instance, "memori_manager") and rag_instance.memori_manager:
                rag_instance.memori_manager.clear_memory(session_id=sid)
            log.info(f"RAGService [{sid}]: Memory cleared successfully.")
        except Exception as e:
            log.error(f"RAGService [{sid}]: Error clearing memory: {e}")
            raise

    def set_context(self, messages: Any, session_id: Optional[str] = None) -> int:
        """Reconstruct dialog turns from provided message history for a specific session."""
        sid = self._resolve_session_id(session_id)
        try:
            msg_list = messages.messages if hasattr(messages, "messages") else messages
            if not isinstance(msg_list, list) or not msg_list:
                return 0

            try:
                rag_instance = self.get_rag(session_id=sid)
            except (ValueError, RuntimeError) as e:
                log.warning(f"RAGService [{sid}]: API keys not yet available during set_context: {e}")
                return len(msg_list) // 2

            self.clear_memory(session_id=sid)

            turns_added = 0
            for i in range(0, len(msg_list) - 1, 2):
                if i + 1 < len(msg_list):
                    user_msg = msg_list[i]
                    assistant_msg = msg_list[i + 1]

                    u_role = user_msg.get("role", "") if isinstance(user_msg, dict) else getattr(user_msg, "role", "")
                    u_content = user_msg.get("content", "") if isinstance(user_msg, dict) else getattr(user_msg, "content", "")
                    a_role = assistant_msg.get("role", "") if isinstance(assistant_msg, dict) else getattr(assistant_msg, "role", "")
                    a_content = assistant_msg.get("content", "") if isinstance(assistant_msg, dict) else getattr(assistant_msg, "content", "")

                    if u_role == "user" and a_role == "assistant":
                        rag_instance.memory.add_dialog_turn(
                            uq=u_content,
                            ar=a_content,
                        )
                        turns_added += 1

            log.info(f"RAGService [{sid}]: Restored {turns_added} dialog turns.")
            return turns_added
        except Exception as e:
            log.error(f"RAGService [{sid}]: Error restoring context: {e}")
            raise

    # -------------------------------------------------------------------------
    # Route-facing Delegation API (keeps routes completely free of business logic)
    # -------------------------------------------------------------------------

    def handle_init_sync(self, request: InitRequest) -> InitResponse:
        """Process synchronous repository indexing and return standardized InitResponse."""
        try:
            indexed_repo = self.initialize_repository(
                repo_url=request.repo_url,
                force_reindex=request.force_reindex,
                token=request.github_token,
                session_id=request.session_id,
            )
            slug = get_repo_slug(indexed_repo)
            return InitResponse(
                status="success",
                message=f"Repository '{indexed_repo}' indexed successfully.",
                repo_url=indexed_repo,
                slug=slug,
            )
        except ValueError:
            raise
        except Exception as e:
            log.error(f"RAGService: Sync initialization failed for '{request.repo_url}': {e}")
            raise RuntimeError(f"Repository indexing failed: {e}") from e

    def handle_init_async(self, request: InitRequest) -> AsyncInitResponse:
        """Dispatch background repository indexing and return AsyncInitResponse descriptor."""
        try:
            job_id = self.initialize_repository_async(
                repo_url=request.repo_url,
                force_reindex=request.force_reindex,
                token=request.github_token,
                session_id=request.session_id,
            )
            slug = get_repo_slug(request.repo_url)
            return AsyncInitResponse(
                status="queued",
                job_id=job_id,
                message="Repository indexing initiated in background.",
                repo_url=request.repo_url,
                slug=slug,
            )
        except ValueError:
            raise
        except Exception as e:
            log.error(f"RAGService: Async initialization failed for '{request.repo_url}': {e}")
            raise RuntimeError(f"Repository indexing failed: {e}") from e

    def get_job_status_response(self, job_id: str) -> Optional[JobStatusResponse]:
        """Fetch real-time job progress and construct JobStatusResponse model."""
        try:
            status_info = self.get_job_status(job_id)
            if not status_info:
                return None
            return JobStatusResponse(
                job_id=job_id,
                status=status_info.get("status", "unknown"),
                progress=float(status_info.get("progress", 0.0)),
                message=status_info.get("message", ""),
                repo_url=status_info.get("repo_url"),
                slug=status_info.get("slug"),
            )
        except Exception as e:
            log.error(f"RAGService: Failed to retrieve job status for '{job_id}': {e}")
            raise RuntimeError(f"Failed to fetch job status: {e}") from e

    def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Process incoming query request against the active session repository."""
        try:
            return self.query(
                query_str=request.query,
                repo_url=request.repo_url if request.repo_url else None,
                session_id=request.session_id,
            )
        except ValueError:
            raise
        except Exception as e:
            log.error(f"RAGService: Query execution failed: {e}")
            raise RuntimeError(f"Query execution failed: {e}") from e

    def create_streaming_response(self, request: QueryRequest) -> StreamingResponse:
        """Construct FastAPI StreamingResponse streaming SSE chunks from Agentic RAG."""
        try:
            generator = self.stream_query(
                query_str=request.query,
                repo_url=request.repo_url if request.repo_url else None,
                session_id=request.session_id,
            )
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except ValueError:
            raise
        except Exception as e:
            log.error(f"RAGService: Streaming response creation failed: {e}")
            raise RuntimeError(f"Streaming failed: {e}") from e

    def clear_conversation_memory(
        self,
        session_id: Optional[str] = None,
        request: Optional[ClearMemoryRequest] = None,
    ) -> ClearMemoryResponse:
        """Clear conversation memory for a session and return confirmation DTO."""
        try:
            resolved_session = (request.session_id if request and request.session_id else None) or session_id
            self.clear_memory(session_id=resolved_session)
            return ClearMemoryResponse(status="success", message="Conversation memory cleared.")
        except ValueError:
            raise
        except Exception as e:
            log.error(f"RAGService: Clear memory failed: {e}")
            raise RuntimeError(f"Clear memory failed: {e}") from e

    def restore_context(
        self,
        payload: Union[SetContextRequest, List[Dict[str, Any]]],
        session_id: Optional[str] = None,
    ) -> SetContextResponse:
        """Restore conversation context from history messages and return count of restored turns."""
        try:
            turns = self.set_context(payload, session_id=session_id)
            return SetContextResponse(status="success", turns=turns)
        except ValueError:
            raise
        except Exception as e:
            log.error(f"RAGService: Restore context failed: {e}")
            raise RuntimeError(f"Restore context failed: {e}") from e

