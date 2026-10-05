"""Enterprise Redis and Upstash Redis caching layer for embeddings, queries, locks, and job statuses."""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import redis
from upstash_redis import Redis as UpstashRedis

log = logging.getLogger(__name__)

REDIS_AVAILABLE = True


class RedisCacheManager:
    """Enterprise Redis and Upstash caching layer with automatic failover to local memory."""

    _instance: Optional['RedisCacheManager'] = None

    def __init__(self, redis_url: Optional[str] = None):
        if redis_url is not None:
            raw_url = redis_url.strip() if redis_url else None
        else:
            raw_url = os.getenv("REDIS_URL", "").strip() or None
        self.redis_url = raw_url
        self.upstash_rest_url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip() or None
        self.upstash_rest_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip() or None
        self._client: Optional[Union[redis.Redis, UpstashRedis]] = None
        self._is_connected = False
        self._local_cache: Dict[str, Tuple[float, Any]] = {}

        if (self.upstash_rest_url and self.upstash_rest_token) or self.redis_url:
            self._connect_redis()
        else:
            log.info("RedisCacheManager: REDIS_URL and Upstash credentials not set. Using high-performance in-memory cache.")

    def _connect_redis(self):
        """Initialize Redis connection (Upstash REST or standard Redis) with health ping and fallback."""
        # 1. Try Upstash REST API if credentials provided
        if self.upstash_rest_url and self.upstash_rest_token:
            try:
                upstash_client = UpstashRedis(url=self.upstash_rest_url, token=self.upstash_rest_token)
                upstash_client.ping()
                self._client = upstash_client
                self._is_connected = True
                log.info("RedisCacheManager: Connected successfully to Upstash Redis via REST API.")
                return
            except Exception as e:
                log.warning(f"RedisCacheManager: Upstash connection failed ({e}). Checking REDIS_URL fallback.")

        # 2. Try standard REDIS_URL (supports Upstash rediss:// endpoints or local redis://)
        if self.redis_url:
            try:
                client = redis.from_url(
                    str(self.redis_url),
                    decode_responses=True,
                    socket_timeout=3.0,
                    socket_connect_timeout=3.0,
                )
                client.ping()
                self._client = client
                self._is_connected = True
                log.info("RedisCacheManager: Connected successfully to Redis via TCP/SSL.")
                return
            except Exception as e:
                log.warning(f"RedisCacheManager: Could not connect to Redis ({e}). Falling back to local in-memory cache.")

        self._client = None
        self._is_connected = False

    @classmethod
    def get_instance(cls) -> 'RedisCacheManager':
        """Singleton accessor for RedisCacheManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def redis_client(self) -> Any:
        """Accessor for underlying Redis client or None if disconnected."""
        return self._client

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self._client is not None

    def get(self, key: str) -> Optional[str]:
        """Fetch raw string value from Redis or local cache fallback."""
        if self.is_connected and self._client is not None:
            try:
                val = self._client.get(key)
                if val is not None:
                    return str(val)
            except Exception as e:
                log.debug(f"Redis get error for {key}: {e}")
        item = self._local_cache.get(key)
        if item:
            exp, val = item
            if exp == 0 or exp > time.time():
                return str(val)
            self._local_cache.pop(key, None)
        return None

    def setex(self, key: str, time_seconds: int, value: str) -> bool:
        """Set key value with expiration in Redis or local cache fallback."""
        if self.is_connected and self._client is not None:
            try:
                return bool(self._client.setex(key, time_seconds, value))
            except Exception as e:
                log.debug(f"Redis setex error for {key}: {e}")
        exp = time.time() + time_seconds if time_seconds > 0 else 0
        self._local_cache[key] = (exp, value)
        return True

    def _hash_key(self, text: str) -> str:
        """Deterministic 32-character SHA-256 hash for cache keys."""
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    @staticmethod
    def _deserialize_dict(val: Any) -> Optional[Dict[str, Any]]:
        """Safely deserialize string, bytes, or dict into Dict[str, Any]."""
        if val is None:
            return None
        if isinstance(val, dict):
            return {str(k): v for k, v in val.items()}
        if isinstance(val, (bytes, bytearray)):
            try:
                val = val.decode("utf-8", errors="replace")
            except Exception:
                return None
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    @staticmethod
    def _deserialize_list_float(val: Any) -> Optional[List[float]]:
        """Safely deserialize string, bytes, or list into List[float]."""
        if val is None:
            return None
        if isinstance(val, list):
            return [float(x) for x in val]
        if isinstance(val, (bytes, bytearray)):
            try:
                val = val.decode("utf-8", errors="replace")
            except Exception:
                return None
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return [float(x) for x in parsed]
            except Exception:
                return None
        return None

    # -------------------------------------------------------------------------
    # Embedding Cache (text -> List[float])
    # -------------------------------------------------------------------------

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding vector for text chunk or query."""
        key = f"emb:{self._hash_key(text)}"
        if self.is_connected and self._client is not None:
            try:
                val = self._client.get(key)
                deserialized = self._deserialize_list_float(val)
                if deserialized is not None:
                    return deserialized
            except Exception as e:
                log.debug(f"Redis get_embedding error: {e}")

        # Local in-memory fallback
        item = self._local_cache.get(key)
        if item:
            exp, val = item
            if exp == 0 or exp > time.time():
                return self._deserialize_list_float(val)
            self._local_cache.pop(key, None)
        return None

    def set_embedding(self, text: str, vector: List[float], ttl: int = 604800):
        """Cache embedding vector (default TTL: 7 days)."""
        key = f"emb:{self._hash_key(text)}"
        if not vector:
            return
        if self.is_connected and self._client is not None:
            try:
                self._client.set(key, json.dumps(vector), ex=ttl)
                return
            except Exception as e:
                log.debug(f"Redis set_embedding error: {e}")

        # Local fallback
        exp = time.time() + ttl if ttl > 0 else 0
        self._local_cache[key] = (exp, vector)

    # -------------------------------------------------------------------------
    # Query Response Cache (repo_slug + query -> {answer, rationale, contexts})
    # -------------------------------------------------------------------------

    def get_query_cache(self, repo_slug: str, query: str) -> Optional[Dict[str, Any]]:
        """Retrieve precomputed RAG answer and contexts for identical query."""
        key = f"qcache:{repo_slug}:{self._hash_key(query)}"
        if self.is_connected and self._client is not None:
            try:
                val = self._client.get(key)
                deserialized = self._deserialize_dict(val)
                if deserialized is not None:
                    log.info(f"RedisCacheManager: Query cache HIT for '{query[:40]}...' on {repo_slug}")
                    return deserialized
            except Exception as e:
                log.debug(f"Redis get_query_cache error: {e}")

        item = self._local_cache.get(key)
        if item:
            exp, val = item
            if exp == 0 or exp > time.time():
                log.info(f"RedisCacheManager (local): Query cache HIT for '{query[:40]}...' on {repo_slug}")
                return self._deserialize_dict(val)
            self._local_cache.pop(key, None)
        return None

    def set_query_cache(self, repo_slug: str, query: str, payload: Dict[str, Any], ttl: int = 3600):
        """Cache RAG query response (default TTL: 1 hour)."""
        key = f"qcache:{repo_slug}:{self._hash_key(query)}"
        if not payload:
            return
        if self.is_connected and self._client is not None:
            try:
                self._client.set(key, json.dumps(payload), ex=ttl)
                return
            except Exception as e:
                log.debug(f"Redis set_query_cache error: {e}")

        exp = time.time() + ttl if ttl > 0 else 0
        self._local_cache[key] = (exp, payload)

    # -------------------------------------------------------------------------
    # Distributed Ingestion Locks & Job Status
    # -------------------------------------------------------------------------

    def acquire_lock(self, lock_key: str, timeout_seconds: int = 300) -> bool:
        """Acquire a distributed lock to prevent duplicate concurrent ingestion."""
        key = f"lock:{lock_key}"
        if self.is_connected and self._client is not None:
            try:
                # SET key val NX EX timeout
                return bool(self._client.set(key, "locked", nx=True, ex=timeout_seconds))
            except Exception as e:
                log.debug(f"Redis acquire_lock error: {e}")

        # Local fallback
        item = self._local_cache.get(key)
        now = time.time()
        if item and item[0] > now:
            return False  # Already locked
        self._local_cache[key] = (now + timeout_seconds, "locked")
        return True

    def release_lock(self, lock_key: str):
        """Release a distributed lock."""
        key = f"lock:{lock_key}"
        if self.is_connected and self._client is not None:
            try:
                self._client.delete(key)
                return
            except Exception as e:
                log.debug(f"Redis release_lock error: {e}")

        self._local_cache.pop(key, None)

    def set_job_status(self, job_id: str, status_data: Dict[str, Any], ttl: int = 86400):
        """Store repository background indexing progress (0-100%, stage, details)."""
        key = f"job:{job_id}"
        payload = dict(status_data)
        payload["updated_at"] = time.time()
        if self.is_connected and self._client is not None:
            try:
                self._client.set(key, json.dumps(payload), ex=ttl)
                return
            except Exception as e:
                log.debug(f"Redis set_job_status error: {e}")

        exp = time.time() + ttl if ttl > 0 else 0
        self._local_cache[key] = (exp, payload)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve background indexing progress."""
        key = f"job:{job_id}"
        if self.is_connected and self._client is not None:
            try:
                val = self._client.get(key)
                deserialized = self._deserialize_dict(val)
                if deserialized is not None:
                    return deserialized
            except Exception as e:
                log.debug(f"Redis get_job_status error: {e}")

        item = self._local_cache.get(key)
        if item:
            exp, val = item
            if exp == 0 or exp > time.time():
                return self._deserialize_dict(val)
            self._local_cache.pop(key, None)
        return None

    def reconcile_orphaned_jobs(self) -> int:
        """On startup, reconcile jobs left in 'processing' or 'pending' state due to server crash/restart.

        Marks orphaned jobs as 'failed' with an explanatory message so clients are not stuck polling forever.
        Returns the count of reconciled jobs.
        """
        reconciled = 0
        now = time.time()

        # Handle local cache
        for k, v in list(self._local_cache.items()):
            if k.startswith("job:"):
                exp, data = v
                if isinstance(data, dict):
                    status = data.get("status")
                    if status in ("processing", "pending"):
                        data["status"] = "failed"
                        data["error"] = "Job interrupted due to abrupt server restart or crash."
                        data["reconciled_at"] = now
                        self._local_cache[k] = (exp, data)
                        reconciled += 1

        # Handle remote Redis if connected
        if self.is_connected and self._client is not None:
            try:
                keys = []
                if hasattr(self._client, "scan_iter"):
                    keys = list(self._client.scan_iter(match="job:*", count=100))
                elif hasattr(self._client, "keys"):
                    keys = self._client.keys("job:*")

                for key in keys:
                    val = self._client.get(key)
                    data = self._deserialize_dict(val)
                    if data and data.get("status") in ("processing", "pending"):
                        data["status"] = "failed"
                        data["error"] = "Job interrupted due to abrupt server restart or crash."
                        data["reconciled_at"] = now
                        ttl = self._client.ttl(key)
                        ex = ttl if ttl and ttl > 0 else 86400
                        self._client.set(key, json.dumps(data), ex=ex)
                        reconciled += 1
            except Exception as e:
                log.warning(f"Error during Redis job reconciliation: {e}")

        if reconciled > 0:
            log.warning(f"Startup Job Reconciliation: Cleaned up {reconciled} orphaned in-flight jobs.")
        return reconciled

    # -------------------------------------------------------------------------
    # Mega-Repo Resumable Ingestion Checkpoints
    # -------------------------------------------------------------------------

    def set_ingest_checkpoint(
        self,
        repo_slug: str,
        last_processed_index: int,
        total_files: int,
        ttl: int = 86400 * 7,
    ):
        """Store resumable ingestion checkpoint for mega-repos (e.g. Linux Kernel)."""
        key = f"checkpoint:{repo_slug}"
        payload = {
            "last_processed_index": last_processed_index,
            "total_files": total_files,
            "updated_at": time.time(),
        }
        if self.is_connected and self._client is not None:
            try:
                self._client.set(key, json.dumps(payload), ex=ttl)
                return
            except Exception as e:
                log.debug(f"Redis set_ingest_checkpoint error: {e}")

        exp = time.time() + ttl if ttl > 0 else 0
        self._local_cache[key] = (exp, payload)

    def get_ingest_checkpoint(self, repo_slug: str) -> Optional[Dict[str, Any]]:
        """Retrieve last saved ingestion checkpoint for resumable indexing."""
        key = f"checkpoint:{repo_slug}"
        if self.is_connected and self._client is not None:
            try:
                val = self._client.get(key)
                deserialized = self._deserialize_dict(val)
                if deserialized is not None:
                    return deserialized
            except Exception as e:
                log.debug(f"Redis get_ingest_checkpoint error: {e}")

        item = self._local_cache.get(key)
        if item:
            exp, val = item
            if exp == 0 or exp > time.time():
                return self._deserialize_dict(val)
            self._local_cache.pop(key, None)
        return None

    def clear_ingest_checkpoint(self, repo_slug: str):
        """Clear ingestion checkpoint once indexing completes fully."""
        key = f"checkpoint:{repo_slug}"
        if self.is_connected and self._client is not None:
            try:
                self._client.delete(key)
                return
            except Exception as e:
                log.debug(f"Redis clear_ingest_checkpoint error: {e}")
        self._local_cache.pop(key, None)

    def clear(self):
        """Clear all local cached items and flush Redis if connected (used for testing)."""
        self._local_cache.clear()
        if self.is_connected and self._client is not None:
            try:
                self._client.flushdb()
            except Exception as e:
                log.debug(f"Redis clear error: {e}")
