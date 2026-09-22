import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    REDIS_AVAILABLE = False


class RedisCacheManager:
    """Enterprise Redis caching layer for embeddings, queries, locks, and job statuses."""

    _instance: Optional['RedisCacheManager'] = None

    def __init__(self, redis_url: Optional[str] = None):
        if redis_url is not None:
            raw_url = redis_url.strip() if redis_url else None
        else:
            raw_url = os.getenv("REDIS_URL", "").strip() or None
        self.redis_url = raw_url
        self._client: Optional[Any] = None
        self._is_connected = False
        self._local_cache: Dict[str, Tuple[float, Any]] = {}

        if self.redis_url and REDIS_AVAILABLE:
            self._connect_redis()
        else:
            log.info("RedisCacheManager: REDIS_URL not set or redis-py not present. Using high-performance in-memory cache.")

    def _connect_redis(self):
        """Initialize Redis connection with health ping and fallback."""
        if not REDIS_AVAILABLE or not self.redis_url or redis is None:
            return
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
            log.info(f"RedisCacheManager: Connected successfully to Redis.")
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
                return self._client.get(key)
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

    # -------------------------------------------------------------------------
    # Embedding Cache (text -> List[float])
    # -------------------------------------------------------------------------

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding vector for text chunk or query."""
        key = f"emb:{self._hash_key(text)}"
        if self.is_connected and self._client is not None:
            try:
                val = self._client.get(key)
                if val:
                    return json.loads(val)
            except Exception as e:
                log.debug(f"Redis get_embedding error: {e}")

        # Local in-memory fallback
        item = self._local_cache.get(key)
        if item:
            exp, val = item
            if exp == 0 or exp > time.time():
                return val
            else:
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
                if val:
                    log.info(f"RedisCacheManager: Query cache HIT for '{query[:40]}...' on {repo_slug}")
                    return json.loads(val)
            except Exception as e:
                log.debug(f"Redis get_query_cache error: {e}")

        item = self._local_cache.get(key)
        if item:
            exp, val = item
            if exp == 0 or exp > time.time():
                log.info(f"RedisCacheManager (local): Query cache HIT for '{query[:40]}...' on {repo_slug}")
                return val
            else:
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
                if val:
                    return json.loads(val)
            except Exception as e:
                log.debug(f"Redis get_job_status error: {e}")

        item = self._local_cache.get(key)
        if item:
            exp, val = item
            if exp == 0 or exp > time.time():
                return val
            else:
                self._local_cache.pop(key, None)
        return None

    def clear(self):
        """Clear all local cached items and flush Redis if connected (used for testing)."""
        self._local_cache.clear()
        if self.is_connected and self._client is not None:
            try:
                self._client.flushdb()
            except Exception as e:
                log.debug(f"Redis clear error: {e}")

