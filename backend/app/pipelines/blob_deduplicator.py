"""Git-blob content-addressable deduplication inspired by GitHub Blackbird code search engine."""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from backend.app.services.redis_manager import RedisCacheManager

log = logging.getLogger(__name__)


class BlobDeduplicator:
    """Computes Git-compatible blob object hashes and coordinates deduplication via Redis."""

    _instance: Optional['BlobDeduplicator'] = None

    def __init__(self, redis_mgr: Optional[RedisCacheManager] = None):
        self.redis_mgr = redis_mgr or RedisCacheManager.get_instance()

    @classmethod
    def get_instance(cls) -> 'BlobDeduplicator':
        """Singleton accessor for BlobDeduplicator."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def compute_blob_hash(content: str) -> str:
        """Compute standard Git blob object SHA-1 hash for content-addressable storage."""
        try:
            encoded = content.encode("utf-8", errors="replace")
            header = f"blob {len(encoded)}\0".encode("utf-8")
            return hashlib.sha1(header + encoded).hexdigest()
        except Exception as e:
            log.warning(f"BlobDeduplicator: Error computing blob hash: {e}")
            return hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()

    def get_cached_chunks(self, blob_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch previously computed chunks and vector metadata for a given blob hash from Redis."""
        try:
            if not self.redis_mgr or not self.redis_mgr.is_connected:
                return None
            key = f"blob:{blob_hash}"
            cached_data = self.redis_mgr.get(key)
            if cached_data:
                parsed = json.loads(cached_data)
                log.debug(f"BlobDeduplicator: Cache HIT for blob {blob_hash[:8]}")
                return parsed
            return None
        except Exception as e:
            log.warning(f"BlobDeduplicator: Error reading blob cache for {blob_hash[:8]}: {e}")
            return None

    def cache_chunks(
        self,
        blob_hash: str,
        chunks: List[Dict[str, Any]],
        ttl: int = 604800,
    ) -> bool:
        """Store chunk records and vector representations under the blob's content-addressable key."""
        try:
            if not self.redis_mgr or not self.redis_mgr.is_connected or not chunks:
                return False
            key = f"blob:{blob_hash}"
            serialized = json.dumps(chunks)
            return bool(self.redis_mgr.setex(key, ttl, serialized))
        except Exception as e:
            log.warning(f"BlobDeduplicator: Error writing blob cache for {blob_hash[:8]}: {e}")
            return False

