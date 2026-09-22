"""Enterprise system service managing API metadata and system health evaluations."""

from datetime import datetime, timezone
import logging
from typing import Dict, Optional

from backend.app.dto.system_dto import HealthResponse
from backend.app.services.redis_manager import RedisCacheManager
from backend.app.utils.env_utils import check_api_keys

log = logging.getLogger(__name__)


class SystemService:
    """Enterprise service providing system status, health checks, and API metadata."""

    _instance: Optional['SystemService'] = None

    def __init__(self):
        """Initialize SystemService with Redis cache manager instance."""
        try:
            self.redis_mgr = RedisCacheManager.get_instance()
        except Exception as e:
            log.warning(f"SystemService: Could not initialize RedisCacheManager: {e}")
            self.redis_mgr = None

    @classmethod
    def get_instance(cls) -> 'SystemService':
        """Singleton accessor for SystemService."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_api_overview(self) -> Dict[str, str]:
        """Return high-level API metadata and documentation links."""
        try:
            return {
                "name": "GithubChat API",
                "description": "Production-grade RAG service for chatting with GitHub repositories",
                "version": "1.0.0",
                "documentation": "/docs",
                "health_check": "/health",
            }
        except Exception as e:
            log.error(f"SystemService: Error generating API overview metadata: {e}")
            raise RuntimeError(f"Failed to generate API overview: {e}") from e

    def get_health_status(self) -> HealthResponse:
        """Evaluate system health, verifying API keys and database/cache connections."""
        try:
            keys = check_api_keys()
            redis_connected = bool(self.redis_mgr and self.redis_mgr.is_connected)
            qdrant_configured = bool(keys.get("qdrant", False))

            return HealthResponse(
                status="healthy",
                timestamp=datetime.now(timezone.utc).isoformat(),
                version="1.0.0",
                gemini_configured=bool(keys.get("gemini", False)),
                groq_configured=bool(keys.get("groq", False)),
                redis_connected=redis_connected,
                qdrant_configured=qdrant_configured,
            )
        except Exception as e:
            log.error(f"SystemService: Error evaluating system health: {e}")
            return HealthResponse(
                status="degraded",
                timestamp=datetime.now(timezone.utc).isoformat(),
                version="1.0.0",
                gemini_configured=False,
                groq_configured=False,
                redis_connected=False,
                qdrant_configured=False,
            )
