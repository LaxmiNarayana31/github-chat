"""FastAPI routing layer for system information and health checks."""

import logging
from typing import Dict

from fastapi import APIRouter, HTTPException, status

from backend.app.dto.system_dto import HealthResponse
from backend.app.services.system_service import SystemService

log = logging.getLogger(__name__)

router = APIRouter(tags=["System"])
system_service = SystemService.get_instance()


@router.get("/", summary="Root endpoint with API overview")
async def root() -> Dict[str, str]:
    """Root endpoint with API description and documentation links."""
    try:
        return system_service.get_api_overview()
    except Exception as e:
        log.error(f"Error fetching API overview: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("/health", response_model=HealthResponse, summary="Health and status check")
async def health_check() -> HealthResponse:
    """Verify service availability and API key configuration status."""
    try:
        return system_service.get_health_status()
    except Exception as e:
        log.error(f"Error checking system health: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("/health/live", summary="Kubernetes Liveness Probe")
async def liveness_probe() -> Dict[str, str]:
    """Lightweight probe verifying the ASGI server process is alive and responsive."""
    return {"status": "alive"}


@router.get("/health/ready", response_model=HealthResponse, summary="Kubernetes Readiness Probe")
async def readiness_probe() -> HealthResponse:
    """Readiness probe verifying database, cache, and vector store readiness for traffic."""
    health = system_service.get_health_status()
    if health.status == "degraded":
        # Process is still accepting requests but operating in degraded capacity
        log.warning("Readiness probe: Service operating in degraded capacity.")
    return health
