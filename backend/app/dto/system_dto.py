from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health check and configuration status."""
    status: str = Field(default="healthy")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    version: str = Field(default="1.0.0")
    gemini_configured: bool = Field(..., description="Whether GEMINI_API_KEY is detected")
    groq_configured: bool = Field(..., description="Whether GROQ_API_KEY is detected")
    redis_connected: bool = Field(default=False, description="Whether Redis caching server is connected")
    qdrant_configured: bool = Field(default=False, description="Whether QDRANT_URL is configured")

