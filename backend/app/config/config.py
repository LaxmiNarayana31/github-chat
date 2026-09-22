import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv

from backend.app.config.constants import (
    ALL_INDEXABLE_EXTENSIONS,
    CODE_EXTENSIONS,
    DOC_EXTENSIONS,
    IGNORED_DIRS,
    IGNORED_FILES,
    IGNORED_FILE_SUFFIXES,
    MAX_FILE_SIZE_BYTES,
    MIN_FILE_CHAR_LENGTH,
)
from backend.app.models.gemini_embedder import GeminiEmbedderClient
from backend.app.models.groq_client import (
    DEFAULT_GENERATION_MODELS,
    CustomGroqClient,
    MultiProviderLLMClient,
)
from backend.app.utils.env_utils import build_postgres_url

log = logging.getLogger(__name__)

# Ensure environment is loaded from backend/.env or root/.env before reading os.getenv
_this_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_this_dir)
_root_dir = os.path.dirname(_backend_dir)
for _env_path in [os.path.join(_backend_dir, ".env"), os.path.join(_root_dir, ".env")]:
    if os.path.exists(_env_path):
        load_dotenv(_env_path, override=False)
load_dotenv(verbose=False)

# Default configuration constants
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_FALLBACK_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSIONS = 768
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_BATCH_SIZE = 100  # Max batch size for Google Gemini Embeddings API
DEFAULT_TEMPERATURE = 0.3


def create_embedder_client() -> GeminiEmbedderClient:
    """Instantiate and configure the GeminiEmbedderClient."""
    try:
        return GeminiEmbedderClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            dimensions=DEFAULT_EMBEDDING_DIMENSIONS,
            model_name=DEFAULT_EMBEDDING_MODEL,
            fallback_models=[DEFAULT_FALLBACK_EMBEDDING_MODEL, "text-embedding-004"],
            max_batch_size=DEFAULT_BATCH_SIZE,
        )
    except Exception as e:
        log.error(f"Error creating GeminiEmbedderClient: {e}")
        raise


def create_generator_client() -> MultiProviderLLMClient:
    """Instantiate and configure the MultiProviderLLMClient."""
    try:
        return MultiProviderLLMClient(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            models=DEFAULT_GENERATION_MODELS,
        )
    except Exception as e:
        log.error(f"Error creating MultiProviderLLMClient: {e}")
        raise


config: Dict[str, Any] = {
    "embedder": {
        "batch_size": DEFAULT_BATCH_SIZE,
        "model_client": create_embedder_client,
        "model_kwargs": {
            "model": DEFAULT_EMBEDDING_MODEL,
            "dimensions": DEFAULT_EMBEDDING_DIMENSIONS,
        },
        "dimensions": DEFAULT_EMBEDDING_DIMENSIONS,
        "encoding_format": "float",
    },
    "retriever": {"top_k": 3},
    "qdrant": {
        "url": os.getenv("QDRANT_URL", "").strip() or None,
        "api_key": os.getenv("QDRANT_API_KEY", "").strip() or None,
        "cloud_inference": True,
        "collection_prefix": "github_chat_",
    },
    "memori": {
        "enabled": True,
        "connection_string": build_postgres_url(),
        "entity_id": "default_user",
        "process_id": "github-chat",
    },
    "generator": {
        "model_client": create_generator_client,
        "model_kwargs": {
            "temperature": DEFAULT_TEMPERATURE,
            "stream": False,
        },
    },
    "text_splitter": {
        "split_by": "word",
        "chunk_size": 200,
        "chunk_overlap": 100,
    },
}

__all__ = [
    "config",
    "create_embedder_client",
    "create_generator_client",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_FALLBACK_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIMENSIONS",
    "DEFAULT_GROQ_MODEL",
    "DEFAULT_GENERATION_MODELS",
    "DEFAULT_BATCH_SIZE",
    "IGNORED_DIRS",
    "IGNORED_FILES",
    "IGNORED_FILE_SUFFIXES",
    "CODE_EXTENSIONS",
    "DOC_EXTENSIONS",
    "ALL_INDEXABLE_EXTENSIONS",
    "MAX_FILE_SIZE_BYTES",
    "MIN_FILE_CHAR_LENGTH",
    "CustomGroqClient",
]

