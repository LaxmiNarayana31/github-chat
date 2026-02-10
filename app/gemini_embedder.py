import os
import logging
from typing import Optional, Any, List

from adalflow.core.model_client import ModelClient
from adalflow.core.types import Embedding, EmbedderOutput
import google.generativeai as genai

log = logging.getLogger(__name__)

class GeminiEmbedderClient(ModelClient):
    """A custom model client for Google Gemini embeddings."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY must be set")
        genai.configure(api_key=self._api_key)
        self.sync_client = genai

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        """Parse the embedding response into EmbedderOutput format."""
        embeddings: List[Embedding] = []
        
        # Response is a list of embedding results from call()
        if isinstance(response, list):
            for idx, result in enumerate(response):
                # Each result is a dict-like object with 'embedding' key
                if hasattr(result, 'embedding'):
                    embeddings.append(Embedding(index=idx, embedding=result.embedding))
                elif isinstance(result, dict) and 'embedding' in result:
                    embeddings.append(Embedding(index=idx, embedding=result['embedding']))
        elif hasattr(response, 'embedding'):
            # Single embedding
            embeddings.append(Embedding(index=0, embedding=response.embedding))
            
        return EmbedderOutput(data=embeddings)
