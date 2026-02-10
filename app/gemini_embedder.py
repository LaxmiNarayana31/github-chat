import os
import logging
from typing import Dict, Sequence, Optional, Any, List

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, Embedding, EmbedderOutput
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

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        """Convert inputs to API kwargs for embedding."""
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.EMBEDDER:
            if isinstance(input, str):
                input = [input]
            if not isinstance(input, Sequence):
                raise TypeError("input must be a sequence of text")
            final_model_kwargs["input"] = input
        else:
            raise ValueError(f"model_type {model_type} is not supported. This client only supports EMBEDDER.")
        return final_model_kwargs
