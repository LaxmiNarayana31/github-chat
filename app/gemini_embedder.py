import os
import backoff
import logging
from typing import Dict, Sequence, Optional, Any, List

from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, Embedding, EmbedderOutput
from google import genai
from google.genai import types

from google.api_core.exceptions import InternalServerError, BadRequest, GoogleAPICallError

from adalflow.utils import printc
log = logging.getLogger(__name__)

class GeminiEmbedderClient(ModelClient):
    """A custom model client for Google Gemini embeddings."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY must be set")

        self.sync_client = genai.Client(api_key=self._api_key)

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        """Parse the embedding response into EmbedderOutput format.
        
        The google-genai SDK returns EmbedContentResponse objects with an
        `embeddings` attribute (plural) — a list of ContentEmbedding objects,
        each having a `values` attribute with the actual embedding vector.
        """
        embeddings: List[Embedding] = []
        
        # Response is a list of EmbedContentResponse from call()
        if isinstance(response, list):
            for idx, result in enumerate(response):
                parsed = False
                
                # google-genai SDK: result.embeddings is a list of ContentEmbedding
                if hasattr(result, 'embeddings') and result.embeddings:
                    for content_emb in result.embeddings:
                        if hasattr(content_emb, 'values') and content_emb.values:
                            embeddings.append(Embedding(index=idx, embedding=list(content_emb.values)))
                            parsed = True
                            break
                    if parsed:
                        continue

                # Fallback: older API style with result.embedding (singular)
                if hasattr(result, 'embedding') and hasattr(result.embedding, 'values'):
                    embeddings.append(Embedding(index=idx, embedding=list(result.embedding.values)))
                    continue
                
                # Check for dict access
                if isinstance(result, dict):
                    if 'embedding' in result:
                        emb = result['embedding']
                        if isinstance(emb, dict) and 'values' in emb:
                            embeddings.append(Embedding(index=idx, embedding=emb['values']))
                        else:
                            embeddings.append(Embedding(index=idx, embedding=emb))
                        continue

                # If we get here, log the problem
                printc(f"GeminiEmbedder: Failed to parse result {idx}. Type: {type(result)}. Contents: {str(result)[:200]}", color="yellow")

        elif hasattr(response, 'embeddings') and response.embeddings:
            # Single EmbedContentResponse
            content_emb = response.embeddings[0]
            if hasattr(content_emb, 'values') and content_emb.values:
                embeddings.append(Embedding(index=0, embedding=list(content_emb.values)))
        elif hasattr(response, 'embedding') and hasattr(response.embedding, 'values'):
            embeddings.append(Embedding(index=0, embedding=list(response.embedding.values)))
            
        printc(f"GeminiEmbedder: Parsed {len(embeddings)} embeddings", color="blue")
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

    @backoff.on_exception(
        backoff.expo,
        (InternalServerError, BadRequest, GoogleAPICallError),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Call the Gemini embedding API."""
        if model_type == ModelType.EMBEDDER:
            # task_type: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY
            task_type = api_kwargs.get("task_type", "RETRIEVAL_DOCUMENT")
            # model = api_kwargs.get("model", "models/text-embedding-005")
            model = api_kwargs.get("model", "gemini-embedding-001")
            input_texts = api_kwargs.get("input", [])
            
            if not input_texts:
                log.warning("No input texts provided for embedding")
                return []
            
            # Use embed_content for batch embedding
            embeddings = []
            printc(f"GeminiEmbedder: Embedding {len(input_texts)} texts with task_type={task_type}", color="blue")
            for text in input_texts:
                result = self.sync_client.models.embed_content(
                    model=model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task_type
                    )
                )
                embeddings.append(result)
            return embeddings
        else:
            raise ValueError(f"model_type {model_type} is not supported. This client only supports EMBEDDER.")
