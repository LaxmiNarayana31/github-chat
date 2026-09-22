import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from adalflow.core.model_client import ModelClient
from adalflow.core.types import Embedding, EmbedderOutput, ModelType
from adalflow.utils import printc
from google import genai
from google.genai import types

from backend.app.services.redis_manager import RedisCacheManager

log = logging.getLogger(__name__)


class GeminiEmbedderClient(ModelClient):
    """
    Production-grade model client for Google Gemini embeddings:
    - Supports primary model (gemini-embedding-2) with seamless fallback (gemini-embedding-001, text-embedding-004).
    - Maximizes throughput using Google's batchEmbedContents limit (up to 100 texts per batch).
    - Protects against rate limits (HTTP 429 / RESOURCE_EXHAUSTED) using exponential backoff and jitter.
    - Preserves compatible dimensionality (768) across both primary and fallback models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        dimensions: int = 768,
        model_name: str = "gemini-embedding-2",
        fallback_models: Optional[List[str]] = None,
        max_batch_size: int = 100,
    ):
        super().__init__()
        self._api_key = api_key
        self.dimensions = int(dimensions)
        self.model_name = model_name or "gemini-embedding-2"
        self.fallback_models = (
            list(fallback_models)
            if fallback_models is not None
            else ["gemini-embedding-001", "text-embedding-004"]
        )
        self.max_batch_size = max(1, min(int(max_batch_size), 100))
        self._sync_client = None

    @property
    def sync_client(self) -> genai.Client:
        try:
            if self._sync_client is None:
                api_key = self._api_key or os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GEMINI_API_KEY must be set. Please provide it in your environment or Streamlit secrets."
                    )
                self._sync_client = genai.Client(api_key=api_key)
            return self._sync_client
        except ValueError:
            raise
        except Exception as e:
            log.error(f"GeminiEmbedder: Failed to instantiate genai.Client: {e}")
            raise

    @sync_client.setter
    def sync_client(self, value: Any):
        self._sync_client = value

    def _sanitize_text(self, text: Any, max_chars: int = 8000) -> str:
        """Sanitize text to avoid embedding failures on empty or excessively large inputs."""
        try:
            if not text:
                return "empty file"
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()
            if not text:
                return "empty file"
            # Truncate to avoid exceeding model context limits (~2048 tokens / ~8000 chars)
            if len(text) > max_chars:
                text = text[:max_chars]
            return text
        except Exception as e:
            log.warning(f"GeminiEmbedder: Error sanitizing text: {e}")
            return "empty file"

    def _embed_batch_with_retry_and_fallback(
        self,
        texts: List[str],
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        max_retries: int = 5,
        base_delay: float = 1.5,
    ) -> Any:
        """
        Embed a batch of texts with exponential backoff and automatic model fallback
        (e.g., gemini-embedding-2 -> gemini-embedding-001 -> text-embedding-004).
        """
        sanitized_batch = [self._sanitize_text(t) for t in texts]
        if not sanitized_batch:
            return None

        primary_model = model or self.model_name
        candidate_models = [primary_model]
        for fb in self.fallback_models:
            if fb not in candidate_models:
                candidate_models.append(fb)

        last_exception = None

        for model_candidate in candidate_models:
            is_embedding_2 = "gemini-embedding-2" in (model_candidate or "").lower()
            config_kwargs: Dict[str, Any] = {}
            # gemini-embedding-2 does not support task_type parameter in EmbedContentConfig
            if not is_embedding_2 and task_type:
                config_kwargs["task_type"] = task_type
            if self.dimensions:
                config_kwargs["output_dimensionality"] = self.dimensions

            config_obj = types.EmbedContentConfig(**config_kwargs) if config_kwargs else None
            contents_to_send: Any = sanitized_batch[0] if len(sanitized_batch) == 1 else sanitized_batch

            for attempt in range(max_retries):
                try:
                    result = self.sync_client.models.embed_content(
                        model=model_candidate,
                        contents=contents_to_send,
                        config=config_obj,
                    )
                    return result
                except Exception as e:
                    last_exception = e
                    err_str = str(e).lower()
                    is_transient = any(
                        x in err_str
                        for x in [
                            "429",
                            "rate",
                            "quota",
                            "resource_exhausted",
                            "503",
                            "500",
                            "unavailable",
                            "timeout",
                            "deadline",
                        ]
                    )
                    if is_transient and attempt < max_retries - 1:
                        sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.1, 1.0)
                        log.warning(
                            f"GeminiEmbedder: Retry {attempt + 1}/{max_retries} on model '{model_candidate}' "
                            f"due to rate limit/transient error: {e}. Sleeping {sleep_time:.2f}s"
                        )
                        time.sleep(sleep_time)
                    else:
                        log.warning(
                            f"GeminiEmbedder: Model '{model_candidate}' failed after {attempt + 1} attempts: {e}."
                        )
                        break  # Break out of retry loop to try the next model candidate

            log.warning(f"GeminiEmbedder: Attempting failover from '{model_candidate}' to next fallback model...")

        # If all candidates fail
        log.error(f"GeminiEmbedder: All candidate models {candidate_models} failed to embed batch. Last error: {last_exception}")
        if last_exception is not None:
            raise last_exception
        raise RuntimeError(f"GeminiEmbedder: All candidate models {candidate_models} failed to embed batch.")

    def _embed_single_with_retry(
        self,
        text: str,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        max_retries: int = 5,
        base_delay: float = 1.5,
    ) -> Any:
        """Embed a single text string with exponential backoff and fallback."""
        return self._embed_batch_with_retry_and_fallback(
            texts=[text],
            model=model or self.model_name,
            task_type=task_type,
            max_retries=max_retries,
            base_delay=base_delay,
        )

    def parse_embedding_response(self, response: Any) -> EmbedderOutput:
        """Parse single or batched embedding responses into EmbedderOutput format and validate dimensions."""
        embeddings: List[Embedding] = []

        def extract_values(item: Any) -> Optional[List[float]]:
            val = getattr(item, "values", None)
            if val is not None and isinstance(val, (list, tuple)):
                return list(val)
            emb = getattr(item, "embedding", None)
            if emb is not None:
                emb_val = getattr(emb, "values", None)
                if emb_val is not None and isinstance(emb_val, (list, tuple)):
                    return list(emb_val)
                if isinstance(emb, (list, tuple)):
                    return list(emb)
                if isinstance(emb, dict) and "values" in emb:
                    return list(emb["values"])
            if isinstance(item, dict):
                if "values" in item and isinstance(item["values"], (list, tuple)):
                    return list(item["values"])
                if "embedding" in item:
                    d_emb = item["embedding"]
                    if isinstance(d_emb, (list, tuple)):
                        return list(d_emb)
                    if isinstance(d_emb, dict) and "values" in d_emb:
                        return list(d_emb["values"])
            if isinstance(item, (list, tuple)):
                return list(item)
            return None

        flat_items = []
        if isinstance(response, list):
            for res in response:
                embs = getattr(res, "embeddings", None)
                if embs is not None and isinstance(embs, (list, tuple)) and len(embs) > 0:
                    flat_items.extend(embs)
                else:
                    flat_items.append(res)
        else:
            embs = getattr(response, "embeddings", None)
            if embs is not None and isinstance(embs, (list, tuple)) and len(embs) > 0:
                flat_items.extend(embs)
            else:
                flat_items.append(response)

        for idx, item in enumerate(flat_items):
            vals = extract_values(item)
            if vals is not None:
                if self.dimensions and len(vals) != self.dimensions:
                    log.warning(
                        f"GeminiEmbedder: Dimension mismatch at index {idx}: expected {self.dimensions}, got {len(vals)}"
                    )
                embeddings.append(Embedding(index=idx, embedding=vals))
            else:
                log.error(f"GeminiEmbedder: Failed to extract vector for item {idx}: {type(item)}")

        printc(f"GeminiEmbedder: Successfully parsed {len(embeddings)} embeddings", color="blue")
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
            # Ensure dimensions and model are present
            if "model" not in final_model_kwargs:
                final_model_kwargs["model"] = self.model_name or "gemini-embedding-2"
            if "dimensions" not in final_model_kwargs:
                final_model_kwargs["dimensions"] = self.dimensions
        else:
            raise ValueError(
                f"model_type {model_type} is not supported. This client only supports EMBEDDER."
            )
        return final_model_kwargs

    def _extract_vectors_from_batch_result(self, result: Any) -> List[List[float]]:
        """Extract float vector lists from Google GenAI EmbedContentResponse."""
        if not result:
            return []
        vectors: List[List[float]] = []
        embs = getattr(result, "embeddings", None)
        if embs is not None and isinstance(embs, (list, tuple)) and len(embs) > 0:
            for e in embs:
                vals = getattr(e, "values", None) or (e.get("values") if isinstance(e, dict) else None)
                if vals is not None:
                    vectors.append(list(vals))
        else:
            emb = getattr(result, "embedding", None)
            if emb is not None:
                vals = getattr(emb, "values", None) or (emb.get("values") if isinstance(emb, dict) else None)
                if vals is not None:
                    vectors.append(list(vals))
            elif hasattr(result, "values") and isinstance(result.values, (list, tuple)):
                vectors.append(list(result.values))
            elif isinstance(result, dict) and "values" in result and isinstance(result["values"], (list, tuple)):
                vectors.append(list(result["values"]))
        return vectors

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """
        Call the Gemini embedding API with:
        - Redis multi-tier caching (0 API calls for repeated queries or duplicate text chunks)
        - Maximum batch size (up to 100 texts per request)
        - Exponential backoff and automatic model failover
        """
        try:
            if model_type == ModelType.EMBEDDER:
                redis_mgr = RedisCacheManager.get_instance()

                model = api_kwargs.get("model") or self.model_name or "gemini-embedding-2"
                task_type = api_kwargs.get("task_type", "RETRIEVAL_DOCUMENT")
                input_texts = api_kwargs.get("input", [])

                if not input_texts:
                    log.warning("No input texts provided for embedding")
                    return []

                total = len(input_texts)
                cached_vectors: Dict[int, List[float]] = {}
                uncached_items: List[Tuple[int, str]] = []

                # Step 1: Query Redis Embedding Cache
                for idx, text in enumerate(input_texts):
                    cached_vec = redis_mgr.get_embedding(text)
                    if cached_vec is not None:
                        cached_vectors[idx] = cached_vec
                    else:
                        uncached_items.append((idx, text))

                # Step 2: Batch embed any uncached texts
                if uncached_items:
                    total_uncached = len(uncached_items)
                    batch_size = self.max_batch_size
                    printc(
                        f"GeminiEmbedder: {len(cached_vectors)}/{total} embeddings found in Redis cache. "
                        f"Embedding remaining {total_uncached} texts in batches of {batch_size} (primary: {model})...",
                        color="blue",
                    )
                    for i in range(0, total_uncached, batch_size):
                        chunk = uncached_items[i : i + batch_size]
                        batch_texts = [item[1] for item in chunk]
                        res = self._embed_batch_with_retry_and_fallback(
                            texts=batch_texts,
                            model=model,
                            task_type=task_type,
                        )
                        extracted = self._extract_vectors_from_batch_result(res)
                        for (orig_idx, text_val), vec in zip(chunk, extracted):
                            cached_vectors[orig_idx] = vec
                            redis_mgr.set_embedding(text_val, vec)

                        # Polite pause between large batches
                        if (i + batch_size) < total_uncached:
                            time.sleep(0.2)
                else:
                    printc(
                        f"GeminiEmbedder: All {total} embeddings retrieved directly from Redis cache (0 Gemini API calls).",
                        color="green",
                    )

                # Return structured objects preserving input ordering
                ordered_responses = [{"values": cached_vectors.get(i, [])} for i in range(total)]
                return ordered_responses
            else:
                raise ValueError(
                    f"model_type {model_type} is not supported. This client only supports EMBEDDER."
                )
        except ValueError:
            raise
        except Exception as e:
            log.error(f"GeminiEmbedder: Error during call execution: {e}")
            raise

    def to_dict(self, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        try:
            exclude_list = list(exclude) if exclude is not None else []
            if "_sync_client" not in exclude_list:
                exclude_list.append("_sync_client")
            output = super().to_dict(exclude=exclude_list)
            return output
        except Exception as e:
            log.error(f"GeminiEmbedder: Error converting to dict: {e}")
            raise
