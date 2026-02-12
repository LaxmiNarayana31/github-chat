
import os
import logging
from typing import Optional, Any, TypeVar

from adalflow.core.model_client import ModelClient
from adalflow.core.types import CompletionUsage, GeneratorOutput
from groq import Groq, AsyncGroq
from groq.types.chat import ChatCompletion as GroqChatCompletion
from groq.types import CompletionUsage as GroqCompletionUsage

T = TypeVar("T")
log = logging.getLogger(__name__)

class CustomGroqClient(ModelClient):
    """A custom Groq client that properly formats messages."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self._api_key = api_key
        self.sync_client = self.init_sync_client()
        self.async_client = None

    def init_sync_client(self):
        api_key = self._api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Environment variable GROQ_API_KEY must be set")
        return Groq(api_key=api_key)

    def init_async_client(self):
        api_key = self._api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Environment variable GROQ_API_KEY must be set")
        return AsyncGroq(api_key=api_key)

    def parse_chat_completion(self, completion: "GroqChatCompletion") -> "GeneratorOutput":
        """Parse the completion to a string output."""
        log.debug(f"completion: {completion}")
        try:
            data = completion.choices[0].message.content
            usage = self.track_completion_usage(completion)
            return GeneratorOutput(data=None, usage=usage, raw_response=data)
        except Exception as e:
            log.error(f"Error parsing completion: {e}")
            return GeneratorOutput(data=str(e), usage=None, raw_response=completion)

    def track_completion_usage(self, completion: Any) -> CompletionUsage:
        usage: GroqCompletionUsage = completion.usage
        return CompletionUsage(
            completion_tokens=usage.completion_tokens,
            prompt_tokens=usage.prompt_tokens,
            total_tokens=usage.total_tokens,
        )

    
