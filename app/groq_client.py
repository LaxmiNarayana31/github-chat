
import os
import re
import logging
from typing import Dict, Sequence, Optional, Any, TypeVar

import backoff
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, CompletionUsage, GeneratorOutput
from groq import Groq, AsyncGroq
from groq.types.chat import ChatCompletion as GroqChatCompletion
from groq.types import CompletionUsage as GroqCompletionUsage
from groq import (
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    UnprocessableEntityError,
)

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

    def _parse_template_to_messages(self, input_text: str) -> Sequence[Dict[str, str]]:
        """Parse the template into separate messages for compound model."""
        messages: Sequence[Dict[str, str]] = []
        
        # Extract system message
        sys_match = re.search(r'<SYS>(.*?)</SYS>', input_text, re.DOTALL)
        if sys_match:
            system_content = sys_match.group(1).strip()
            messages.append({"role": "system", "content": system_content})
        
        # Extract any content between </SYS> and <USER> (context, conversation history)
        # This will be added as additional system context
        middle_match = re.search(r'</SYS>(.*?)<USER>', input_text, re.DOTALL)
        if middle_match:
            middle_content = middle_match.group(1).strip()
            if middle_content:
                # Append context to system message if it exists
                if messages:
                    messages[0]["content"] += "\n\n" + middle_content
                else:
                    messages.append({"role": "system", "content": middle_content})
        
        # Extract user message - MUST be last for compound-mini
        user_match = re.search(r'<USER>(.*?)</USER>', input_text, re.DOTALL)
        if user_match:
            user_content = user_match.group(1).strip()
            messages.append({"role": "user", "content": user_content})
        else:
            if not messages:
                messages.append({"role": "user", "content": input_text})
            else:
                messages.append({"role": "user", "content": "Please respond based on the above context."})
        
        return messages

    def convert_inputs_to_api_kwargs(
        self,
        input: Optional[Any] = None,
        model_kwargs: Dict = {},
        model_type: ModelType = ModelType.UNDEFINED,
    ) -> Dict:
        final_model_kwargs = model_kwargs.copy()
        if model_type == ModelType.LLM:
            if input is not None and input != "":
                messages = self._parse_template_to_messages(input)
            else:
                # If no input, create a minimal user message
                messages = [{"role": "user", "content": "Hello"}]
            final_model_kwargs["messages"] = messages
        else:
            raise ValueError(f"model_type {model_type} is not supported")
        return final_model_kwargs

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
        ),
        max_time=5,
    )
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        assert "model" in api_kwargs, f"model must be specified in api_kwargs: {api_kwargs}"
        if model_type == ModelType.LLM:
            completion = self.sync_client.chat.completions.create(**api_kwargs)
            return completion
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @backoff.on_exception(
        backoff.expo,
        (
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            UnprocessableEntityError,
        ),
        max_time=5,
    )
    async def acall(
        self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED
    ):
        if self.async_client is None:
            self.async_client = self.init_async_client()
        assert "model" in api_kwargs, "model must be specified"
        if model_type == ModelType.LLM:
            completion = await self.async_client.chat.completions.create(**api_kwargs)
            return completion
        else:
            raise ValueError(f"model_type {model_type} is not supported")

    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        obj = super().from_dict(data)
        obj.sync_client = obj.init_sync_client()
        obj.async_client = obj.init_async_client()
        return obj

    def to_dict(self) -> Dict[str, Any]:
        exclude = ["sync_client", "async_client"]
        output = super().to_dict(exclude=exclude)
        return output

    def list_models(self):
        return self.sync_client.models.list()
