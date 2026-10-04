import asyncio
import logging
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar

from adalflow.core.model_client import ModelClient
from adalflow.core.types import CompletionUsage, GeneratorOutput, ModelType
from adalflow.utils import printc
from google import genai
from google.genai import types
from groq import AsyncGroq, Groq

T = TypeVar("T")
log = logging.getLogger(__name__)

# Fallback sequence of active, supported models across Gemini and Groq
DEFAULT_GENERATION_MODELS: List[Dict[str, str]] = [
    {"provider": "gemini", "model": "gemini-2.5-flash"},
    {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    {"provider": "gemini", "model": "gemini-2.0-flash"},
    {"provider": "groq", "model": "llama-3.1-8b-instant"},
    {"provider": "gemini", "model": "gemini-1.5-flash"},
    {"provider": "groq", "model": "mixtral-8x7b-32768"},
]


class UnifiedChatCompletion:
    """Universal chat completion envelope compatible with Groq/OpenAI choices schema."""

    def __init__(
        self,
        content: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        model: str = "",
        provider: str = "",
        raw: Any = None,
    ):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        self.model = model
        self.provider = provider
        self.raw = raw

    @property
    def text(self) -> str:
        return self.choices[0].message.content


class MultiProviderLLMClient(ModelClient):
    """Multi-provider LLM client with automatic fallback across Gemini and Groq models."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        models: Optional[List[Dict[str, str]]] = None,
        api_key: Optional[str] = None,  # Legacy alias for groq_api_key
    ):
        super().__init__()
        self._gemini_api_key = gemini_api_key
        self._groq_api_key = groq_api_key or api_key
        self.models = list(models) if models is not None else list(DEFAULT_GENERATION_MODELS)
        self._groq_sync: Optional[Groq] = None
        self._groq_async: Optional[AsyncGroq] = None
        self._gemini_sync: Optional[genai.Client] = None
        self._sync_client: Any = None

    @property
    def gemini_api_key(self) -> Optional[str]:
        return self._gemini_api_key or os.getenv("GEMINI_API_KEY")

    @property
    def groq_api_key(self) -> Optional[str]:
        return self._groq_api_key or os.getenv("GROQ_API_KEY")

    @property
    def groq_client(self) -> Groq:
        if self._groq_sync is None:
            if not self.groq_api_key:
                raise ValueError("GROQ_API_KEY must be set to use Groq models.")
            self._groq_sync = Groq(api_key=self.groq_api_key)
        return self._groq_sync

    @property
    def gemini_client(self) -> genai.Client:
        if self._gemini_sync is None:
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY must be set to use Gemini models.")
            self._gemini_sync = genai.Client(api_key=self.gemini_api_key)
        return self._gemini_sync

    def init_sync_client(self) -> Any:
        if self._sync_client is not None:
            return self._sync_client
        return self.groq_client if self.groq_api_key else (self.gemini_client if self.gemini_api_key else None)

    @property
    def sync_client(self) -> Any:
        return self.init_sync_client()

    @sync_client.setter
    def sync_client(self, value: Any):
        self._sync_client = value

    def init_async_client(self) -> Optional[AsyncGroq]:
        if self._groq_async is None and self.groq_api_key:
            self._groq_async = AsyncGroq(api_key=self.groq_api_key)
        return self._groq_async

    def _parse_template_to_messages(self, input_text: str) -> Sequence[Dict[str, str]]:
        """Parse template with <SYS> and <USER> tags into OpenAI/Gemini compatible messages."""
        messages: List[Dict[str, str]] = []

        sys_match = re.search(r'<SYS>(.*?)</SYS>', input_text, re.DOTALL)
        if sys_match:
            system_content = sys_match.group(1).strip()
            messages.append({"role": "system", "content": system_content})

        middle_match = re.search(r'</SYS>(.*?)<USER>', input_text, re.DOTALL)
        if middle_match:
            middle_content = middle_match.group(1).strip()
            if middle_content:
                if messages:
                    messages[0]["content"] += "\n\n" + middle_content
                else:
                    messages.append({"role": "system", "content": middle_content})

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

    def _format_gemini_contents(
        self, messages: Sequence[Dict[str, str]], model: str = ""
    ) -> Tuple[Optional[str], List[types.Content]]:
        """
        Format generic messages into Google Gemini system_instruction and Content objects.
        For Gemma models (gemma-4-31b-it, gemma-4-26b-a4b-it), merges system instruction into
        the user prompt because Gemma does not support separate system_instruction in config.
        """
        system_instruction: Optional[str] = None
        contents: List[types.Content] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                if system_instruction is None:
                    system_instruction = content
                else:
                    system_instruction += "\n\n" + content
            elif role in ("assistant", "model"):
                contents.append(
                    types.Content(role="model", parts=[types.Part.from_text(text=content)])
                )
            else:
                contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=content)])
                )

        if not contents:
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text="Please respond.")])
            )

        # Handle Gemma models: merge system instructions directly into first user message
        is_gemma = "gemma" in model.lower()
        if is_gemma and system_instruction:
            if contents and contents[0].role == "user" and contents[0].parts:
                orig_text = contents[0].parts[0].text or ""
                contents[0] = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"{system_instruction}\n\n{orig_text}")],
                )
            else:
                contents.insert(
                    0,
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=system_instruction)],
                    ),
                )
            system_instruction = None

        return system_instruction, contents

    def _call_gemini(
        self,
        model: str,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.3,
        **kwargs,
    ) -> UnifiedChatCompletion:
        """Call Google Gemini generate_content API with format and naming resilience."""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        system_instruction, contents = self._format_gemini_contents(messages, model=model)
        contents_input: Any = contents
        config_kwargs: Dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if temperature is not None:
            config_kwargs["temperature"] = float(temperature)

        config_obj = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        # Execute call with fallback for models/ prefix if 404
        try:
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=contents_input,
                config=config_obj,
            )
        except Exception as e:
            if ("404" in str(e) or "not found" in str(e).lower()) and not model.startswith("models/"):
                log.info(f"Gemini model {model} returned 404, retrying with models/{model}...")
                response = self.gemini_client.models.generate_content(
                    model=f"models/{model}",
                    contents=contents_input,
                    config=config_obj,
                )
            else:
                raise e

        text = getattr(response, "text", "") or ""
        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage_meta, "total_token_count", 0) or (prompt_tokens + completion_tokens)

        return UnifiedChatCompletion(
            content=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            provider="gemini",
            raw=response,
        )

    def _call_groq(
        self,
        model: str,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.3,
        **kwargs,
    ) -> UnifiedChatCompletion:
        """Call Groq chat.completions.create API with role compatibility fallback."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        groq_messages = []
        for m in messages:
            role = m.get("role", "user")
            if role not in ("system", "user", "assistant"):
                role = "user"
            groq_messages.append({"role": role, "content": m.get("content", "")})

        call_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": groq_messages,
            "temperature": float(temperature) if temperature is not None else 0.3,
        }
        if "max_tokens" in kwargs:
            call_kwargs["max_tokens"] = kwargs["max_tokens"]

        try:
            completion = self.groq_client.chat.completions.create(**call_kwargs)
        except Exception as e:
            err_str = str(e).lower()
            # If the model does not accept system messages, merge system into first user message
            if "system" in err_str and any(x in err_str for x in ["not supported", "role", "invalid"]):
                merged_messages = []
                system_text = ""
                for m in groq_messages:
                    if m["role"] == "system":
                        system_text = f"{system_text}\n\n{m['content']}".strip()
                    else:
                        merged_messages.append(dict(m))
                if merged_messages and system_text:
                    merged_messages[0]["content"] = f"{system_text}\n\n{merged_messages[0]['content']}"
                elif system_text:
                    merged_messages.append({"role": "user", "content": system_text})
                call_kwargs["messages"] = merged_messages
                completion = self.groq_client.chat.completions.create(**call_kwargs)
            else:
                raise e

        content = ""
        if completion.choices and completion.choices[0].message:
            content = completion.choices[0].message.content or ""

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

        return UnifiedChatCompletion(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            provider="groq",
            raw=completion,
        )

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
                messages = [{"role": "user", "content": "Hello"}]
            final_model_kwargs["messages"] = messages
        else:
            raise ValueError(f"model_type {model_type} is not supported. This client only supports LLM.")
        return final_model_kwargs

    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """
        Execute generation by cycling through configured models across Gemini and Groq.
        If any model encounters an error or rate limit, it falls back to the next model.
        """
        if model_type != ModelType.LLM:
            raise ValueError(f"model_type {model_type} is not supported. Only LLM is supported.")

        messages = api_kwargs.get("messages")
        if not messages:
            input_text = api_kwargs.get("input")
            if input_text:
                messages = self._parse_template_to_messages(input_text)
            else:
                messages = [{"role": "user", "content": "Hello"}]

        temperature = api_kwargs.get("temperature", 0.3)

        candidate_entries = list(self.models)
        requested_model = api_kwargs.get("model")
        if requested_model:
            specific_entry = None
            for m in candidate_entries:
                if m.get("model") == requested_model:
                    specific_entry = m
                    break
            if specific_entry:
                candidate_entries.remove(specific_entry)
                candidate_entries.insert(0, specific_entry)
            else:
                prov = "gemini" if ("gemini" in requested_model.lower() or "gemma" in requested_model.lower()) else "groq"
                candidate_entries.insert(0, {"provider": prov, "model": requested_model})

        errors = []
        for idx, entry in enumerate(candidate_entries):
            provider = entry.get("provider", "").lower()
            model = entry.get("model", "")

            # Verify credentials exist for the provider
            if provider == "gemini" and not self.gemini_api_key:
                errors.append(f"gemini:{model} -> GEMINI_API_KEY not configured, skipping")
                continue
            if provider == "groq" and not self.groq_api_key:
                errors.append(f"groq:{model} -> GROQ_API_KEY not configured, skipping")
                continue

            printc(
                f"MultiProviderLLMClient: [{idx + 1}/{len(candidate_entries)}] Generating with {provider}:{model}...",
                color="blue",
            )
            try:
                if provider == "gemini":
                    completion = self._call_gemini(model=model, messages=messages, temperature=temperature)
                elif provider == "groq":
                    completion = self._call_groq(model=model, messages=messages, temperature=temperature)
                else:
                    log.warning(f"Unknown provider '{provider}', skipping...")
                    continue

                printc(
                    f"MultiProviderLLMClient: Successfully generated response using {provider}:{model}.",
                    color="green",
                )
                return completion
            except Exception as e:
                err_msg = f"{provider}:{model} -> {e}"
                errors.append(err_msg)
                printc(
                    f"MultiProviderLLMClient: Model '{provider}:{model}' failed ({e}). Falling back to next model...",
                    color="yellow",
                )

        error_summary = "\n".join(f"  - {err}" for err in errors)
        log.error(f"MultiProviderLLMClient: All models in fallback chain failed:\n{error_summary}")
        raise RuntimeError(f"All models in fallback chain failed:\n{error_summary}")

    def _stream_gemini(
        self,
        model: str,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """Stream chunks from Google Gemini generate_content_stream API."""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        system_instruction, contents = self._format_gemini_contents(messages, model=model)
        contents_input: Any = contents
        config_kwargs: Dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if temperature is not None:
            config_kwargs["temperature"] = float(temperature)

        config_obj = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        try:
            response_stream = self.gemini_client.models.generate_content_stream(
                model=model,
                contents=contents_input,
                config=config_obj,
            )
        except Exception as e:
            if ("404" in str(e) or "not found" in str(e).lower()) and not model.startswith("models/"):
                response_stream = self.gemini_client.models.generate_content_stream(
                    model=f"models/{model}",
                    contents=contents_input,
                    config=config_obj,
                )
            else:
                raise e

        for chunk in response_stream:
            text = getattr(chunk, "text", "") or ""
            if text:
                yield text

    def _stream_groq(
        self,
        model: str,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """Stream chunks from Groq chat.completions.create(stream=True) API."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        groq_messages = []
        for m in messages:
            role = m.get("role", "user")
            if role not in ("system", "user", "assistant"):
                role = "user"
            groq_messages.append({"role": role, "content": m.get("content", "")})

        call_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": groq_messages,
            "temperature": float(temperature) if temperature is not None else 0.3,
            "stream": True,
        }

        try:
            stream = self.groq_client.chat.completions.create(**call_kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if "system" in err_str and any(x in err_str for x in ["not supported", "role", "invalid"]):
                merged_messages = []
                system_text = ""
                for m in groq_messages:
                    if m["role"] == "system":
                        system_text = f"{system_text}\n\n{m['content']}".strip()
                    else:
                        merged_messages.append(dict(m))
                if merged_messages and system_text:
                    merged_messages[0]["content"] = f"{system_text}\n\n{merged_messages[0]['content']}"
                elif system_text:
                    merged_messages.append({"role": "user", "content": system_text})
                call_kwargs["messages"] = merged_messages
                stream = self.groq_client.chat.completions.create(**call_kwargs)
            else:
                raise e

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def stream_call(self, api_kwargs: Dict = {}) -> Iterator[str]:
        """Stream generated text chunks using the multi-provider fallback chain."""
        messages = api_kwargs.get("messages") or api_kwargs.get("model_kwargs", {}).get("messages")
        if not messages:
            input_text = api_kwargs.get("input") or api_kwargs.get("model_kwargs", {}).get("input")
            if input_text:
                messages = [{"role": "user", "content": str(input_text)}]
            else:
                raise ValueError("No messages or input provided to stream_call.")

        temperature = api_kwargs.get("temperature", 0.3)
        candidate_entries = list(self.models)

        errors = []
        for idx, entry in enumerate(candidate_entries):
            provider = entry.get("provider", "").lower()
            model = entry.get("model", "")

            if provider == "gemini" and not self.gemini_api_key:
                errors.append(f"gemini:{model} -> GEMINI_API_KEY not configured")
                continue
            if provider == "groq" and not self.groq_api_key:
                errors.append(f"groq:{model} -> GROQ_API_KEY not configured")
                continue

            printc(f"MultiProviderLLMClient: Streaming with {provider}:{model}...", color="blue")
            try:
                if provider == "gemini":
                    yield from self._stream_gemini(model=model, messages=messages, temperature=temperature)
                    return
                elif provider == "groq":
                    yield from self._stream_groq(model=model, messages=messages, temperature=temperature)
                    return
            except Exception as e:
                err_msg = f"{provider}:{model} -> {e}"
                errors.append(err_msg)
                printc(
                    f"MultiProviderLLMClient: Streaming with '{provider}:{model}' failed ({e}). Trying next...",
                    color="yellow",
                )

        printc("MultiProviderLLMClient: All streaming attempts failed, falling back to sync call...", color="yellow")
        comp = self.call(api_kwargs)
        if hasattr(comp, "choices") and comp.choices:
            yield comp.choices[0].message.content or ""
        elif hasattr(comp, "text"):
            yield comp.text or ""
        else:
            yield str(comp)

    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED) -> Any:  # type: ignore[override]
        """Async fallback generation."""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return await loop.run_in_executor(None, self.call, api_kwargs, model_type)
        except Exception as e:
            log.error(f"MultiProviderLLMClient: Error in acall execution: {e}")
            raise

    def parse_chat_completion(self, completion: Any) -> GeneratorOutput:
        """Parse completion output into GeneratorOutput."""
        try:
            if hasattr(completion, "choices") and completion.choices:
                data = completion.choices[0].message.content
            elif hasattr(completion, "text"):
                data = completion.text
            elif isinstance(completion, dict) and "choices" in completion:
                data = completion["choices"][0]["message"]["content"]
            else:
                data = str(completion)

            usage = self.track_completion_usage(completion)
            return GeneratorOutput(data=None, usage=usage, raw_response=data)
        except Exception as e:
            log.error(f"Error parsing completion: {e}")
            return GeneratorOutput(data=str(e), usage=None, raw_response=str(completion))

    def track_completion_usage(self, completion: Any) -> CompletionUsage:
        try:
            usage = getattr(completion, "usage", None)
            if usage:
                return CompletionUsage(
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                )
            usage_meta = getattr(completion, "usage_metadata", None)
            if usage_meta:
                prompt = getattr(usage_meta, "prompt_token_count", 0) or 0
                comp = getattr(usage_meta, "candidates_token_count", 0) or 0
                tot = getattr(usage_meta, "total_token_count", 0) or (prompt + comp)
                return CompletionUsage(
                    completion_tokens=comp,
                    prompt_tokens=prompt,
                    total_tokens=tot,
                )
            return CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)
        except Exception as e:
            log.warning(f"Error tracking completion usage: {e}")
            return CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0)

    @classmethod
    def from_dict(cls: type[T], data: Dict[str, Any]) -> T:
        try:
            obj = super().from_dict(data)
            return obj
        except Exception as e:
            log.error(f"MultiProviderLLMClient: Error in from_dict: {e}")
            raise

    def to_dict(self, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        try:
            exclude_list = list(exclude) if exclude is not None else []
            for key in ["_groq_sync", "_groq_async", "_gemini_sync", "_sync_client"]:
                if key not in exclude_list:
                    exclude_list.append(key)
            output = super().to_dict(exclude=exclude_list)
            return output
        except Exception as e:
            log.error(f"MultiProviderLLMClient: Error in to_dict: {e}")
            raise

    def list_models(self):
        try:
            return self.groq_client.models.list()
        except Exception as e:
            log.warning(f"Failed to query remote models, returning configured defaults: {e}")
            return self.models


# Backwards compatibility alias
CustomGroqClient = MultiProviderLLMClient

__all__ = [
    "MultiProviderLLMClient",
    "CustomGroqClient",
    "UnifiedChatCompletion",
    "DEFAULT_GENERATION_MODELS",
]
