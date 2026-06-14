"""
OpenRouter LLM client for text classification, analysis, and embeddings.

Provides a unified interface for LLM calls via OpenRouter API.
Supports Zero Data Retention (ZDR) for privacy-sensitive data.
"""

import os
import json
import time
import asyncio
from typing import Optional, Any, Awaitable, Callable, TypeVar
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI, LengthFinishReasonError
import httpx
from pydantic import ValidationError

from lib.config import OPENROUTER_BASE_URL, DEFAULT_MODELS
from lib.model_naming import resolve_remote_concurrency_default, resolve_temperature

STRUCTURED_OUTPUT_MAX_ATTEMPTS = 3
STRUCTURED_OUTPUT_RETRY_BASE_DELAY_SECONDS = 1.0
VALIDATION_REPAIR_MAX_ATTEMPTS = 4

T = TypeVar("T")


@dataclass
class LLMResponse:
    """Structured response from LLM call."""
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    raw_response: Any = None


@dataclass
class EmbeddingResponse:
    """Structured response from embedding call."""
    embeddings: list[list[float]]
    model: str
    usage: dict = field(default_factory=dict)


class OpenRouterClient:
    """
    Client for OpenRouter API calls.

    Usage:
        client = OpenRouterClient()
        response = client.complete("What is the meaning of life?")
        print(response.content)

        # With Zero Data Retention for sensitive data
        client = OpenRouterClient(zero_data_retention=True)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        zero_data_retention: bool = True,  # Default to ZDR for medical data
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key. If not provided, loads from .env
            default_model: Default model to use. Falls back to config default.
            zero_data_retention: Enable Zero Data Retention policy (recommended for medical data)
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
                "Set it in .env or pass directly."
            )

        self.default_model = default_model or DEFAULT_MODELS["classification"]
        self.zero_data_retention = zero_data_retention
        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
        )

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
    ) -> LLMResponse:
        """
        Send a completion request to the LLM.

        Args:
            prompt: User prompt/message
            model: Model to use (defaults to self.default_model)
            system_prompt: Optional system message
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            LLMResponse with content and metadata.
        """
        model = model or self.default_model
        temperature = resolve_temperature(model, temperature)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        # Add Zero Data Retention if enabled
        if self.zero_data_retention:
            kwargs["extra_body"] = {"provider": {"zdr": True}}

        response = self.client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw_response=response,
        )

    def classify_text(
        self,
        text: str,
        categories: list[str],
        category_descriptions: Optional[dict[str, str]] = None,
        model: Optional[str] = None,
        examples: Optional[list[dict]] = None,
    ) -> dict:
        """
        Classify text into one of the given categories.

        Args:
            text: Text to classify
            categories: List of category names
            category_descriptions: Optional dict mapping category -> description
            model: Model to use
            examples: Optional few-shot examples [{"text": ..., "category": ...}]

        Returns:
            Dict with 'category', 'confidence', and 'reasoning'.
        """
        # Build category description
        cat_desc = "\n".join([
            f"- {cat}: {category_descriptions.get(cat, 'No description')}"
            if category_descriptions else f"- {cat}"
            for cat in categories
        ])

        # Build examples section
        examples_text = ""
        if examples:
            examples_text = "\n\nExamples:\n" + "\n".join([
                f'Text: "{ex["text"]}"\nCategory: {ex["category"]}'
                for ex in examples
            ])

        system_prompt = f"""You are a medical text classifier. Classify the given text into exactly one of these categories:

{cat_desc}

Respond in JSON format:
{{"category": "<category_name>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}
{examples_text}"""

        response = self.complete(
            prompt=f'Classify this text: "{text}"',
            model=model,
            system_prompt=system_prompt,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "category": "error",
                "confidence": 0.0,
                "reasoning": f"Failed to parse response: {response.content}",
            }

    def batch_classify(
        self,
        texts: list[str],
        categories: list[str],
        category_descriptions: Optional[dict[str, str]] = None,
        model: Optional[str] = None,
        delay_seconds: float = 0.1,
        progress_callback: Optional[callable] = None,
    ) -> list[dict]:
        """
        Classify multiple texts with rate limiting.

        Args:
            texts: List of texts to classify
            categories: List of category names
            category_descriptions: Optional category descriptions
            model: Model to use
            delay_seconds: Delay between API calls
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of classification results.
        """
        results = []
        total = len(texts)

        for i, text in enumerate(texts):
            result = self.classify_text(
                text=text,
                categories=categories,
                category_descriptions=category_descriptions,
                model=model,
            )
            result["text"] = text
            result["index"] = i
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

            if i < total - 1:  # Don't delay after last item
                time.sleep(delay_seconds)

        return results

    def embed(
        self,
        texts: list[str] | str,
        model: str = "openai/text-embedding-3-large",
    ) -> EmbeddingResponse:
        """
        Generate embeddings for texts using OpenRouter's embedding API.

        Args:
            texts: Single text or list of texts to embed
            model: Embedding model to use (default: text-embedding-3-large)

        Returns:
            EmbeddingResponse with embeddings and metadata.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Use httpx directly for embeddings since OpenAI SDK may not support
        # the provider parameter correctly for embeddings
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "input": texts,
        }

        # Add Zero Data Retention if enabled
        if self.zero_data_retention:
            payload["provider"] = {"zdr": True}

        response = httpx.post(
            f"{OPENROUTER_BASE_URL}/embeddings",
            headers=headers,
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]

        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            usage=data.get("usage", {}),
        )

    def batch_embed(
        self,
        texts: list[str],
        model: str = "openai/text-embedding-3-large",
        batch_size: int = 100,
        progress_callback: Optional[callable] = None,
    ) -> list[list[float]]:
        """
        Generate embeddings for many texts with batching.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            batch_size: Number of texts per API call
            progress_callback: Optional callback(current, total) for progress

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            response = self.embed(batch, model=model)
            all_embeddings.extend(response.embeddings)

            if progress_callback:
                progress_callback(min(i + batch_size, total), total)

        return all_embeddings


def extract_pydantic_json(text: str, model_cls):
    """Extract and validate a Pydantic model from LLM text response.

    Handles thinking tags, markdown code blocks, and multiple JSON objects.
    Tries code blocks first, then scans for top-level JSON objects.
    """
    import re as _re
    # Strip thinking tags
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
    # 1) Try code blocks first (most reliable)
    for block in _re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=_re.DOTALL):
        try:
            return model_cls.model_validate_json(block)
        except Exception:
            continue
    # 2) Find all top-level JSON objects and try each
    for m in _re.finditer(r"\{", text):
        start = m.start()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return model_cls.model_validate_json(candidate)
                    except Exception:
                        break
    raise ValueError(f"Could not extract valid JSON from response. Response (first 500): {text[:500]}")


def make_openai_client(base_url: str | None = None, api_key: str | None = None, async_client: bool = True):
    """Create an OpenAI client, either for OpenRouter or a local server.

    Args:
        base_url: Custom base URL (e.g. http://192.168.0.70:1234/v1 for LM Studio).
                  If None, uses OpenRouter.
        api_key: API key. For local servers defaults to "lm-studio".
        async_client: If True return AsyncOpenAI, else OpenAI.

    Returns:
        (client, is_local) tuple.
    """
    ClientCls = AsyncOpenAI if async_client else OpenAI
    if base_url:
        return ClientCls(
            api_key=api_key or "lm-studio",
            base_url=base_url,
            timeout=httpx.Timeout(None),
        ), True
    load_dotenv()
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    return ClientCls(api_key=key, base_url=OPENROUTER_BASE_URL), False


def resolve_concurrency(
    concurrency: int | None,
    is_local: bool,
    model: str | None = None,
    remote_default: int | None = None,
    local_default: int = 1,
) -> int:
    """Resolve concurrency defaults based on provider type."""
    if concurrency is not None:
        return concurrency
    if is_local:
        return local_default
    if remote_default is not None:
        return remote_default
    return resolve_remote_concurrency_default(model)


def _format_retry_error(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    if len(message) > 240:
        message = f"{message[:237]}..."
    return message


def _is_retryable_structured_output_error(exc: Exception) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, LengthFinishReasonError, ValidationError, json.JSONDecodeError)):
        return True

    if isinstance(exc, ValueError):
        message = str(exc).lower()
        markers = (
            "could not parse response content",
            "could not extract valid json",
            "no parsed content",
            "empty response from model",
            "length limit was reached",
        )
        return any(marker in message for marker in markers)

    return False


@dataclass(frozen=True)
class ValidationRetryContext:
    """Repair context passed into a second-shot retry attempt."""
    validation_error: str
    failed_response: str | None = None


class ValidationRetryError(ValueError):
    """Validation error that carries a parsed response for repair retries."""

    def __init__(self, message: str, *, failed_result: Any | None = None):
        super().__init__(message)
        self.failed_result = failed_result


def serialize_failed_response(result: Any) -> str:
    """Best-effort JSON serialization for a parsed structured response."""
    if hasattr(result, "model_dump_json"):
        return result.model_dump_json(indent=2)
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
    return json.dumps(result, indent=2, ensure_ascii=False, default=str)


def build_validation_feedback_user_message(repair_context: ValidationRetryContext) -> str:
    """Build a user-message repair instruction for a failed structured response."""
    if repair_context.failed_response:
        message = [
            "Your previous response was parsed, but it failed validation.",
            f"Validation error: {repair_context.validation_error}",
        ]
    else:
        message = [
            "Your previous attempt failed. Return the full corrected response for the same request.",
            f"Error to fix: {repair_context.validation_error}",
        ]
    if repair_context.failed_response:
        message.extend(
            [
                "Here is your previous structured response. Return the full corrected response for the same request.",
                "```json",
                repair_context.failed_response,
                "```",
            ]
        )
    message.append(
        "Do not omit items. Keep identifiers exactly as requested."
    )
    return "\n\n".join(message)


def append_validation_feedback_messages(
    messages: list[dict[str, str]],
    repair_context: ValidationRetryContext | None,
) -> list[dict[str, str]]:
    """Append repair feedback as a final user message when a retry is needed."""
    if repair_context is None:
        return list(messages)
    updated = list(messages)
    updated.append(
        {
            "role": "user",
            "content": build_validation_feedback_user_message(repair_context),
        }
    )
    return updated


def append_validation_feedback_prompt(
    prompt: str,
    repair_context: ValidationRetryContext | None,
) -> str:
    """Append repair feedback to a prompt-only local call."""
    if repair_context is None:
        return prompt
    return f"{prompt}\n\n{build_validation_feedback_user_message(repair_context)}"


async def run_with_validation_repair(
    *,
    model: str,
    operation_label: str,
    run_attempt: Callable[[ValidationRetryContext | None], Awaitable[T]],
    max_attempts: int = VALIDATION_REPAIR_MAX_ATTEMPTS,
    retry_base_delay_seconds: float = 1.0,
    serialize_result: Callable[[Any], str] = serialize_failed_response,
    repair_retry_count: int = 2,
) -> T:
    """Run with retries, using repair-context only on the last retry attempts."""
    repair_context: ValidationRetryContext | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            use_repair_context = repair_context if attempt > max_attempts - repair_retry_count else None
            return await run_attempt(use_repair_context)
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_attempts:
                raise

            failed_result = getattr(exc, "failed_result", None)
            failed_response = None
            if failed_result is not None:
                try:
                    failed_response = serialize_result(failed_result)
                except Exception:  # noqa: BLE001
                    failed_response = None

            repair_context = ValidationRetryContext(
                validation_error=_format_retry_error(exc),
                failed_response=failed_response,
            )
            delay = retry_base_delay_seconds * attempt
            print(
                f"Batch retry {attempt}/{max_attempts - 1} for {model} "
                f"{operation_label} after {type(exc).__name__}: {_format_retry_error(exc)}"
            )
            await asyncio.sleep(delay)


async def structured_parse_call(
    client,
    *,
    model: str,
    messages: list[dict[str, str]],
    response_format,
    extra_body: dict | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    reasoning_effort: str | None = None,
    max_attempts: int = STRUCTURED_OUTPUT_MAX_ATTEMPTS,
    retry_base_delay_seconds: float = STRUCTURED_OUTPUT_RETRY_BASE_DELAY_SECONDS,
    request_timeout_seconds: float | None = None,
    max_tokens: int | None = 25000,
):
    """Call structured parsing with retries for length and parse failures."""
    temperature = resolve_temperature(model, temperature)

    for attempt in range(1, max_attempts + 1):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "response_format": response_format,
                "extra_body": extra_body,
                "temperature": temperature,
            }
            if top_p is not None:
                kwargs["top_p"] = top_p
            if presence_penalty is not None:
                kwargs["presence_penalty"] = presence_penalty
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            completion_coro = client.beta.chat.completions.parse(**kwargs)
            if request_timeout_seconds is not None:
                completion = await asyncio.wait_for(completion_coro, timeout=request_timeout_seconds)
            else:
                completion = await completion_coro
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Structured parse returned no parsed content")
            return parsed
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_attempts or not _is_retryable_structured_output_error(exc):
                raise

            delay = retry_base_delay_seconds * attempt
            print(
                f"Structured parse retry {attempt}/{max_attempts - 1} for {model} "
                f"after {type(exc).__name__}: {_format_retry_error(exc)}"
            )
            await asyncio.sleep(delay)


def structured_parse_call_sync(
    client,
    *,
    model: str,
    messages: list[dict[str, str]],
    response_format,
    extra_body: dict | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    reasoning_effort: str | None = None,
    max_attempts: int = STRUCTURED_OUTPUT_MAX_ATTEMPTS,
    retry_base_delay_seconds: float = STRUCTURED_OUTPUT_RETRY_BASE_DELAY_SECONDS,
    request_timeout_seconds: float | None = None,
    max_tokens: int | None = 25000,
):
    """Synchronous structured parse call with retries for length and parse failures."""
    temperature = resolve_temperature(model, temperature)

    for attempt in range(1, max_attempts + 1):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "response_format": response_format,
                "extra_body": extra_body,
                "temperature": temperature,
            }
            if top_p is not None:
                kwargs["top_p"] = top_p
            if presence_penalty is not None:
                kwargs["presence_penalty"] = presence_penalty
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if request_timeout_seconds is not None:
                kwargs["timeout"] = request_timeout_seconds
            completion = client.beta.chat.completions.parse(**kwargs)
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Structured parse returned no parsed content")
            return parsed
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_attempts or not _is_retryable_structured_output_error(exc):
                raise

            delay = retry_base_delay_seconds * attempt
            print(
                f"Structured parse retry {attempt}/{max_attempts - 1} for {model} "
                f"after {type(exc).__name__}: {_format_retry_error(exc)}"
            )
            time.sleep(delay)


async def local_structured_call(
    client,
    model: str,
    prompt: str,
    response_model,
    schema_str: str | None = None,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    extra_body: dict | None = None,
    max_attempts: int = STRUCTURED_OUTPUT_MAX_ATTEMPTS,
):
    """Make an LLM call to a local model and parse structured JSON output.

    Appends the JSON schema to the prompt and extracts the response.
    Does NOT set temperature or extra_body — let the inference server handle those.
    """
    if schema_str is None:
        schema_str = json.dumps(response_model.model_json_schema(), indent=2)
    json_instruction = (
        "\n\nYou MUST respond with ONLY a JSON object (no markdown, no explanation) "
        f"matching this schema:\n{schema_str}"
    )
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt + json_instruction}],
    }
    temperature = resolve_temperature(model, temperature)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    for attempt in range(1, max_attempts + 1):
        try:
            completion = await client.chat.completions.create(**kwargs)
            raw = completion.choices[0].message.content or ""
            if not raw:
                raise ValueError(f"Empty response from model. finish_reason={completion.choices[0].finish_reason}")
            return extract_pydantic_json(raw, response_model)
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_attempts or not _is_retryable_structured_output_error(exc):
                raise

            delay = STRUCTURED_OUTPUT_RETRY_BASE_DELAY_SECONDS * attempt
            print(
                f"Local structured retry {attempt}/{max_attempts - 1} for {model} "
                f"after {type(exc).__name__}: {_format_retry_error(exc)}"
            )
            await asyncio.sleep(delay)


def local_structured_call_sync(
    client,
    model: str,
    prompt: str,
    response_model,
    schema_str: str | None = None,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    extra_body: dict | None = None,
    max_attempts: int = STRUCTURED_OUTPUT_MAX_ATTEMPTS,
):
    """Synchronous version of local_structured_call."""
    if schema_str is None:
        schema_str = json.dumps(response_model.model_json_schema(), indent=2)
    json_instruction = (
        "\n\nYou MUST respond with ONLY a JSON object (no markdown, no explanation) "
        f"matching this schema:\n{schema_str}"
    )
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt + json_instruction}],
    }
    temperature = resolve_temperature(model, temperature)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    for attempt in range(1, max_attempts + 1):
        try:
            completion = client.chat.completions.create(**kwargs)
            raw = completion.choices[0].message.content or ""
            if not raw:
                raise ValueError(f"Empty response from model. finish_reason={completion.choices[0].finish_reason}")
            return extract_pydantic_json(raw, response_model)
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_attempts or not _is_retryable_structured_output_error(exc):
                raise

            delay = STRUCTURED_OUTPUT_RETRY_BASE_DELAY_SECONDS * attempt
            print(
                f"Local structured retry {attempt}/{max_attempts - 1} for {model} "
                f"after {type(exc).__name__}: {_format_retry_error(exc)}"
            )
            time.sleep(delay)


def get_client(model: Optional[str] = None, zero_data_retention: bool = True) -> OpenRouterClient:
    """
    Factory function to get an OpenRouter client.

    Args:
        model: Default model to use
        zero_data_retention: Enable ZDR for medical/sensitive data (default: True)

    Returns:
        Configured OpenRouterClient instance.
    """
    return OpenRouterClient(default_model=model, zero_data_retention=zero_data_retention)


# =============================================================================
# Async Client for Batch Processing
# =============================================================================

import asyncio
from asyncio import Semaphore
from openai import AsyncOpenAI


@dataclass
class AsyncLLMResponse:
    """Response from async LLM call."""
    content: str
    success: bool
    usage: dict = field(default_factory=dict)
    error: Optional[str] = None


class AsyncOpenRouterClient:
    """
    Async OpenRouter client with concurrency and rate limiting.

    Usage:
        client = AsyncOpenRouterClient(max_concurrent=10, requests_per_minute=60)
        results = await client.process_batch(
            texts,
            prompt_fn=lambda t: f"Classify: {t}",
            model="openai/gpt-4.1",
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_concurrent: int = 10,
        requests_per_minute: int = 60,
        zero_data_retention: bool = True,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

        self.client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.api_key,
        )
        self.semaphore = Semaphore(max_concurrent)
        self.rpm_delay = 60.0 / requests_per_minute
        self.zdr = zero_data_retention
        self._last_request = 0.0

    async def _rate_limit(self):
        """Ensure minimum delay between requests."""
        now = asyncio.get_event_loop().time()
        wait = self._last_request + self.rpm_delay - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = asyncio.get_event_loop().time()

    async def complete(
        self,
        prompt: str,
        model: str = "openai/gpt-4.1",
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncLLMResponse:
        """Single completion with rate limiting."""
        async with self.semaphore:
            await self._rate_limit()
            temperature = resolve_temperature(model, temperature)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            extra_body = {}
            if self.zdr:
                extra_body["provider"] = {"zdr": True}

            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body if extra_body else None,
                )
                return AsyncLLMResponse(
                    content=response.choices[0].message.content,
                    success=True,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    },
                )
            except Exception as e:
                return AsyncLLMResponse(
                    content="",
                    success=False,
                    error=str(e),
                )

    async def process_batch(
        self,
        items: list,
        prompt_fn: callable,
        model: str = "openai/gpt-4.1",
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        desc: str = "Processing",
    ) -> list[AsyncLLMResponse]:
        """Process items in parallel with concurrency control."""
        from tqdm.asyncio import tqdm_asyncio

        async def process_item(item):
            prompt = prompt_fn(item)
            return await self.complete(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        tasks = [process_item(item) for item in items]
        return await tqdm_asyncio.gather(*tasks, desc=desc)
