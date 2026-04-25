from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "offline": {
        "kind": "offline",
        "network": False,
        "description": "Deterministic local fallback. No network calls and no API key.",
    },
    "ollama": {
        "kind": "openai-compatible",
        "network": True,
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:14b",
        "api_key_env": None,
        "description": "Local Ollama server using its OpenAI-compatible endpoint.",
    },
    "openrouter": {
        "kind": "openai-compatible",
        "network": True,
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "api_key_env": "OPENROUTER_API_KEY",
        "headers": {
            "HTTP-Referer": "https://github.com/jmaxwell2000/ThreadSieve",
            "X-Title": "ThreadSieve",
        },
        "description": "OpenRouter hosted model routing through its OpenAI-style chat completions API.",
    },
    "openai": {
        "kind": "openai-compatible",
        "network": True,
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "description": "OpenAI API through the OpenAI-compatible chat completions path.",
    },
    "openai-compatible": {
        "kind": "openai-compatible",
        "network": True,
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:14b",
        "api_key_env": "THREADSIEVE_API_KEY",
        "description": "Custom OpenAI-compatible endpoint such as LM Studio, vLLM, llama.cpp, or a self-hosted gateway.",
    },
}


@dataclass(frozen=True)
class Provider:
    name: str
    kind: str
    base_url: str | None
    model: str | None
    api_key_env: str | None
    api_key: str | None
    headers: dict[str, str]
    timeout_seconds: float
    temperature: float
    network: bool
    description: str

    @property
    def chat_completions_url(self) -> str:
        if not self.base_url:
            raise RuntimeError(f"Provider {self.name} does not have a base_url.")
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)


def build_provider(model_config: dict[str, Any]) -> Provider:
    name = str(model_config.get("provider") or "offline").lower()
    preset = PROVIDER_PRESETS.get(name, PROVIDER_PRESETS["openai-compatible"])
    merged = merge_dicts(preset, model_config)
    kind = str(merged.get("kind") or ("offline" if name == "offline" else "openai-compatible"))
    api_key_env = merged.get("api_key_env")
    if api_key_env is not None:
        api_key_env = str(api_key_env)
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if not api_key:
        api_key = merged.get("api_key")
    headers = {str(key): str(value) for key, value in dict(merged.get("headers") or {}).items()}
    return Provider(
        name=name,
        kind=kind,
        base_url=str(merged.get("base_url")).rstrip("/") if merged.get("base_url") else None,
        model=str(merged.get("model")) if merged.get("model") else None,
        api_key_env=api_key_env,
        api_key=str(api_key) if api_key else None,
        headers=headers,
        timeout_seconds=float(merged.get("timeout_seconds", 120)),
        temperature=float(merged.get("temperature", 0.1)),
        network=bool(merged.get("network", kind != "offline")),
        description=str(merged.get("description") or ""),
    )


def provider_status(provider: Provider) -> dict[str, Any]:
    return {
        "provider": provider.name,
        "kind": provider.kind,
        "network": provider.network,
        "base_url": provider.base_url,
        "model": provider.model,
        "api_key_env": provider.api_key_env,
        "api_key_loaded": provider.has_api_key,
        "headers": sorted(provider.headers.keys()),
        "description": provider.description,
    }


def provider_request(provider: Provider, messages: list[dict[str, str]], response_format: dict[str, Any] | None = None) -> urllib.request.Request:
    if provider.kind != "openai-compatible":
        raise RuntimeError(f"Provider {provider.name} does not support chat completions.")
    if not provider.base_url or not provider.model:
        raise RuntimeError(f"Provider {provider.name} must include base_url and model.")
    if provider.api_key_env and not provider.api_key:
        raise RuntimeError(f"Provider {provider.name} requires ${provider.api_key_env}.")

    payload: dict[str, Any] = {
        "model": provider.model,
        "temperature": provider.temperature,
        "messages": messages,
    }
    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Content-Type": "application/json",
        **provider.headers,
    }
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    else:
        headers["Authorization"] = "Bearer no-key"

    return urllib.request.Request(
        provider.chat_completions_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def fetch_json(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Provider request failed with HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Provider request failed: {exc}") from exc


def response_message_content(response_data: dict[str, Any], operation: str = "provider request") -> str:
    """Return OpenAI-compatible message content or raise a useful provider error."""
    if isinstance(response_data.get("error"), dict):
        error = response_data["error"]
        message = error.get("message") or error.get("code") or json.dumps(error, ensure_ascii=False)
        raise RuntimeError(f"{operation} failed: {message}")

    choices = response_data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"{operation} returned no choices. Response preview: {response_preview(response_data)}")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"{operation} returned an invalid choice. Response preview: {response_preview(response_data)}")

    message = first.get("message")
    content: Any = None
    if isinstance(message, dict):
        content = message.get("content")
    if content is None:
        content = first.get("text")

    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        joined = "\n".join(parts).strip()
        if joined:
            return joined

    finish_reason = first.get("finish_reason")
    if finish_reason:
        raise RuntimeError(f"{operation} returned empty content with finish_reason={finish_reason}. Response preview: {response_preview(response_data)}")
    raise RuntimeError(f"{operation} returned no message content. Response preview: {response_preview(response_data)}")


def response_preview(response_data: dict[str, Any], limit: int = 800) -> str:
    try:
        text = json.dumps(response_data, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(response_data)
    return " ".join(text.split())[:limit]


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
