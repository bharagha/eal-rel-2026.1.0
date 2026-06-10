# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LitellmProvider."""

from unittest.mock import patch

import pytest

from src.config import ProviderConfig
from src.models import ChatCompletionMessage, ChatCompletionRequest, ChatCompletionRole
from src.providers.litellm_provider import LitellmProvider


@pytest.mark.unit
def test_litellm_provider_initialization(litellm_provider_config):
    """Adapter pulls name/type/model and full settings (endpoint, timeout, auth) from ProviderConfig."""
    provider = LitellmProvider(litellm_provider_config)

    assert provider.name == "test-vllm"
    assert provider.provider_type == "hosted_vllm"
    assert provider.model == "test-model"
    assert provider.endpoint == "http://localhost:8000/v1"
    assert provider.timeout == 10.0
    assert provider.auth_scheme == "none"
    assert provider.api_key is None
    assert provider.custom_headers == {}


@pytest.mark.unit
@pytest.mark.anyio("asyncio")
async def test_litellm_provider_chat_success(litellm_provider_config, mock_http_response):
    """chat() forwards to litellm.completion and adapts the raw dict response."""
    provider = LitellmProvider(litellm_provider_config)

    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatCompletionMessage(role=ChatCompletionRole.USER, content="Hello")
        ],
        temperature=0.7,
        max_tokens=100,
    )

    with patch("src.providers.litellm_provider.litellm.completion") as mock_completion:
        mock_completion.return_value = mock_http_response

        response = await provider.chat(request)

        assert response.model == "test-model"
        assert response.choices[0].message.content == "Hello, world!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

        mock_completion.assert_called_once()
        kwargs = mock_completion.call_args.kwargs
        # Composes "<provider_type>/<model>" for litellm
        assert kwargs["model"] == "hosted_vllm/test-model"
        assert kwargs["api_base"] == "http://localhost:8000/v1"
        assert kwargs["stream"] is False
        # Pass-through params survive
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100
        # Adapter strips reserved keys
        assert "messages" in kwargs and kwargs["messages"][0]["role"] == "user"
        # Configured timeout from settings is forwarded to litellm
        assert kwargs["timeout"] == 10.0
        # auth.scheme="none" → no api_key supplied (even though type≠openai)
        assert "api_key" not in kwargs


@pytest.mark.unit
@pytest.mark.anyio("asyncio")
async def test_litellm_provider_chat_stream_yields_chunks(litellm_provider_config):
    """chat_stream() iterates litellm's iterator and adapts each chunk."""
    provider = LitellmProvider(litellm_provider_config)

    chunks_in = [
        {
            "id": "chatcmpl-x",
            "created": 111,
            "model": "test-model",
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-x",
            "created": 111,
            "model": "test-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        },
    ]

    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            ChatCompletionMessage(role=ChatCompletionRole.USER, content="hi")
        ],
    )

    with patch("src.providers.litellm_provider.litellm.completion") as mock_completion:
        mock_completion.return_value = iter(chunks_in)

        results = []
        async for chunk in provider.chat_stream(request):
            results.append(chunk)

        assert len(results) == 2
        assert results[0].choices[0]["delta"]["content"] == "Hi"
        # Final chunk preserves usage via extra="allow"
        dumped = results[1].model_dump()
        assert dumped["usage"]["prompt_tokens"] == 4
        assert dumped["usage"]["completion_tokens"] == 1

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["stream"] is True
        # Streaming defaults stream_options.include_usage so usage flows through
        assert kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.unit
@pytest.mark.anyio("asyncio")
async def test_litellm_provider_list_models(litellm_provider_config):
    """list_models() reports the configured model id under this provider name."""
    provider = LitellmProvider(litellm_provider_config)

    models = await provider.list_models()
    assert len(models) == 1
    assert models[0]["id"] == "test-model"
    assert models[0]["owned_by"] == "test-vllm"


@pytest.mark.unit
@pytest.mark.anyio("asyncio")
async def test_litellm_provider_health_check_success(litellm_provider_config):
    """health_check() returns True when list_models works."""
    provider = LitellmProvider(litellm_provider_config)
    assert await provider.health_check() is True


@pytest.mark.unit
@pytest.mark.anyio("asyncio")
async def test_litellm_provider_custom_headers_and_api_key_forwarded():
    """auth.api_key + custom_headers from config are forwarded to litellm."""
    cfg = ProviderConfig(
        name="cloud-openai",
        type="openai",
        model="gpt-4o-mini",
        enabled=True,
        settings={
            "endpoint": "https://api.openai.com/v1",
            "timeout": 60.0,
            "auth": {
                "scheme": "bearer",
                "api_key": "sk-secret",
                "custom_headers": {"X-Org": "acme"},
            },
        },
    )
    provider = LitellmProvider(cfg)

    request = ChatCompletionRequest(
        model="gpt-4o-mini",
        messages=[ChatCompletionMessage(role=ChatCompletionRole.USER, content="hi")],
    )

    with patch("src.providers.litellm_provider.litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "id": "x", "created": 0, "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        await provider.chat(request)

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["api_key"] == "sk-secret"
    assert kwargs["timeout"] == 60.0
    assert kwargs["extra_headers"] == {"X-Org": "acme"}


@pytest.mark.unit
@pytest.mark.anyio("asyncio")
async def test_litellm_provider_openai_with_auth_none_skips_fake_key():
    """auth.scheme='none' suppresses the OpenAI placeholder api_key fallback."""
    cfg = ProviderConfig(
        name="openai-noauth",
        type="openai",
        model="local-model",
        enabled=True,
        settings={
            "endpoint": "http://localhost:5000/v1",
            "auth": {"scheme": "none", "api_key": None},
        },
    )
    provider = LitellmProvider(cfg)

    request = ChatCompletionRequest(
        model="local-model",
        messages=[ChatCompletionMessage(role=ChatCompletionRole.USER, content="hi")],
    )

    with patch("src.providers.litellm_provider.litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "id": "x", "created": 0, "model": "local-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        await provider.chat(request)

    kwargs = mock_completion.call_args.kwargs
    assert "api_key" not in kwargs


@pytest.mark.unit
@pytest.mark.anyio("asyncio")
async def test_litellm_provider_openai_type_supplies_fake_key():
    """For ``type: openai`` without auth, adapter supplies a placeholder key."""
    cfg = ProviderConfig(
        name="openai-local",
        type="openai",
        model="local-model",
        enabled=True,
        settings={"endpoint": "http://localhost:5000/v1"},
    )
    provider = LitellmProvider(cfg)

    request = ChatCompletionRequest(
        model="local-model",
        messages=[
            ChatCompletionMessage(role=ChatCompletionRole.USER, content="hello")
        ],
    )

    with patch("src.providers.litellm_provider.litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "id": "x",
            "created": 0,
            "model": "local-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        await provider.chat(request)

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["api_key"] == "fake"
