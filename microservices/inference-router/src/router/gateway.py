#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Inference Router API Gateway

Provides OpenAI-compatible API interface
"""
# pyright: reportMissingImports=false

import asyncio
import logging
import time
import uuid
import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from src.config.base import RouterConfig
from src.config.loader import load_config
from src.exceptions import RoutingError
from src.models import (
    ChatCompletionChoice,
    ChatCompletionMessage as ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage as Usage,
)
from src.router.orchestrator import RouterOrchestrator

# Use new observability infrastructure
from ..observability import FileBasedTelemetry, InMemoryTelemetry, RequestCompletedEvent
from ..observability.telemetry import Telemetry as _TelemetryBase
from src.config.base import TelemetryBackendType

# Shared logging utilities
from .logging_utils import (
    is_verbose_enabled,
    is_verbose_full_enabled,
    log_to_gateway_file,
    log_verbose_response,
)

# Logger
logger = logging.getLogger("gateway")


# ==================== Tool Call Parsing ====================

def parse_tool_calls_from_content(content: str) -> tuple[str, list[dict] | None]:
    """
    FALLBACK: Parse tool calls from response content for models without native support.

    This is a workaround for models that embed tool calls in text content instead
    of using the standard OpenAI tool_calls field. Prefer native tool support.

    Supports multiple formats:
    - MiniMax XML: <minimax:tool_call><invoke name="..." .../></tool_call>
    - Hermes/Qwen: <function=name><parameter=key>value</parameter></function>
    - JSON tool_call: <tool_call>{"name": "...", "arguments": {...}}</tool_call>

    Returns:
        (cleaned_content, tool_calls) where tool_calls is None if no tools detected
    """
    import re
    import json

    if not content or not isinstance(content, str):
        return content or "", None

    tool_calls = []
    cleaned_content = content

    try:
        # Pattern 1: MiniMax XML format
        # <minimax:tool_call><invoke name="write" filename="..." content="..."/></tool_call>
        minimax_pattern = r'<minimax:tool_call>\s*<invoke\s+name="([^"]+)"([^/>]*)/>\s*</(?:minimax:)?tool_call>'

        for match in re.finditer(minimax_pattern, content, re.DOTALL):
            try:
                tool_name = match.group(1)
                attrs_str = match.group(2)

                # Parse attributes
                args = {}
                attr_pattern = r'(\w+)="([^"]*)"'
                for attr_match in re.finditer(attr_pattern, attrs_str):
                    args[attr_match.group(1)] = attr_match.group(2)

                # Create OpenAI-format tool call
                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args)
                    }
                }
                tool_calls.append(tool_call)

                # Remove from content
                cleaned_content = cleaned_content.replace(match.group(0), "", 1)
            except Exception as e:
                logger.warning(f"Failed to parse MiniMax tool call: {e}")
                continue

        # Pattern 2: Hermes/Qwen function call format
        # <function=exec><parameter=command>some command</parameter></function>
        # Optionally followed by </tool_call>
        hermes_pattern = r'<function=(\w+)>(.*?)</function>(?:\s*</tool_call>)?'

        for match in re.finditer(hermes_pattern, cleaned_content, re.DOTALL):
            try:
                tool_name = match.group(1)
                params_block = match.group(2)

                # Extract <parameter=key>value</parameter> pairs
                args = {}
                param_pattern = r'<parameter=(\w+)>(.*?)</parameter>'
                for param_match in re.finditer(param_pattern, params_block, re.DOTALL):
                    param_name = param_match.group(1)
                    param_value = param_match.group(2).strip()
                    args[param_name] = param_value

                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args)
                    }
                }
                tool_calls.append(tool_call)

                cleaned_content = cleaned_content.replace(match.group(0), "", 1)
                logger.debug(f"Parsed Hermes/Qwen pseudo tool call: function={tool_name}, args_keys={list(args.keys())}")
            except Exception as e:
                logger.warning(f"Failed to parse Hermes/Qwen tool call: {e}")
                continue

        # Pattern 3: JSON tool_call format
        # <tool_call>{"name": "exec", "arguments": {"command": "..."}}</tool_call>
        json_tc_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'

        for match in re.finditer(json_tc_pattern, cleaned_content, re.DOTALL):
            try:
                tc_data = json.loads(match.group(1))
                tool_name = tc_data.get("name") or tc_data.get("function", {}).get("name")
                if not tool_name:
                    continue

                arguments = tc_data.get("arguments", tc_data.get("parameters", {}))
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
                elif not isinstance(arguments, str):
                    arguments = json.dumps({})

                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                }
                tool_calls.append(tool_call)

                cleaned_content = cleaned_content.replace(match.group(0), "", 1)
                logger.debug(f"Parsed JSON pseudo tool call: function={tool_name}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse JSON tool call: {e}")
                continue

    except re.error as e:
        logger.warning(f"Regex error in tool call parsing: {e}")
        return content, None
    except Exception as e:
        logger.warning(f"Unexpected error in tool call parsing: {e}")
        return content, None

    cleaned_content = cleaned_content.strip()
    return cleaned_content, tool_calls if tool_calls else None


# ==================== Application ====================

app = FastAPI(
    title="Inference Router API",
    description="OpenAI-compatible API for intelligent inference routing",
    version="1.0.0",
)

# Global variables
orchestrator: Optional[RouterOrchestrator] = None
telemetry: Optional[_TelemetryBase] = None
config: Optional[RouterConfig] = None
verbose: bool = False
verbose_full: bool = False
log_dir: Optional[Path] = None

# Concurrency management (vLLM-inspired load tracking)
_active_requests: int = 0
_max_concurrency: int = 0  # 0 = unlimited


async def _concurrency_guard():
    """FastAPI dependency that tracks active requests and enforces max_concurrency.

    Inspired by vLLM's load_aware_call decorator. Uses a yield-dependency so the
    counter is decremented after the full response (including streaming bodies)
    has been sent.
    """
    global _active_requests

    if _max_concurrency > 0 and _active_requests >= _max_concurrency:
        logger.warning(
            f"Concurrency limit reached: active={_active_requests}, max={_max_concurrency}"
        )
        raise HTTPException(
            status_code=429,
            detail=f"Server busy: {_active_requests}/{_max_concurrency} concurrent requests. Retry later.",
        )

    _active_requests += 1
    try:
        yield
    finally:
        _active_requests -= 1


def setup_logging(config: RouterConfig):
    """Setup logging based on configuration"""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    level_name = (config.log_level or "info").lower()
    log_level = level_map.get(level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    logger.setLevel(log_level)
    logger.info(f"Logging initialized at {level_name.upper()} level")


async def initialize_router(config_path: str = "config.yaml"):
    """Initialize router and orchestrator from config file."""
    global orchestrator, telemetry, config, verbose, verbose_full, log_dir
    global _max_concurrency

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    verbose = is_verbose_enabled()
    verbose_full = is_verbose_full_enabled()
    # log_dir is set in startup_event

    config = load_config(str(config_file))

    # Setup logging based on config
    setup_logging(config)

    # Apply concurrency limit (CLI --max-concurrency is the single source, default 3)
    env_max = os.environ.get("GATEWAY_MAX_CONCURRENCY")
    _max_concurrency = int(env_max) if env_max else 3

    # Build the telemetry backend from config.
    # - backend: file → persist events to JSONL (path from config.file_path,
    #   or co-located with gateway logs / config.yaml as a fallback)
    # - backend: memory → in-process only, no disk artifact
    # - enabled: false → also use in-memory (events still queryable via /v1/stats
    #   for the lifetime of the process)
    telemetry_cfg = config.telemetry
    if telemetry_cfg.enabled and telemetry_cfg.backend == TelemetryBackendType.FILE:
        if telemetry_cfg.file_path:
            telemetry_path = Path(telemetry_cfg.file_path)
        else:
            telemetry_dir = log_dir if log_dir is not None else config_file.parent
            telemetry_path = telemetry_dir / "telemetry.jsonl"
        telemetry = FileBasedTelemetry(telemetry_path)
        logger.info(f"Telemetry backend: file ({telemetry_path})")
    else:
        telemetry = InMemoryTelemetry()
        if not telemetry_cfg.enabled:
            logger.info("Telemetry backend: memory (telemetry.enabled=false)")
        else:
            logger.info("Telemetry backend: memory")

    orchestrator = RouterOrchestrator(config, telemetry=telemetry)
    await orchestrator.initialize()

    msg = f"✅ Router initialized with config: {config_path}"
    print(msg)
    log_to_gateway_file(msg, log_dir)
    logger.info(f"Router initialized with config: {config_path}")

    provider_names = list(orchestrator.provider_map)
    msg = f"   - Providers: {len(provider_names)} ({', '.join(provider_names) or 'none'})"
    print(msg)
    log_to_gateway_file(msg, log_dir)
    logger.info(f"Providers: {len(provider_names)}")

    if isinstance(telemetry, FileBasedTelemetry):
        msg = f"   - Telemetry: file ({telemetry_path})"
    elif telemetry_cfg.enabled:
        msg = "   - Telemetry: memory"
    else:
        msg = "   - Telemetry: memory (disabled in config)"
    print(msg)
    log_to_gateway_file(msg, log_dir)

    msg = f"   - Max concurrency: {'unlimited' if _max_concurrency <= 0 else _max_concurrency}"
    print(msg)
    log_to_gateway_file(msg, log_dir)
    logger.info(f"Max concurrency: {'unlimited' if _max_concurrency <= 0 else _max_concurrency}")

    msg = f"   - Verbose logging: {verbose}"
    print(msg)
    log_to_gateway_file(msg, log_dir)
    logger.debug(f"Verbose logging: {verbose}")

    msg = f"   - Verbose full logging: {verbose_full}"
    print(msg)
    log_to_gateway_file(msg, log_dir)
    logger.debug(f"Verbose full logging: {verbose_full}")


@app.on_event("startup")
async def startup_event():
    """Initialize on application startup"""
    global log_dir
    try:
        log_base = os.getenv("GATEWAY_LOG_DIR")
        if log_base:
            log_dir = Path(log_base)
            log_dir.mkdir(parents=True, exist_ok=True)
            msg = f"📝 Gateway logs will be saved to: {log_dir}"
            print(msg)
            log_to_gateway_file(msg, log_dir)
            log_to_gateway_file("="*80, log_dir)
            log_to_gateway_file("Gateway Started", log_dir)
            log_to_gateway_file("="*80, log_dir)
        await initialize_router(os.getenv("GATEWAY_CONFIG", "config.yaml"))
    except Exception as e:
        msg = f"❌ Failed to initialize router: {e}"
        print(msg)
        log_to_gateway_file(msg, log_dir)
        logger.error(f"Failed to initialize router: {e}")
        raise


# ==================== Endpoints ====================

@app.get("/")
async def root():
    """Root path"""
    return {
        "name": "Inference Router API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "stats": "/stats",
        }
    }


@app.get("/health")
async def health_check():
    """Health check"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    return {
        "status": "healthy",
        "router": "initialized",
        "timestamp": int(time.time()),
        "concurrency": {
            "active_requests": _active_requests,
            "max_concurrency": _max_concurrency if _max_concurrency > 0 else "unlimited",
        },
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    if config is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    models = []

    # One entry per configured provider
    for prov in config.providers:
        if not prov.enabled:
            continue
        models.append({
            "id": prov.name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": prov.type,
        })

    # Router virtual model
    models.append({
        "id": "router",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "inference-router",
    })

    return {"object": "list", "data": models}


def _record_request_telemetry(
    *,
    request_id: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_latency_ms: float,
    start_time: float,
    first_token_time: float | None,
    provider_name: str | None,
    is_direct: bool,
    is_streaming: bool = True,
) -> bool:
    """Record telemetry for a completed request.

    Telemetry is bucketed by ``provider_name`` (the configured provider that
    handled the request). ``route_path`` is recorded as a diagnostic flag
    only — ``"direct"`` when the client picked the provider by name, or
    ``"routed"`` when DecisionEngine selected it.
    """
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return False

    if is_streaming:
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time is not None else None
        tpot_ms = None
        if completion_tokens > 0 and ttft_ms is not None:
            time_after_first_token = total_latency_ms - ttft_ms
            if time_after_first_token > 0:
                tpot_ms = time_after_first_token / completion_tokens
    else:
        ttft_ms = total_latency_ms
        tpot_ms = total_latency_ms / completion_tokens if completion_tokens > 0 else None

    telemetry.record_event(RequestCompletedEvent(
        request_id=request_id,
        route_path="direct" if is_direct else "routed",
        provider_name=provider_name or "unknown",
        models_used=[model_name],
        final_model=model_name,
        total_input_tokens=prompt_tokens,
        total_output_tokens=completion_tokens,
        total_latency_ms=total_latency_ms,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
    ))

    logger.debug(
        f"Recorded telemetry: request_id={request_id}, provider={provider_name}, "
        f"input={prompt_tokens}, output={completion_tokens}"
        + (f", ttft={ttft_ms:.2f}ms, tpot={tpot_ms:.4f}ms" if ttft_ms is not None and tpot_ms is not None else "")
    )
    return True


async def stream_chat_completions(request: ChatCompletionRequest, request_id: str = None):
    """Stream chat completions (generator function)"""
    import json

    if request_id is None:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if is_verbose_full_enabled(verbose_full):
        log_verbose_response("Raw streaming request", request.model_dump(), request_id, log_dir, verbose)

    start_time = time.time()

    try:
        # Ask the orchestrator to route. Routing failures (unknown model name,
        # no providers, etc.) become a clean SSE error before any chunks flow.
        try:
            chunk_iter, route_info = await orchestrator.chat_stream(request)
        except RoutingError as routing_err:
            logger.error(f"Routing failed: request_id={request_id}, error={routing_err}")
            error_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "error",
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"[ERROR] {routing_err}"},
                    "finish_reason": "error",
                }]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        is_direct = route_info.is_direct
        final_provider_name: str | None = route_info.provider_name
        routing_reason: str | None = route_info.reason

        completion_logged = False
        telemetry_recorded = False
        final_model_name: str | None = None
        first_token_time = None
        has_native_tool_calls = False
        accumulated_content = ""
        tool_calling_disabled = request.tool_choice == "none"

        async for raw_chunk in chunk_iter:
            chunk = raw_chunk.model_dump()

            if not final_model_name and "model" in chunk:
                final_model_name = chunk["model"]

            # Record telemetry the moment any chunk carries ``usage`` — it
            # may arrive on a normal content chunk (vLLM's final chunk) or as
            # a separate choices-empty chunk (OpenAI). We only record once.
            if "usage" in chunk and chunk["usage"] and not telemetry_recorded:
                try:
                    usage_data = chunk["usage"]
                    telemetry_recorded = _record_request_telemetry(
                        request_id=request_id,
                        model_name=final_model_name or final_provider_name or "unknown",
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_latency_ms=(time.time() - start_time) * 1000,
                        start_time=start_time,
                        first_token_time=first_token_time,
                        provider_name=final_provider_name,
                        is_direct=is_direct,
                    )
                except Exception as telemetry_error:
                    logger.warning(f"Telemetry recording failed: request_id={request_id}, error={str(telemetry_error)}")

            # Handle chunks with empty choices (e.g., usage-only chunks from
            # OpenAI when ``stream_options.include_usage`` is set).
            if not chunk.get("choices"):
                if "usage" in chunk:
                    usage_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": final_model_name or final_provider_name or "router",
                        "choices": [],
                        "usage": chunk["usage"],
                    }
                    yield f"data: {json.dumps(usage_chunk)}\n\n"
                    await asyncio.sleep(0)
                continue

            delta = chunk["choices"][0].get("delta", {})
            finish_reason = chunk["choices"][0].get("finish_reason")

            if first_token_time is None and (delta.get("content") or delta.get("tool_calls")):
                first_token_time = time.time()

            if not tool_calling_disabled and "tool_calls" in delta and delta["tool_calls"]:
                has_native_tool_calls = True

            # Accumulate content for fallback tool call detection (only if no native tool_calls and tool calling not disabled)
            if not tool_calling_disabled and not has_native_tool_calls and "content" in delta and delta["content"]:
                accumulated_content += delta["content"]

            # Strip tool_calls from delta when tool_choice="none"
            if tool_calling_disabled and "tool_calls" in delta:
                del delta["tool_calls"]

            response_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": final_model_name or final_provider_name or "router",
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }]
            }

            if finish_reason and not tool_calling_disabled and not has_native_tool_calls and accumulated_content:
                try:
                    _, tool_calls = parse_tool_calls_from_content(accumulated_content)
                    if tool_calls:
                        response_chunk["choices"][0]["finish_reason"] = "tool_calls"
                        response_chunk["choices"][0]["delta"]["tool_calls"] = tool_calls
                except Exception as parse_error:
                    logger.warning(f"Tool call parsing failed: request_id={request_id}, error={str(parse_error)}")


            if "usage" in chunk:
                response_chunk["usage"] = chunk["usage"]

            if finish_reason and not completion_logged:
                total_latency_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"Streaming request completed: request_id={request_id}, "
                    f"provider={final_provider_name or 'unknown'}, model={final_model_name or 'unknown'}, "
                    f"reason={routing_reason or 'unknown'}, latency={total_latency_ms:.2f}ms"
                )
                completion_logged = True

            log_verbose_response("Stream chunk", response_chunk, request_id, log_dir, verbose)

            yield f"data: {json.dumps(response_chunk)}\n\n"

            await asyncio.sleep(0)

        yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"Streaming error: request_id={request_id}, error={str(e)}")
        logger.debug(f"Traceback: request_id={request_id}, traceback={error_detail}")
        log_verbose_response("Streaming error", {"error": str(e), "traceback": error_detail}, request_id, log_dir, verbose)

        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "error",
            "choices": [{
                "index": 0,
                "delta": {"content": f"[ERROR] {str(e)}"},
                "finish_reason": "error",
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", dependencies=[Depends(_concurrency_guard)])
async def chat_completions(request: ChatCompletionRequest, raw_request: Request = None):
    """Chat completions (OpenAI-compatible)"""

    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    logger.info(f"Request received: request_id={request_id}, model={request.model}, stream={request.stream}, active_requests={_active_requests}")
    logger.debug(f"Request details: request_id={request_id}, messages={len(request.messages)}, tools={bool(request.tools)}")

    if request.stream:
        async def streaming_wrapper(req, req_id):
            async for chunk in stream_chat_completions(req, req_id):
                yield chunk
        return StreamingResponse(
            streaming_wrapper(request, request_id),
            media_type="text/event-stream"
        )

    try:
        if raw_request is not None and is_verbose_full_enabled(verbose_full):
            body_bytes = await raw_request.body()
            raw_body = body_bytes.decode("utf-8", errors="replace")
            log_verbose_response("Raw request", raw_body, request_id, log_dir, verbose)
    except Exception as e:
        msg = f"[log error] Failed to log raw request: {e}"
        print(msg)
        log_to_gateway_file(msg, log_dir)

    try:
        start_time = time.time()

        chat_response, route_info = await orchestrator.chat(request)
        response = chat_response.model_dump()
        is_direct = route_info.is_direct
        final_provider_name = route_info.provider_name
        routing_reason = route_info.reason

        if is_direct:
            log_verbose_response("Direct provider call", {"provider": request.model}, request_id, log_dir, verbose)

        total_latency_ms = (time.time() - start_time) * 1000
        log_verbose_response("Raw backend response", response, request_id, log_dir, verbose)
    except RoutingError as routing_err:
        logger.error(f"Routing failed: request_id={request_id}, error={routing_err}")
        raise HTTPException(status_code=400, detail=str(routing_err))
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"Inference error: request_id={request_id}, error={str(e)}")
        logger.debug(f"Traceback: request_id={request_id}, traceback={error_detail}")
        log_verbose_response("Non-streaming error", {"error": str(e), "traceback": error_detail}, request_id, log_dir, verbose)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Extract usage information from OpenAI format response
    usage_obj = None
    usage_data = response.get("usage", {})
    if usage_data:
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        usage_obj = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    # Process all choices from the backend response (supports n>1)
    tool_calling_disabled = request.tool_choice == "none"
    response_choices = []

    for raw_choice in response.get("choices", [{}]):
        message = raw_choice.get("message", {})
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        finish_reason = raw_choice.get("finish_reason") or "stop"
        choice_index = raw_choice.get("index", len(response_choices))
        tool_calls = None
        native_tool_calls = message.get("tool_calls")

        # --- Tool calls: prefer native; fall back to regex extraction from content ---
        if native_tool_calls and not tool_calling_disabled:
            tool_calls = native_tool_calls
            finish_reason = "tool_calls"
            logger.debug(f"Native tool calls detected: request_id={request_id}, choice={choice_index}, count={len(tool_calls)}")
        elif content and not tool_calling_disabled:
            try:
                cleaned_content, parsed_tools = parse_tool_calls_from_content(content)
                if parsed_tools:
                    tool_calls = parsed_tools
                    content = cleaned_content if cleaned_content else None
                    finish_reason = "tool_calls"
                    logger.debug(f"Parsed tool calls from content: request_id={request_id}, choice={choice_index}, count={len(tool_calls)}")
            except Exception as parse_error:
                logger.warning(f"Tool call parsing failed: request_id={request_id}, choice={choice_index}, error={str(parse_error)}")

        response_choices.append(
            ChatCompletionChoice(
                index=choice_index,
                message=ChatMessage(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        )

    model_name = response.get("model", "unknown")

    if usage_obj:
        try:
            _record_request_telemetry(
                request_id=request_id,
                model_name=model_name,
                prompt_tokens=usage_obj.prompt_tokens,
                completion_tokens=usage_obj.completion_tokens,
                total_latency_ms=total_latency_ms,
                start_time=start_time,
                first_token_time=None,
                provider_name=final_provider_name,
                is_direct=is_direct,
                is_streaming=False,
            )
        except Exception as telemetry_error:
            logger.warning(f"Telemetry recording failed: request_id={request_id}, error={str(telemetry_error)}")

    logger.info(
        f"Request completed: request_id={request_id}, "
        f"model={model_name}, provider={final_provider_name}, "
        f"reason={routing_reason}, latency={total_latency_ms:.2f}ms"
    )

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=response_choices,
        usage=usage_obj,
    )


@app.get("/v1/stats")
async def get_stats():
    """Get statistics including token metrics, bucketed by provider name."""
    if telemetry is None:
        raise HTTPException(status_code=503, detail="Telemetry not initialized")

    metrics = telemetry.get_metrics()
    total_requests = metrics.total_requests
    total_tokens = metrics.total_tokens

    by_provider_tokens = {}
    by_provider_latency = {}
    distribution = {}
    for name, p in metrics.by_provider.items():
        distribution[name] = p.requests
        by_provider_tokens[name] = {
            "input_tokens": p.input_tokens,
            "output_tokens": p.output_tokens,
            "total_tokens": p.total_tokens,
            "request_count": p.requests,
            "avg_tokens_per_request": round(p.avg_tokens_per_request, 1),
            "request_share": round(p.requests / total_requests, 3) if total_requests else 0.0,
            "token_share": round(p.total_tokens / total_tokens, 3) if total_tokens else 0.0,
        }
        by_provider_latency[name] = {
            "avg_latency_ms": round(p.avg_latency_ms, 2),
            "avg_ttft_ms": round(p.avg_ttft_ms, 2),
            "avg_tpot_ms": round(p.avg_tpot_ms, 4),
            "ttft_count": p.ttft_count,
            "tpot_count": p.tpot_count,
        }

    return {
        "routing_stats": {
            "total_requests": total_requests,
            "by_provider": distribution,
        },
        "token_metrics": {
            "by_provider": by_provider_tokens,
            "overall": {
                "total_tokens": total_tokens,
                "total_input_tokens": metrics.total_input_tokens,
                "total_output_tokens": metrics.total_output_tokens,
                "total_requests": total_requests,
                "avg_tokens_per_request": round(metrics.avg_tokens_per_request, 1),
            },
        },
        "latency_metrics": {
            "by_provider": by_provider_latency,
            "overall": {
                "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                "avg_ttft_ms": round(metrics.avg_ttft_ms, 2),
                "avg_tpot_ms": round(metrics.avg_tpot_ms, 4),
                "ttft_count": metrics.ttft_count,
                "tpot_count": metrics.tpot_count,
            },
        },
    }

@app.post("/v1/stats/reset")
async def reset_stats():
    """Reset all statistics metrics"""
    if telemetry is None:
        raise HTTPException(status_code=503, detail="Telemetry not initialized")

    telemetry.reset()

    logger.info("Statistics metrics reset via API")

    return {
        "status": "success",
        "message": "All statistics metrics have been reset",
        "timestamp": int(time.time()),
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Validation error handler with request body logging"""
    raw_body = (await request.body()).decode("utf-8", errors="replace")

    msg = "❌ Request validation failed"
    print(msg)
    log_to_gateway_file(msg, log_dir)
    logger.error(f"Request validation failed: {request.method} {request.url.path}")

    msg = f"   Path: {request.method} {request.url.path}"
    print(msg)
    log_to_gateway_file(msg, log_dir)

    msg = f"   Body: {raw_body}"
    print(msg)
    log_to_gateway_file(msg, log_dir)
    logger.debug(f"Request body: {raw_body}")

    msg = f"   Errors: {exc.errors()}"
    print(msg)
    log_to_gateway_file(msg, log_dir)
    logger.error(f"Validation errors: {exc.errors()}")

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Request validation failed",
                "type": "RequestValidationError",
                "detail": exc.errors(),
                "body": raw_body,
            }
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    """Global exception handler"""
    import traceback
    msg = f"❌ Unhandled exception: {type(exc).__name__}: {str(exc)}"
    print(msg)
    log_to_gateway_file(msg, log_dir)
    log_to_gateway_file(traceback.format_exc(), log_dir)
    logger.error(f"Unhandled exception: {type(exc).__name__}: {str(exc)}")
    logger.debug(f"Traceback: {traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": type(exc).__name__}}
    )


# ==================== Main ====================

def main():
    """Start server"""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Inference Router API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--verbose", action="store_true", help="Print raw backend responses to the terminal")
    parser.add_argument("--verbose_full", action="store_true", help="Print raw requests and raw backend responses to the terminal")
    parser.add_argument("--save_logs_to", default=None, help="Directory to save gateway logs (requests and responses)")
    parser.add_argument("--max-concurrency", type=int, default=3, help="Max concurrent requests (0 = unlimited)")

    args = parser.parse_args()

    if args.verbose_full:
        args.verbose = True

    if args.verbose:
        os.environ["GATEWAY_VERBOSE"] = "1"
    if args.verbose_full:
        os.environ["GATEWAY_VERBOSE_FULL"] = "1"
    if args.save_logs_to:
        os.environ["GATEWAY_LOG_DIR"] = args.save_logs_to
    os.environ["GATEWAY_MAX_CONCURRENCY"] = str(args.max_concurrency)

    banner = [
        "=" * 80,
        "🚀 Starting Inference Router API Server",
        "=" * 80,
        f"   Host: {args.host}",
        f"   Port: {args.port}",
        f"   Config: {args.config}",
        f"   Verbose: {args.verbose}",
        f"   Verbose full: {args.verbose_full}",
        f"   Max concurrency: {'unlimited' if args.max_concurrency <= 0 else args.max_concurrency}",
        "=" * 80,
    ]
    for line in banner:
        print(line)

    uvicorn.run(
        "src.router.gateway:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
