# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""v1 API endpoints."""

import logging
from typing import Optional
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    PluginResponse,
    PluginListResponse,
    PluginSettingsResponse,
    PluginConfigUpdateRequest,
)
from src.exceptions import RoutingError, ProviderError


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat/completions", response_model=Optional[ChatCompletionResponse])
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    Accepts a chat completion request and routes it to the configured provider(s).
    """
    router_instance = http_request.app.state.router
    request_id = str(uuid.uuid4())

    logger.info(
        f"[{request_id}] Chat request: model={request.model}, messages={len(request.messages)}"
    )

    try:
        if request.stream:
            # Streaming response
            return StreamingResponse(
                _stream_response(router_instance, request, request_id),
                media_type="text/event-stream",
            )
        else:
            # Single response
            response = await router_instance.chat(request)
            logger.info(f"[{request_id}] Chat response successful")
            return response

    except RoutingError as e:
        logger.error(f"[{request_id}] Routing error: {e}")
        raise HTTPException(status_code=500, detail=f"Routing error: {e}")

    except ProviderError as e:
        logger.error(f"[{request_id}] Provider error: {e}")
        raise HTTPException(status_code=502, detail=f"Provider error: {e}")

    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/models")
async def list_models(http_request: Request):
    """List available models from all providers."""
    router_instance = http_request.app.state.router

    try:
        models = await router_instance.list_models()
        return {
            "object": "list",
            "data": models,
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


async def _stream_response(router_instance, request: ChatCompletionRequest, request_id: str):
    """
    Helper to stream chat completion responses.

    Yields SSE-formatted chunks.
    """
    try:
        async for chunk in router_instance.chat_stream(request):
            # Format as Server-Sent Events
            chunk_dict = chunk.to_dict() if hasattr(chunk, "to_dict") else chunk.dict()
            yield f"data: {chunk_dict}\n\n"

        # Send completion marker
        yield "data: [DONE]\n\n"
        logger.info(f"[{request_id}] Streaming completed")

    except Exception as e:
        logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)
        yield f"data: {{'error': 'Stream error: {e}'}}\n\n"


# Plugin Configuration API Endpoints


@router.get("/plugins")
async def list_plugins(http_request: Request) -> PluginListResponse:
    """List all configured plugins."""
    plugin_manager = http_request.app.state.plugin_manager

    try:
        plugins_config = plugin_manager.get_all_plugins_config()
        plugins_response = []
        for config in plugins_config:
            plugin_resp = PluginResponse(
                name=config["name"],
                node=config["node"],
                enabled=config["enabled"],
                trigger=config["trigger"],
                settings=PluginSettingsResponse(**config["settings"]),
            )
            plugins_response.append(plugin_resp)

        return PluginListResponse(data=plugins_response)
    except Exception as e:
        logger.error(f"Failed to list plugins: {e}")
        raise HTTPException(status_code=500, detail="Failed to list plugins")


@router.get("/plugins/{name}/{node}")
async def get_plugin(name: str, node: str, http_request: Request) -> PluginResponse:
    """Get plugin configuration by name and node type."""
    plugin_manager = http_request.app.state.plugin_manager

    try:
        plugin = plugin_manager.get_plugin_by_name_and_node(name, node)
        if not plugin:
            raise HTTPException(
                status_code=404,
                detail=f"Plugin '{name}' with node '{node}' not found",
            )

        return PluginResponse(
            name=plugin.name,
            node=plugin.plugin_type(),
            enabled=True,  # Plugins in manager are always enabled
            trigger=plugin.trigger,
            settings=PluginSettingsResponse(
                extra_config=getattr(plugin.parsed_settings, "extra_config", {})
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get plugin {name}/{node}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get plugin configuration")


@router.put("/plugins/{name}/{node}")
async def update_plugin(
    name: str, node: str, update_req: PluginConfigUpdateRequest, http_request: Request
) -> PluginResponse:
    """Update plugin configuration by name and node type."""
    plugin_manager = http_request.app.state.plugin_manager

    try:
        plugin = plugin_manager.get_plugin_by_name_and_node(name, node)
        if not plugin:
            raise HTTPException(
                status_code=404,
                detail=f"Plugin '{name}' with node '{node}' not found",
            )

        # Update settings if provided
        if update_req.settings:
            new_settings = update_req.settings.dict(exclude_unset=True)
            if not plugin_manager.update_plugin_settings(name, node, new_settings):
                raise HTTPException(status_code=500, detail="Failed to update plugin settings")

        # Return updated configuration
        return PluginResponse(
            name=plugin.name,
            node=plugin.plugin_type(),
            enabled=True,  # Plugins in manager are always enabled
            trigger=plugin.trigger,
            settings=PluginSettingsResponse(
                extra_config=getattr(plugin.parsed_settings, "extra_config", {})
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update plugin {name}/{node}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update plugin configuration")


@router.post("/plugins/{name}/{node}")
async def create_or_update_plugin(
    name: str, node: str, update_req: PluginConfigUpdateRequest, http_request: Request
) -> PluginResponse:
    """Create or update plugin configuration by name and node type."""
    plugin_manager = http_request.app.state.plugin_manager

    try:
        # Check if plugin exists
        plugin = plugin_manager.get_plugin_by_name_and_node(name, node)
        if plugin:
            # Plugin exists, update it
            if update_req.settings:
                new_settings = update_req.settings.dict(exclude_unset=True)
                if not plugin_manager.update_plugin_settings(name, node, new_settings):
                    raise HTTPException(status_code=500, detail="Failed to update plugin settings")

            return PluginResponse(
                name=plugin.name,
                node=plugin.plugin_type(),
                enabled=True,
                trigger=plugin.trigger,
                settings=PluginSettingsResponse(
                    extra_config=getattr(plugin.parsed_settings, "extra_config", {})
                ),
            )
        else:
            # Plugin doesn't exist - for now, just return error
            # In future, could support dynamic plugin registration
            raise HTTPException(
                status_code=404,
                detail=f"Plugin '{name}' with node '{node}' not found",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create/update plugin {name}/{node}: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure plugin")
