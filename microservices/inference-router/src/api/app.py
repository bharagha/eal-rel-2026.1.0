# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application factory."""

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import RouterConfig
from src.router import RouterOrchestrator
from src.observability import TelemetryRecorder


logger = logging.getLogger(__name__)


def create_app(
    router: RouterOrchestrator,
    config: RouterConfig,
    telemetry_recorder: Optional[TelemetryRecorder] = None,
) -> FastAPI:
    """
    Create FastAPI application with configured middleware and routes.

    Args:
        router: RouterOrchestrator instance
        config: RouterConfig with settings
        telemetry_recorder: Optional telemetry recorder

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Inference Router",
        description="A pluggable inference router for chat completion requests",
        version="0.1.0",
    )

    # Store router in app state for use in endpoints
    app.state.router = router
    app.state.plugin_manager = router.plugin_manager
    app.state.telemetry_recorder = telemetry_recorder
    app.state.config = config

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include routers
    from src.api.v1 import router as v1_router

    app.include_router(v1_router.router, prefix="/v1")

    # Health check endpoint
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    # Detailed health check
    @app.get("/health/detailed")
    async def health_detailed():
        """Detailed health check with provider status."""
        provider_health = await router.health_check()
        return {
            "status": "healthy",
            "providers": provider_health,
        }

    # Metrics/telemetry endpoint
    @app.get("/metrics")
    async def metrics():
        """Get telemetry metrics."""
        if not telemetry_recorder:
            return {"error": "Telemetry not enabled"}

        events = await telemetry_recorder.get_events(limit=1000)
        return {
            "total_events": len(events),
            "recent_events": events[-10:] if events else [],
        }

    logger.info(f"Created FastAPI app with {len(app.routes)} routes")
    return app
