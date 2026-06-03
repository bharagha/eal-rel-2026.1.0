# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference Router main entry point."""

import logging
import asyncio
import os
import signal
from pathlib import Path

import uvicorn

from src.config import load_config
from src.router import RouterOrchestrator
from src.observability import create_telemetry, TelemetryRecorder
from src.api.app import create_app


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the inference router."""
    logger.info("Starting Inference Router")

    # Load configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, trying config.example.yaml")
        config_path = Path("config.example.yaml")

    logger.info(f"Loading config from {config_path}")
    config = load_config(str(config_path))

    # Initialize telemetry
    logger.info(f"Initializing telemetry backend: {config.telemetry.backend}")
    telemetry = create_telemetry(config.telemetry)
    telemetry_recorder = TelemetryRecorder(telemetry)

    # Initialize router
    logger.info("Initializing router with configured providers")
    router = RouterOrchestrator(config, telemetry=telemetry_recorder)
    await router.initialize()

    # Create FastAPI app
    logger.info("Creating FastAPI application")
    app = create_app(router, config, telemetry_recorder)

    # Configure Uvicorn server.
    # Bind host/port are supplied at deployment time (CLI flags / docker-compose),
    # not via config.yaml.
    host = os.environ.get("ROUTER_HOST", "0.0.0.0")
    port = int(os.environ.get("ROUTER_PORT", "8000"))
    config_uvicorn = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=config.log_level.lower(),
        access_log=True,
    )

    server = uvicorn.Server(config_uvicorn)

    # Handle shutdown gracefully
    async def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received, cleaning up...")
        server.should_exit = True
        await telemetry_recorder.shutdown()
        router.shutdown()

    # Start server
    logger.info(f"Starting API server on {host}:{port}")
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, shutting down...")
        await telemetry_recorder.shutdown()
        router.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
