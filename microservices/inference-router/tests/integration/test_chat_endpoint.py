# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for chat endpoint."""

import pytest
from fastapi.testclient import TestClient

from src.config import RouterConfig, ProviderConfig, TelemetryConfig
from src.router import RouterOrchestrator
from src.api.app import create_app
from src.observability import InMemoryTelemetry, TelemetryRecorder


@pytest.fixture
def test_router_config():
    """Create test router config."""
    return RouterConfig(
        providers=[
            ProviderConfig(
                name="test-vllm",
                type="vllm",
                enabled=True,
                settings={
                    "endpoint": "http://localhost:9999",  # Fake endpoint
                    "timeout": 5.0,
                },
            )
        ],
        telemetry=TelemetryConfig(backend="memory", enabled=True),
    )


@pytest.fixture
async def test_router(test_router_config):
    """Create test router."""
    router = RouterOrchestrator(test_router_config)
    await router.initialize()
    yield router
    router.shutdown()


@pytest.fixture
def test_app(test_router, test_router_config):
    """Create test FastAPI app."""
    telemetry = InMemoryTelemetry()
    telemetry_recorder = TelemetryRecorder(telemetry)
    return create_app(test_router, test_router_config, telemetry_recorder)


@pytest.mark.integration
def test_health_endpoint(test_app):
    """Test health check endpoint."""
    client = TestClient(test_app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.integration
def test_health_detailed_endpoint(test_app):
    """Test detailed health check endpoint."""
    client = TestClient(test_app)
    response = client.get("/health/detailed")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "providers" in data


@pytest.mark.integration
def test_list_models_endpoint(test_app):
    """Test list models endpoint."""
    client = TestClient(test_app)
    response = client.get("/v1/models")

    assert response.status_code in [200, 502]  # Either success or provider unavailable
    assert "data" in response.json() or "error" in response.json()


@pytest.mark.integration
def test_chat_completions_invalid_request(test_app):
    """Test chat completions with invalid request."""
    client = TestClient(test_app)

    # Missing required fields
    response = client.post("/v1/chat/completions", json={})

    assert response.status_code == 422  # Validation error


@pytest.mark.integration
def test_metrics_endpoint(test_app):
    """Test metrics endpoint."""
    client = TestClient(test_app)
    response = client.get("/metrics")

    assert response.status_code == 200


@pytest.mark.integration
def test_list_plugins_endpoint(test_app):
    """Test list plugins endpoint."""
    client = TestClient(test_app)
    response = client.get("/v1/plugins")

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


@pytest.mark.integration
def test_get_plugin_by_name_and_node(test_app):
    """Test get plugin endpoint."""
    client = TestClient(test_app)

    # First list all plugins to find one
    list_response = client.get("/v1/plugins")
    assert list_response.status_code == 200

    plugins = list_response.json()["data"]
    if plugins:
        # Get first plugin
        plugin = plugins[0]
        response = client.get(f"/v1/plugins/{plugin['name']}/{plugin['node']}")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == plugin["name"]
        assert data["node"] == plugin["node"]
        assert "settings" in data
        assert "extra_config" in data["settings"]


@pytest.mark.integration
def test_get_nonexistent_plugin(test_app):
    """Test get nonexistent plugin returns 404."""
    client = TestClient(test_app)
    response = client.get("/v1/plugins/nonexistent/unknown")

    assert response.status_code == 404


@pytest.mark.integration
def test_update_plugin_settings(test_app):
    """Test update plugin settings endpoint."""
    client = TestClient(test_app)

    # First list plugins
    list_response = client.get("/v1/plugins")
    plugins = list_response.json()["data"]

    if plugins:
        plugin = plugins[0]
        # Update plugin settings
        update_data = {
            "settings": {
                "extra_config": {"key": "value"}
            }
        }
        response = client.put(
            f"/v1/plugins/{plugin['name']}/{plugin['node']}",
            json=update_data,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["settings"]["extra_config"] == {"key": "value"}


@pytest.mark.integration
def test_create_or_update_plugin(test_app):
    """Test create/update plugin endpoint."""
    client = TestClient(test_app)

    # First list plugins
    list_response = client.get("/v1/plugins")
    plugins = list_response.json()["data"]

    if plugins:
        plugin = plugins[0]
        # Create or update plugin
        update_data = {
            "settings": {
                "extra_config": {"test": "data"}
            }
        }
        response = client.post(
            f"/v1/plugins/{plugin['name']}/{plugin['node']}",
            json=update_data,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == plugin["name"]
        assert data["node"] == plugin["node"]
