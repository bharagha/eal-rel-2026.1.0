# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration data classes for the inference router."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum


class TelemetryBackendType(str, Enum):
    """Types of telemetry backends."""

    MEMORY = "memory"
    FILE = "file"


@dataclass
class ProviderAuthConfig:
    """Authentication configuration for a provider."""

    scheme: str = "bearer"  # 'bearer', 'api_key', 'none'
    api_key: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """Generic provider configuration."""

    name: str
    type: str  # 'vllm', 'ollama', 'openai', etc.
    model: Optional[str] = None  # Backend model identifier, e.g. "Qwen/Qwen3.5-9B"
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelemetryConfig:
    """Configuration for telemetry/observability."""

    backend: TelemetryBackendType = TelemetryBackendType.MEMORY
    file_path: Optional[str] = None  # For file backend
    enabled: bool = True


@dataclass
class RoutingConfig:
    """Configuration for strategy routing."""

    policy: Optional[str] = None
    strategy: Optional[str] = None


@dataclass
class PluginConfig:
    """Generic plugin configuration."""

    name: str
    node: str  # plugin selector key
    enabled: bool = True
    trigger: str = "prerouting"  # 'prerouting', 'postrouting', or 'postresponse'
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterConfig:
    """Top-level router configuration."""

    providers: List[ProviderConfig] = field(default_factory=list)
    plugins: List[PluginConfig] = field(default_factory=list)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    log_level: str = "INFO"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    class Config:
        """Dataclass config."""

        pass
