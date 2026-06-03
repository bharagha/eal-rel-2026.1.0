# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from src.config.loader import load_config


def test_load_config_accepts_blank_plugin_sections(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
providers:
  - name: local
    type: hosted_vllm
    model: Qwen/Qwen3.5-9B
    settings:
      endpoint: http://localhost:5000/v1
plugins:
  prerouting:
  postrouting:
  postresponse:
routing:
  policy: Balanced
telemetry:
  backend: memory
""".lstrip()
    )

    config = load_config(str(config_path))

    assert config.plugins == []


def test_load_config_accepts_empty_plugin_lists(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
providers:
  - name: local
    type: hosted_vllm
    model: Qwen/Qwen3.5-9B
    settings:
      endpoint: http://localhost:5000/v1
plugins:
  prerouting: []
  postrouting: []
  postresponse: []
routing:
  policy: Balanced
telemetry:
  backend: memory
""".lstrip()
    )

    config = load_config(str(config_path))

    assert config.plugins == []