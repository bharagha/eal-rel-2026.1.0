# Quick Start Guide

Get the Inference Router running with one configured backend and verify the
OpenAI-compatible API.

## Prerequisites

- Docker 25.0 or higher for Docker deployment.
- Python 3.10+ for local development.
- An OpenAI-compatible inference backend, such as vLLM, reachable from this
  host, or an API key for a cloud provider supported by LiteLLM.

The router itself is lightweight. Local model serving requirements depend on
the backend you connect to.

## 1. Configure

Copy the example config and edit it to point at your backend:

```bash
cd inference-router
mkdir -p workspace
cp config.example.yaml workspace/config.yaml
```

A minimal `workspace/config.yaml` with one local vLLM model:

```yaml
providers:
  - name: "local"
    type: "hosted_vllm"
    model: "Qwen/Qwen3.5-9B"
    enabled: true
    metadata:
      labels:
        - "local"
      cost: 0
      performance: 0.85
      capability:
        complexity: 0.75
    settings:
      endpoint: "http://localhost:8088/v1"
      timeout: 300.0
      auth:
        scheme: "none"
        api_key: null
        custom_headers: {}
```

The router uses [LiteLLM](https://docs.litellm.ai/docs/#litellm-python-sdk) to
support different provider backends. `type` is passed to LiteLLM as the prefix
in `type/model`. Use `hosted_vllm` for a self-hosted vLLM server, or any other
[LiteLLM-supported provider](https://docs.litellm.ai/docs/providers).

## 2. Build Image

Build the Docker image:

```bash
bash scripts/deploy_docker.sh --build
```

## 3. Deploy

Start the router on port `8000` by default:

```bash
bash scripts/deploy_docker.sh
```

Check that the container is running:

```bash
docker ps --filter name=inference-router
```

To stop the router:

```bash
bash scripts/deploy_docker.sh --down
```

To use a different host port:

```bash
ROUTER_PORT=9000 bash scripts/deploy_docker.sh
```

## 4. Verify

List available models. The response includes `router` plus your configured
providers:

```bash
curl http://localhost:8000/v1/models
```

Let the router pick the provider based on the configured policy:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "router",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

Force a specific provider by using any provider name returned by `/v1/models`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

Stream a chat completion:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "router",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }'
```

View router stats:

```bash
curl http://localhost:8000/v1/stats
```

For endpoint details, see [api-reference.md](api-reference.md).
