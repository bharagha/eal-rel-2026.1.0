# API Reference

The router exposes an OpenAI-compatible API. All examples assume the router is
running on `localhost:8000`.

## Service Info

Endpoint:

```bash
GET /
```

**Description:**

Returns service name, version, status, and a map of available endpoints.

**Response:**

- 200 OK:

  ```json
  {
      "name": "Inference Router API",
      "version": "1.0.0",
      "status": "running",
      "endpoints": {
          "health": "/health",
          "chat": "/v1/chat/completions",
          "models": "/v1/models",
          "audio_transcriptions": "/v1/audio/transcriptions",
          "audio_speech": "/v1/audio/speech",
          "embeddings": "/v1/embeddings",
          "rerank": "/v1/rerank",
          "stats": "/stats"
      }
  }
  ```

## Health Check

Endpoint:

```bash
GET /health
```

**Description:**

Liveness check. Includes router initialization status and current concurrency
counters.

**Response:**

- 200 OK:

  ```json
  {
      "status": "healthy",
      "router": "initialized",
      "timestamp": 1733040000,
      "concurrency": {
          "active_requests": 0,
          "max_concurrency": 3
      }
  }
  ```

  `max_concurrency` is the integer limit, or the string `"unlimited"` when no
  limit is set.

- 503 Service Unavailable:

  ```json
  {"detail": "Router not initialized"}
  ```

## List Models

Endpoint:

```bash
GET /v1/models
```

**Description:**

Lists every available model. The response always includes the virtual model
`"router"`, all configured local models, and the cloud model if enabled.

**Response:**

- 200 OK:

  ```json
  {
      "object": "list",
      "data": [
          {
              "id": "Qwen/Qwen3-8B",
              "object": "model",
              "created": 1733040000,
              "owned_by": "local"
          },
          {
              "id": "MiniMax-M2.7",
              "object": "model",
              "created": 1733040000,
              "owned_by": "minimax"
          },
          {
              "id": "router",
              "object": "model",
              "created": 1733040000,
              "owned_by": "inference-router"
          }
      ]
  }
  ```

## Chat Completions

Endpoint:

```bash
POST /v1/chat/completions
```

**Description:**

OpenAI-compatible chat completion. Set `model` to a concrete ID to pin the
backend, or to `"router"` for smart routing. Set `stream: true` for SSE
streaming.

**Request Body:**

```json
{
    "model": "router",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"}
    ],
    "stream": false,
    "temperature": 0.7,
    "max_tokens": 200
}
```

- `model`: Either `"router"` for smart routing or any model ID from
  `/v1/models`.
- `messages`: List of OpenAI-format messages.
- `stream`: When `true`, response is streamed as SSE.
- Other OpenAI parameters, such as `temperature`, `max_tokens`, `top_p`,
  `tools`, `tool_choice`, and `response_format`, pass through to the backend.

**Response (non-streaming):**

- 200 OK:

  ```json
  {
      "id": "chatcmpl-...",
      "object": "chat.completion",
      "created": 1733040000,
      "model": "Qwen/Qwen3-8B",
      "choices": [
          {
              "index": 0,
              "message": {"role": "assistant", "content": "..."},
              "finish_reason": "stop"
          }
      ],
      "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
  }
  ```

**Response (streaming):**

- 200 OK with `Content-Type: text/event-stream`. Each chunk is an SSE
  `data: {...}` line. The stream ends with `data: [DONE]`.

**Errors:**

- 400 Bad Request: unknown model name.
- 422 Unprocessable Entity: request validation failed.
- 429 Too Many Requests: concurrency limit reached.
- 500 Internal Server Error: inference or unexpected failure.
- 503 Service Unavailable: router not initialized.

## Stats

Endpoint:

```bash
GET /v1/stats
```

**Query Parameters:**

- `model` (optional): filter compression stats to a specific model name.

**Description:**

Aggregated routing, token, latency, and compression metrics.

**Response:**

- 200 OK: object containing `routing_stats`, `token_metrics`,
  `latency_metrics`, and `compression`. Each section reports per-target
  breakdowns where applicable.

- 503 Service Unavailable:

  ```json
  {"detail": "Telemetry not initialized"}
  ```

## Reset Stats

Endpoint:

```bash
POST /v1/stats/reset
```

**Description:**

Clears all telemetry metrics.

**Response:**

- 200 OK:

  ```json
  {
      "status": "success",
      "message": "All statistics metrics have been reset",
      "timestamp": 1733040000
  }
  ```

## Embeddings

Endpoint:

```bash
POST /v1/embeddings
```

**Description:**

Forwards to a configured embeddings service. Requires `embeddings_service` in
`config.yaml` with `enabled: true`.

**Request Body:**

```json
{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "The quick brown fox jumps over the lazy dog"
}
```

**Errors:**

- 502 Bad Gateway: upstream embeddings service unreachable.
- 503 Service Unavailable: embeddings service not configured.
- 504 Gateway Timeout: upstream timed out.

## Audio Transcriptions

Endpoint:

```bash
POST /v1/audio/transcriptions
```

**Description:**

Forwards multipart audio uploads to a configured transcription service.
Requires `transcription_service` in `config.yaml` with `enabled: true`.

**Request:**

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"
```

**Errors:** same shape as Embeddings: 502, 503, or 504.


## Audio Speech

Endpoint:

```bash
POST /v1/audio/speech
```

**Description:**

Forwards to a configured TTS service. Requires `tts_service` in `config.yaml`
with `enabled: true`. Returns audio bytes with the upstream `Content-Type`
preserved. The default content type is `audio/mpeg`.

**Request Body:**

```json
{
    "model": "tts-1",
    "input": "Hello world"
}
```

**Errors:** same shape as Embeddings: 502, 503, or 504.

## Rerank

Endpoint:

```bash
POST /v1/rerank
```

**Description:**

Forwards to a Cohere-compatible rerank service. Requires `rerank_service` in
`config.yaml` with `enabled: true`.

**Request Body:**

```json
{
    "model": "rerank-english-v3.0",
    "query": "What is the capital of France?",
    "documents": ["Paris is the capital.", "London is nice.", "Berlin is cold."]
}
```

**Errors:** same shape as Embeddings: 502, 503, or 504.