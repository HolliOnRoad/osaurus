# Osaurus — vmlx-engine Integration

## Architecture

Osaurus is a native macOS SwiftUI app. Inference is handled by **vmlx-engine**, a Python MLX backend running as a subprocess per model.

```
SwiftNIO gateway (port 1337) → ChatEngine → VMLXService → HTTP POST → Python engine (random port)
                                                          ← SSE stream ←
```

### Key Components

| Component | File | Role |
|-----------|------|------|
| VMLXService | Services/Inference/VMLXService.swift | HTTP client to Python engine, SSE parser, tool call accumulation |
| VMLXProcessManager | Services/Inference/VMLXProcessManager.swift | Process spawning, health polling, idle sleep, crash restart |
| VMLXGateway | Services/Inference/VMLXGateway.swift | Model-to-port registry (actor) |
| VMLXEngineConfig | Services/Inference/VMLXEngineConfig.swift | Maps ServerConfiguration → CLI args |
| VMLXSSEParser | Services/Inference/VMLXSSEParser.swift | Parses SSE lines, extracts content/reasoning/tool_calls/usage |
| ChatEngine | Services/Chat/ChatEngine.swift | Routes requests to services, stream wrapping |
| ServerConfiguration | Models/Configuration/ServerConfiguration.swift | All settings with defaults |
| ConfigurationView | Views/Settings/ConfigurationView.swift | Settings UI |

### Python Engine Source

- `Resources/vmlx_engine/` — stripped engine source (no audio/mcp/image_gen/gradio/commands)
- `scripts/bundle-python.sh` — builds relocatable Python 3.12 + all deps
- `Resources/bundled-python/` — output of bundle script (gitignored, ~400-640MB)

### Process Lifecycle

1. User sends message → `VMLXService.ensureEngineRunning()`
2. If not running: `VMLXProcessManager.launchEngine()` spawns Python on random port
3. Health check polls `/health` every 2s for up to 120s
4. Registers with `VMLXGateway`
5. HTTP POST to `http://127.0.0.1:<port>/v1/chat/completions`
6. SSE stream parsed and yielded as `AsyncThrowingStream<String>`

### Parser Auto-Detection

Both `--tool-call-parser auto` and `--reasoning-parser auto` (defaults) trigger Python's `model_config_registry` to detect the right parser from the model's `config.json` `model_type` field.

### Settings → CLI Flag Mapping

All 28+ settings in ServerConfiguration map to Python CLI flags via `VMLXEngineConfig.buildArgs()`. See the file for the complete mapping.

### Idle Sleep

- Soft sleep: `POST /admin/soft-sleep` — clears GPU caches, model stays loaded
- Deep sleep: `POST /admin/deep-sleep` — unloads model from VRAM
- Auto-wake: next request triggers automatic reload

### Build

```bash
make bundle-python   # Build bundled Python (once, ~10 min)
make app             # Build app + embed CLI + Python bundle
```

Dev mode: Falls back to `~/mlx/vllm-mlx/panel/bundled-python/python/bin/python3` if bundled Python not built.

## TODO

- [ ] Inference stats display (TTFT, TPS, cache hit, pp/s) — VMLXSSEParser already extracts `VMLXUsage` (prompt_tokens, completion_tokens, cached_tokens, cache_detail) but it's not surfaced in the chat UI. Needs a stats overlay component in ChatView reading from the stream's final usage chunk. Client-side timing (TTFT, TPS) can be computed from delta timestamps in the streaming loop.
- [ ] Periodic health monitor (currently only detects crashes on next request)
- [ ] Per-model engine configuration (all models share global config)
