# vmlx-engine Integration

Osaurus uses **vmlx-engine** as its local inference backend. This replaces the previous mlx-swift-lm in-process approach with a Python subprocess architecture.

## Architecture

```
Client (SwiftUI / API)
       |
SwiftNIO Gateway (port 1337)
       |
  ChatEngine (actor)
       |
  VMLXService (actor) ──HTTP POST──> Python engine (127.0.0.1:<random-port>)
       |                              vmlx_engine.cli serve <model>
  VMLXSSEParser     <────SSE stream──
       |
  AsyncThrowingStream<String>
```

Each model runs as its own Python subprocess on a dynamically allocated localhost port. The Swift gateway handles authentication, CORS, model routing, and re-emits SSE to external clients.

## Model Types

### Standard MLX Models
HuggingFace repos with `.safetensors` weights + `config.json` + tokenizer files.
- Text LLMs: Qwen, Llama, Mistral, DeepSeek, GPT-OSS, Gemma, Phi, etc.
- Vision LLMs (VLM): Qwen-VL, Pixtral, LLaVA — pass images via `content` array with `image_url` type
- Detected automatically from `config.json` `model_type` field

### JANG Quantized Models
Custom 2-3 bit quantization format from JANGQ-AI. These use `jang_loader.py` for weight loading and automatically enable **TurboQuant KV cache** (3-bit KV compression with 5x memory savings). No user configuration needed — detected from weight file format.

### Model Sizes & Memory
- Models are downloaded to `~/Library/Application Support/osaurus/models/`
- The engine requires the full model to fit in unified memory (RAM + VRAM shared on Apple Silicon)
- Rule of thumb: model file size on disk = approximate memory needed
- JANG models use ~40-60% less memory than equivalent MLX quantized models

## Settings Reference

### Engine (always-on)
| Setting | CLI Flag | Default | Notes |
|---------|----------|---------|-------|
| Continuous Batching | `--continuous-batching` | ON | Required for all cache types and multi-user |
| Max Concurrent Sequences | `--max-num-seqs` | 256 | Max parallel requests in batched mode |
| Stream Interval | `--stream-interval` | 1 | Tokens between SSE updates (1 = every token) |

### Cache (all on by default)
| Setting | CLI Flag | Default | Notes |
|---------|----------|---------|-------|
| Prefix Cache | `--enable-prefix-cache` | ON | Reuses KV states for shared system prompts |
| Prefix Cache Size | `--prefix-cache-size` | 100 | Max cached prefixes |
| Cache Memory % | `--cache-memory-percent` | 0.30 | Fraction of RAM for cache (0.1-0.8) |
| Cache Memory MB | `--cache-memory-mb` | auto | Fixed MB budget (overrides %) |
| Cache TTL | `--cache-ttl-minutes` | 0 | Entry expiry (0 = never) |
| Paged Cache | `--use-paged-cache` | ON | Block-based KV cache for memory efficiency |
| Block Size | `--paged-cache-block-size` | 64 | Tokens per cache block |
| Max Blocks | `--max-cache-blocks` | 1000 | Max blocks in memory |
| Disk Cache (L2) | `--enable-disk-cache` | ON | Persist prompt KV states to SSD |
| Disk Cache Max | `--disk-cache-max-gb` | 10.0 | Max disk usage |
| Block Disk Cache | `--enable-block-disk-cache` | ON | Block-level L2 for paged cache |
| Block Disk Max | `--block-disk-cache-max-gb` | 10.0 | Max block disk usage |

### KV Cache Quantization
| Setting | CLI Flag | Default | Notes |
|---------|----------|---------|-------|
| Quantization | `--kv-cache-quantization` | none | none/q4/q8. JANG models auto-use TurboQuant instead |
| Group Size | `--kv-cache-group-size` | 64 | Quantization group size |

### Parsers (auto-detected per model)
| Setting | CLI Flag | Default | Notes |
|---------|----------|---------|-------|
| Tool Call Parser | `--tool-call-parser` | auto | Auto-detects from model_config_registry |
| Reasoning Parser | `--reasoning-parser` | auto | Auto-detects from model_config_registry |

#### Tool Parser by Model Family
| Family | Parser |
|--------|--------|
| Qwen 2/3/3.5 | qwen |
| Llama 3/4 | llama |
| Mistral/Codestral/Pixtral | mistral |
| DeepSeek V2/V3/R1 | deepseek |
| Gemma 4 | gemma4 |
| Gemma 3, Phi 4 | hermes |
| GLM-4/Z1, GPT-OSS | glm47 |
| Nemotron | nemotron |
| MiniMax | minimax |
| Kimi K2 | kimi |
| Step 3.5 | step3p5 |
| Granite | granite |

#### Reasoning Parser by Model Family
| Family | Parser | Token Format |
|--------|--------|--------------|
| Qwen 3/3.5, MiniMax, Step | qwen3 | `<think>...</think>` |
| DeepSeek R1, Gemma 3, Phi4-R, Kimi | deepseek_r1 | `<think>...</think>` |
| Mistral 4 | mistral | `[THINK]...[/THINK]` |
| Gemma 4 | gemma4 | `<\|channel>thought...<channel\|>` |
| GLM-Z1, GPT-OSS | openai_gptoss | Harmony channel protocol |

### Performance
| Setting | CLI Flag | Default | Notes |
|---------|----------|---------|-------|
| JIT Compilation | `--enable-jit` | ON | mx.compile on forward pass |
| Speculative Model | `--speculative-model` | none | Draft model path for spec decoding |
| Draft Tokens | `--num-draft-tokens` | 3 | Tokens per speculative step |
| Prompt Lookup | `--enable-pld` | OFF | Prompt Lookup Decoding |

### Power Management
| Setting | Default | Notes |
|---------|---------|-------|
| Soft Sleep | ON, 10min | Clears GPU caches after idle timeout. Model stays loaded. |
| Deep Sleep | ON, 30min | Unloads model from VRAM after idle timeout. |
| Auto-Wake | Always | Next request automatically reloads the model. |

### Thinking / Reasoning
Controlled per-conversation via the thinking toggle button in the chat UI (not in global settings). When no preference is set, the engine auto-detects based on the model family.

## Streaming Protocol

The engine uses Server-Sent Events (SSE) following the OpenAI streaming format:

```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}\n\n
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n\n
data: {"id":"chatcmpl-xxx","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{...}}\n\n
data: [DONE]\n\n
```

### Reasoning Content
Models with reasoning emit `reasoning_content` in delta alongside `content`:
```json
{"delta": {"reasoning_content": "Let me think..."}}
{"delta": {"content": "The answer is 42."}}
```

### Tool Calls (two-chunk protocol)
1. **Chunk 1**: `delta.tool_calls` array with `finish_reason: null`
2. **Chunk 2**: empty delta with `finish_reason: "tool_calls"`

The Swift SSE parser accumulates tool call arguments across chunks and emits `ServiceToolInvocation` on the finish chunk.

### Usage Stats
When `stream_options.include_usage: true`, the final chunk includes:
```json
{
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 50,
    "total_tokens": 200,
    "prompt_tokens_details": {
      "cached_tokens": 100,
      "cache_detail": "paged+tq"
    }
  }
}
```

## Process Management

### Spawning
- Python binary: bundled (`Resources/bundled-python/python/bin/python3`) or dev fallback
- Environment: `PYTHONNOUSERSITE=1`, `PYTHONPATH=""`, `PYTHONHOME=<bundle>`, `-s` flag
- `HF_TOKEN` passed through for gated models (Llama 3, Gemma)

### Health Checks
- Polls `GET /health` every 2s for up to 120s during startup
- No periodic health monitor (crash detected on next request via auto-restart)

### Crash Recovery
- Monitor task watches process exit
- Non-zero exit → auto-restart with exponential backoff (2s, 4s, 6s)
- Max 3 restart attempts before giving up
- Last stderr line captured and logged for diagnostics

### Model Eviction
- **Strict Single Model** (default): Stops all other engines before loading a new model
- **Manual Multi Model**: Multiple engines coexist; user manually unloads via right-click context menu

### Shutdown
- App quit → `VMLXProcessManager.shared.stopAll()`
- Per-engine: SIGTERM → 1.5s grace → SIGKILL
- Orphaned process cleanup via `lsof -ti :<port>` before each launch

## Building the Python Bundle

```bash
./scripts/bundle-python.sh
```

Downloads Astral's python-build-standalone 3.12, installs all dependencies (mlx, mlx-lm, mlx-vlm, transformers, fastapi, uvicorn, etc.), applies patches for torch-free environment, cleans up to ~400-640MB.

Output: `Resources/bundled-python/python/`

### Patches Applied
1. `transformers/processing_utils.py` — Allow None sub-processors for VLM without torchvision
2. `transformers/processing_utils.py` — Skip ImportError for unavailable backends
3. `transformers/models/auto/video_processing_auto.py` — Null check for extractors
4. `mlx_vlm/utils.py` — Lazy-import soundfile
5. `mlx_vlm/models/qwen3_5/language.py` — mRoPE dimension fix for MoE
6. `mlx_lm/models/ssm.py` — Mamba state fixes (clip→maximum, float32 state dtype)

## Files Changed (from upstream osaurus)

### Deleted (MLX removal)
- `Services/Inference/MLXService.swift`
- `Services/ModelRuntime.swift`
- `Services/ModelRuntime/MLXGenerationEngine.swift`
- `Services/ModelRuntime/KVCacheStore.swift`
- `Services/ModelRuntime/StreamAccumulator.swift`
- `Services/ModelRuntime/RuntimeConfig.swift`
- `Services/ModelRuntime/Events.swift`
- 7 test files for deleted components

### Added
- `Services/Inference/VMLXService.swift`
- `Services/Inference/VMLXProcessManager.swift`
- `Services/Inference/VMLXGateway.swift`
- `Services/Inference/VMLXEngineConfig.swift`
- `Services/Inference/VMLXSSEParser.swift`
- `Services/Inference/PrefixHash.swift`
- `Resources/vmlx_engine/` (engine Python source)
- `scripts/bundle-python.sh`
- `docs/VMLX_ENGINE.md`

### Modified
- `Package.swift` — Removed MLX/MLXLLM/MLXVLM deps, kept Hub for downloads
- `ServerConfiguration.swift` — 28+ new vmlx engine fields, removed old KV fields
- `ConfigurationView.swift` — New engine/cache/parser/power settings UI
- `ChatEngine.swift` — Routes to VMLXService instead of MLXService
- `Router.swift` — Uses VMLXService.getAvailableModels()
- `HTTPHandler.swift` — Same
- `ModelManager.swift` — Removed MLX registry references
- `ModelCacheInspectorView.swift` — Shows running engine instances
- `FloatingInputCard.swift` — Engine status indicator, right-click unload
- `Makefile` — `bundle-python` target, app embeds Python bundle
- `.gitignore` — Excludes `Resources/bundled-python/`

## Status

### Done
- [x] MLX-swift removal — all imports, deps, files deleted
- [x] Python engine source copy (`Resources/vmlx_engine/`) with excluded modules (audio, mcp, image_gen, gradio, commands)
- [x] Bundle script (`scripts/bundle-python.sh`) — downloads Python 3.12, installs all deps, applies 6 patches
- [x] VMLXService — HTTP bridge to Python engine, SSE streaming, tool call accumulation
- [x] VMLXSSEParser — parses content, reasoning_content, tool_calls, usage stats
- [x] VMLXProcessManager — process spawning, health polling, crash restart (max 3 with backoff), SIGTERM→SIGKILL, orphan cleanup
- [x] VMLXGateway — model-to-port registry with fuzzy matching
- [x] VMLXEngineConfig — maps all 28+ ServerConfiguration fields to CLI flags
- [x] ServerConfiguration — all new vmlx fields with defaults (prefix/paged/disk cache ON, JIT ON, deep sleep 30min)
- [x] ConfigurationView — engine/cache/parser/power/performance settings UI
- [x] Reasoning content → thinking box (wraps `reasoning_content` in `<think>` tags for StreamingDeltaProcessor)
- [x] VLM multimodal content (image_url arrays passed to Python engine)
- [x] Tool calls in conversation history (multi-turn tool calling)
- [x] session_id / cache_hint for prefix cache reuse across turns
- [x] Per-model parser selection (chat UI picker → ModelOptionsStore → engine CLI args)
- [x] Concurrent launch guard (prevents duplicate Python processes for same model)
- [x] Idle timer reset after stream completes (not before)
- [x] Model unload via right-click context menu on model chip
- [x] Engine status indicator (green dot = running, gray = not loaded)
- [x] Idle sleep checkboxes (separate soft/deep with individual timeouts)
- [x] HF_TOKEN passthrough for gated models
- [x] Dev fallback to vmlx repo bundled Python

### Not Done
- [ ] **Inference stats display** (TTFT, TPS, cache hit, pp/s) — VMLXSSEParser already extracts `VMLXUsage` but it's not surfaced in the chat UI. Needs a stats overlay component.
- [ ] **Periodic health monitor** — currently crashes are only detected on next request via auto-restart. No 5-second polling loop like the Electron app.
- [ ] **Per-model engine configuration** — all models share global ServerConfiguration. Engine-level settings (cache, JIT) are baked at process launch.
- [ ] **Loading progress UI** — no progress bar during model loading. The chat just shows a spinner until health check passes (up to 120s).
- [ ] **Memory warnings** — no pre-load check for available RAM vs model size.
- [ ] **Process adoption** — if app crashes, orphaned Python processes aren't re-adopted on restart (they're killed and restarted instead).
- [ ] **Gateway health aggregation** — `GET /health` on port 1337 always returns healthy regardless of engine status.

### Known Limitations
- Parser changes require engine restart (unload + reload model) since parsers are CLI args
- TOCTOU port race on allocation (unlikely on loopback, standard mitigation would require fd passing)
- No `--chat-template`, `--api-key`, `--rate-limit` CLI flags exposed in UI
- No SSD offload (`--stream-from-disk`) for models larger than RAM
