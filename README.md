<h1 align="center">Jangosaurus</h1>

<p align="center">
  <strong>VMLXRuntime: Native Swift Inference Engine for Osaurus</strong><br>
  <em>Development branch for replacing <code>mlx-swift-lm</code> and external model-library dependencies inside Osaurus.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Dev-yellow" alt="Status">
  <img src="https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-black?logo=apple" alt="Platform">
  <img src="https://img.shields.io/badge/Swift-6.2-orange?logo=swift" alt="Swift">
  <img src="https://img.shields.io/badge/MLX-Metal%20GPU-blue" alt="MLX">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

> Status note, April 1 2026: the repo contains a real native VMLX runtime and real Osaurus integration, but some older docs below this repo previously mixed planned work, landed code, and aspirational status. This README is now aligned to the current branch state.

---

## Goal

Replace Osaurus's `mlx-swift-lm` backend with a native Swift inference engine that:
- uses `mlx-swift` for tensor ops only
- loads architectures natively in Swift
- handles hybrid SSM, MoE, MLA, and JANG mixed-precision weights
- provides a multi-layer cache stack with hybrid-safe behavior
- integrates directly into the Osaurus app, API, tools, and settings flow

---

## Current Branch Status

### Verified Working

| Area | Status |
|------|--------|
| Standard transformer path | Verified with real local models such as Llama 3.2 1B and Qwen 2.5 0.5B |
| Qwen 3.5 hybrid SSM path | Verified with real JANG models; hybrid cache split and restore paths are in use |
| Osaurus integration | `VMLXServiceBridge` is wired into `ChatEngine`, settings, model discovery, and cache inspector |
| Cache stack fundamentals | Paged cache, memory cache, prefix cache, disk cache, SSM companion cache, and cache coordinator are implemented |
| Tool/reasoning parsing | VMLX tool/reasoning parsers are wired through the bridge and app settings; MLX reasoning overrides still apply on the app streaming path |

### Implemented But Still Under Active Validation

| Area | Status |
|------|--------|
| NemotronH | Native model class landed and recent fixes corrected Mamba2 scan, MoE routing, latent projections, and attention projection dimensions |
| Mistral Small 4 | Native MLA + MoE path landed and recent fixes corrected config decoding and inference alignment with Python reference |
| TurboQuant | Swift encode/decode and cache-store/fetch preservation path exist; the live runtime still uses a decode-once float cache after prefill rather than a persistent `TurboQuantKVCache` object |

### Important Current Caveats

- The runtime still executes one active generation task at a time. Scheduler and batching primitives exist, but true multi-request continuous batching is not the active hot path yet.
- Hybrid partial hits now attempt boundary-aligned SSM re-derive when attention KV exists but the matching SSM companion state is missing. The current request still full-prefills when re-derive is pending or unavailable.
- `SSMReDeriver` is now on the live VMLX hybrid recovery path, but it is still fallback-heavy for large or unavailable checkpoints.
- Hybrid prefill currently uses single-phase prefill plus a post-prefill SSM snapshot. Earlier two-phase checkpointing plans were backed out from the hot path after SSD/Mamba2 hangs.
- VMLX paged cache is still a block-level prefix reconstruction layer, not a live paged runtime KV allocator.
- Vision preprocessing and embedding cache exist; full vision encoder inference is still pending.
- MiniMax tokenizer incompatibilities still block trustworthy text generation quality.
- Qwen 3.5 35B JANG still needs a deeper MoE-path optimization pass. Adaptive prefill chunking is in, but the quantized expert prefill path remains the current bottleneck.

---

## What The Engine Actually Does Today

### Cache Stack

- Paged KV cache via block hash chains for the main prefix-reuse path
- Memory LRU cache and token-trie prefix cache
- Disk L2 safetensors cache with SQLite index
- SSM companion cache for hybrid models
- `gen_prompt_len` stripping so cache keys ignore assistant-generation suffix tokens
- When paged cache is off, prefix cache is now populated even if the memory tier is on
- Paged cache and disk cache can now preserve `.compressedAttention` entries on the normal reuse path instead of always degrading them to float

### Hybrid SSM Behavior

- Hybrid caches are represented explicitly as mixed `.attention` and `.ssm` layers
- SSM state is treated as path-dependent and non-truncatable
- Prefix-style reuse is safe only when the KV and SSM sides agree on the same boundary
- Hybrid exact-hit replay and hybrid partial-hit reuse now attempt boundary-aligned SSM re-derive before falling back
- The current request still full-prefills when the needed SSM checkpoint is unavailable or only arrives asynchronously

### TurboQuant Reality

- Swift implementations exist for `TurboQuantEncoder`, `TurboQuantKVCache`, `EncodedKeys`, `EncodedValues`, and `TQDiskStore`
- TurboQuant policy is now resolved from `TurboQuantConfig`, so hybrid JANG models keep the rule "KV only, never SSM"
- The live runtime can compress post-prefill KV, decode once back into float buffers, and use that for lower-memory decode
- Cross-turn cache store/fetch can now preserve `.compressedAttention` through paged, memory, prefix, and disk reuse paths, including cache-hit turns that extend an already-restored prefix
- `TurboQuantKVCache` and `TQDiskStore` still are not the active hot-path cache representation, so the runtime is not yet a fully persistent compressed-KV system

---

## Package Snapshot

```
Packages/VMLXRuntime/
  Sources/VMLXRuntime/   88 source files
  Tests/VMLXRuntimeTests 49 test files
```

Main areas:
- `Core/` model loading, configs, hybrid cache abstractions
- `Cache/` paged, memory, prefix, disk, SSM companion, coordinator
- `Quantization/` JANG loading and TurboQuant pieces
- `Models/` standard, Qwen3.5, NemotronH, Mistral4, GPT-OSS, MLA, MoE, SSM utilities
- `Integration/` runtime actor and Osaurus-facing service layer

---

## Osaurus Integration

The current app wiring is real, not stubbed:

- `Packages/OsaurusCore/Services/Inference/VMLXServiceBridge.swift`
- `Packages/OsaurusCore/Services/Chat/ChatEngine.swift`
- `Packages/OsaurusCore/Views/Settings/ConfigurationView.swift`
- `Packages/OsaurusCore/Views/Model/ModelCacheInspectorView.swift`
- `Packages/OsaurusCore/Managers/Model/ModelManager.swift`

That integration covers:
- model discovery and de-duplication with MLX models
- model load/unload routing
- Local Inference settings passthrough
- parser overrides, with VMLX reasoning chunks bridged back into the app's `<think>` UI path
- cache stats/unload visibility in the app

---

## Recent Fix Run

Recent branch work has been concentrated on hybrid-model correctness:

- `582a6e8b` fix: verified audit fixes
- `5a7e1315` fix: NemotronH + Mistral4 inference corrections from Python reference
- `44506ac0` fix: single-phase prefill for hybrid models + SSD state projection
- `b479140a` fix: efficient 4D SSD state projection
- `b34f28ea` feat: SSD parallel scan for Mamba2
- `466dcc9a` feat: native NemotronH model class
- `8a0a7d05` feat: TurboQuant decode-once lifecycle

---

## Build

```bash
cd Packages/VMLXRuntime && swift build
xcodebuild test -scheme VMLXRuntime -destination 'platform=macOS'
open osaurus.xcworkspace
```

---

## Documentation

- [Current Architecture](docs/ARCHITECTURE.md)
- [Current Feature Comparison](docs/FEATURE_COMPARISON.md)
- [Historical Implementation Plan](docs/plans/2026-03-29-vmlx-runtime-integration.md)

---

## Credits

- VMLXRuntime: Jinho Eric Jang
- Osaurus: [osaurus-ai](https://github.com/osaurus-ai/osaurus)
- MLX: Apple
- JANG Quantization: [JANGQ-AI](https://huggingface.co/JANGQ-AI)
