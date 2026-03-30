# VMLXRuntime Architecture

**Package:** `Packages/VMLXRuntime/`
**Target:** Replace Osaurus's `mlx-swift-lm` backend with native Swift VMLX engine
**Repo:** https://github.com/jjang-ai/jangosaurus

---

## 1. System Overview

```
Osaurus App (SwiftUI)
  |
  v
VMLXService (ToolCapableService protocol)          <-- drop-in for MLXService
  |
  v
VMLXRuntime Actor (singleton)                       <-- replaces ModelRuntime
  |
  +-- ModelContainer (model + tokenizer + config)
  +-- Scheduler (continuous batching)
  +-- CacheCoordinator (5-layer cache stack)
  +-- GenerationEngine (prefill + decode loop)
  +-- SSMReDeriver (async re-derive for thinking models)
  |
  v
mlx-swift (tensor ops only, Metal GPU)
```

---

## 2. Core Type System

### 2.1 Unified Cache Abstraction
- [x] **LayerCacheEntry** -- enum: `.attention(KVCacheLayer)` | `.ssm(SSMStateLayer)`
  - [x] **KVCacheLayer** -- attention layer KV cache (keys, values, offset)
    - Positional: can be sliced/truncated
    - TQ-compressible (3-bit via TurboQuant)
    - Stored in paged blocks
  - [x] **SSMStateLayer** -- SSM cumulative state (`[MLXArray]`)
    - Path-dependent: CANNOT be truncated
    - NOT TQ-compressible
    - Stored in SSM companion cache
- [x] **HybridCache** -- `[LayerCacheEntry]` wrapper
  - `isHybrid` / `isPureAttention` / `isPureSSM` introspection
  - `canTruncate` -- safety gate (false if any SSM layer)
  - `truncated(to:)` -- returns nil for hybrid models
  - `materialized()` -- forces lazy MLXArray computation before cache storage
  - `fromPattern(_:kvFactory:ssmFactory:)` -- build from layer type blueprint
- [x] **LayerType** -- enum: `.attention` | `.ssm` | `.expert`
- [x] **parseHybridPattern(_:)** -- "MMM*" string to `[LayerType]`

### 2.2 Request/Response Types
- [x] **InferenceRequest** -- full request lifecycle
  - Prompt token IDs, sampling params, priority
  - Status machine: waiting -> running -> finished
  - Cache state: promptCache, cachedTokens, remainingTokenIds
  - Paged cache: blockTableIds, sharedPrefixBlocks
  - Multimodal: pixelValues, imageGridTHW, attentionMask
  - Thinking: enableThinking, reasoningEffort
- [x] **SamplingParams** -- temperature, topP, topK, minP, repetitionPenalty, stop sequences
- [x] **RequestOutput** -- per-step streaming delta
- [x] **RequestStatus** -- 6-state enum with `isFinished`
- [x] **FinishReason** -- stop | length | abort | toolCalls
- [x] **CacheDetail** -- diagnostic: full | prefix | paged | disk | memory | +tq

### 2.3 SSM Checkpoint (for thinking models)
- [ ] **SSMCheckpoint** -- mid-prefill SSM state snapshot
  - `ssmStates: [SSMStateLayer]` -- one per SSM layer
  - `boundary: Int` -- token position of checkpoint
  - `tokenHash: String` -- SHA-256 of tokens[:boundary]
  - Safe to store/fetch because key matches truncated KV cache key

---

## 3. Cache Stack (5 Layers)

All cache layers are hybrid-SSM-aware from day one. The `LayerCacheEntry` enum ensures every operation handles both attention and SSM layers correctly.

### 3.1 Paged Cache (L1 -- Block-Level)
- [x] **CacheBlock** -- fixed-size block with reference counting
  - `blockId`, `blockSize`, `refCount`, `tokenCount`
  - `blockHash: BlockHash?` -- content-based SHA-256 for prefix lookup
  - `cacheData: [(keys: MLXArray, values: MLXArray)]?` -- per-layer attention KV
  - COW: `isShared` when refCount > 1
  - `computeBlockHash(parentHash:tokenIds:extraKeys:)` -- hash chain algorithm
- [x] **BlockHash** -- 32-byte SHA-256 hash (Hashable, Sendable)
- [x] **FreeBlockQueue** -- O(1) doubly-linked list (LRU front, MRU back)
  - Sentinel head/tail nodes
  - `popleft()`, `popleftN(_:)`, `append(_:)`, `remove(_:)`
- [x] **BlockHashMap** -- hash -> CacheBlock dictionary for O(1) prefix lookup
  - `getBlock(hash:)`, `insert(hash:block:)`, `pop(hash:blockId:)`
- [x] **BlockTable** -- request -> [blockId] mapping with token count
- [x] **PagedCacheManager** -- thread-safe block pool manager
  - `OSAllocatedUnfairLock` for concurrency
  - Block 0 = null sentinel (never freed)
  - `allocateBlock()` / `allocateBlocksByTokens(_:)`
  - `freeBlock(_:)` -- ref count decrement, return to pool at 0
  - `forkBlock(_:hash:)` -- COW: increment refCount (no copy)
  - `markCached(block:hash:)` / `findCachedBlock(hash:)` -- prefix reuse
  - `registerBlockTable(_:blockIds:)` / `deleteBlockTable(_:)` -- request lifecycle
  - LRU eviction of hash-cached blocks when pool exhausted
  - **CacheStats** -- totalBlocks, allocated, free, hits, misses, COW copies, evictions

### 3.2 Prefix Cache (L1 -- Token Trie)
- [ ] **PrefixCache** -- trie-based token sequence matching
  - Nested dict trie: `{modelKey: {tok1: {tok2: {..., "cache": entry}}}}`
  - LRU eviction via OrderedDict equivalent
  - **Fetch modes:**
    - Exact match: all tokens cached
    - Shorter prefix: cache is prefix of request
    - Longer prefix: request is prefix of cache (truncate)
  - Truncation safety: returns nil if HybridCache.canTruncate == false
  - Max entries configurable (default 100)

### 3.3 Block-Aware Prefix Cache (L1 -- Hybrid-Safe)
- [ ] **BlockAwarePrefixCache** -- block-level prefix cache
  - Uses PagedCacheManager for block allocation
  - Handles truncation at block boundaries (safe for hybrid)
  - Block hash chain for O(1) prefix matching
  - `fetch_cache(requestId:tokens:)` -> (BlockTable?, remaining tokens)
  - `store_cache(requestId:tokens:kvCache:)` -> Bool
  - `release_cache(requestId:)` -- free blocks, decrement refs

### 3.4 Memory-Aware Cache (L1 -- RAM LRU)
- [ ] **MemoryCache** -- RAM-budget-aware LRU
  - **MemoryCacheConfig:**
    - `maxMemoryMB: Int?` -- explicit limit (nil = auto)
    - `maxMemoryPercent: Float` -- 30% of available RAM
    - `maxEntries: Int` -- 1000 hard limit
    - `ttlMinutes: Float` -- 0 = disabled
  - Memory pressure adaptation: check `os_proc_available_memory()` every 60s
    - Available < 20%: shrink budget to available/2
    - Available >= 20%: restore to base budget
  - Token-keyed OrderedDict with LRU eviction
  - Prefix scan for partial matches (O(n) intentional tradeoff)
  - Truncation safety: returns nil for hybrid models

### 3.5 Disk Cache (L2 -- SSD Persistence)
- [ ] **DiskCache** -- SQLite-indexed safetensors storage
  - **SQLite schema:** `cache_entries(token_hash PK, file_name, num_tokens, file_size, created_at, last_accessed, access_count, metadata)`
  - WAL mode for concurrent reads
  - Connection pool (max 4)
  - Token hash: SHA-256 of JSON-serialized token array
  - **Background writer:** pre-serialize MLXArrays on calling thread (Metal safety), write on background Task.detached
  - LRU eviction by `last_accessed` when over max_size
  - **TQ-native format detection:** `__tq_native__` metadata marker
  - Max size configurable (default 10 GB)

### 3.6 TurboQuant Disk Store (L2 -- 26x Compressed)
- [ ] **TQDiskStore** -- TQ-native serialization format
  - Stores EncodedKeys/EncodedValues directly (packed uint32 + float16 norms)
  - 26x smaller than float16 (40KB vs 1MB per 100 tokens)
  - safetensors format with `__tq_native__` metadata
  - Per-layer metadata: shape, bit widths, offsets, dims
  - `serialize_tq_cache(cache:)` -> (tensors, metadata)
  - `deserialize_tq_cache(tensors:metadata:)` -> HybridCache

### 3.7 SSM State Cache (Hybrid Companion)
- [ ] **SSMStateCache** -- LRU companion for SSM layer state
  - Max 50 entries, keyed by token hash + boundary position
  - Deep-copy on fetch (SSM state is mutable)
  - **Critical invariant:** empty `[MLXArray]` == MISS, not just nil
  - Stores `SSMCheckpoint` objects for thinking model support
  - Disk checkpoint support for persistence across eviction

### 3.8 SSM Re-Deriver (Async Recovery)
- [ ] **SSMReDeriver** -- actor for async SSM state recovery
  - **When:** KV blocks cached but SSM checkpoint evicted
  - **How:** Run full forward pass on cached tokens (all layers, not just SSM)
  - **Cost:** = full prefill, but amortized across turns via background execution
  - **Side effect:** Refreshes attention KV (TQ compressed) during re-derive
  - Decision logic: sync for < 512 tokens, async + full prefill for longer
  - Deduplicates concurrent re-derive requests for same token hash
  - Stores SSMCheckpoint at stable boundary

### 3.9 Cache Coordinator (Orchestrator)
- [ ] **CacheCoordinator** -- unified fetch/store interface
  - **CacheCoordinatorConfig:**
    - enablePrefixCache, usePagedCache, useMemoryAwareCache
    - enableDiskCache, diskCacheMaxGB, diskCacheDir
    - pagedBlockSize (64), maxCacheBlocks (1000)
    - cacheMemoryPercent (0.30)
  - **Fetch cascade:**
    1. Paged cache (block hash chain) -> attention KV
    2. If hybrid: SSM checkpoint fetch
       - HIT: combine KV + SSM = full HybridCache
       - MISS: check disk checkpoint -> if MISS: re-derive decision
    3. Memory cache (token-trie LRU)
    4. Disk cache (SQLite -> safetensors)
    5. MISS -> full prefill
  - **Store cascade:**
    1. Memory cache (hot tier)
    2. If TQ: TQ-native to disk (26x). Else: background safetensors
    3. Paged blocks: mark with content hashes
    4. If hybrid: SSM checkpoint at stable boundary
    5. If thinking model: mid-prefill checkpoint (before gen_prompt_len)

---

## 4. TurboQuant (3-bit KV Compression)

### 4.1 TurboQuantConfig
- [ ] Per-layer bit widths with critical layer overrides
  - Default: 3-bit keys, 3-bit values
  - Critical layers (first/last 3): 4-bit
  - `keyBits(forLayer:totalLayers:)` -- returns nil for SSM layers
  - Hybrid-aware: skips SSM layers entirely
  - MLA support: custom key_dim/value_dim for kv_lora_rank > 0

### 4.2 Encoded Keys & Values
- [ ] **EncodedKeys** -- packed codebook indices (uint32), QJL sign bits, residual norms, vector norms
- [ ] **EncodedValues** -- packed indices (uint32), vector norms
- Stay compressed in GPU memory during decode (zero decompression overhead)

### 4.3 TurboQuantKVCache
- [ ] Two-phase operation:
  - **FILL** (prefill): accumulate float16 KV normally, zero overhead
  - **COMPRESS** (after prefill): compress float -> 3-bit indices+norms
  - **DECODE**: stays compressed, no decompression needed
  - **RECOMPRESS**: when new tokens added, compress only the delta
- [ ] `_compressed_keys: EncodedKeys`, `_compressed_values: EncodedValues`
- [ ] Recompress skips SSM layers (checked via TurboQuantConfig)
- [ ] Dimension validation for MLA/GQA head count mismatch

### 4.4 TurboQuant Encode/Decode
- [ ] Codebook-based vector quantization via MLX Metal ops
- [ ] `encode_keys(keys:config:)` -> EncodedKeys
- [ ] `encode_values(values:config:)` -> EncodedValues
- [ ] `decode_keys(encoded:)` -> MLXArray (for attention computation)
- [ ] `decode_values(encoded:)` -> MLXArray

---

## 5. JANG Model Loading

### 5.1 JangLoader
- [ ] Auto-detect JANG models: search for `jang_config.json`, `jjqf_config.json`, `jang_cfg.json`, `mxq_config.json`
- [ ] **v2 format** (primary): MLX-native safetensors, instant load via mmap
- [ ] **v1 format** (legacy): uint8 repacking to MLX uint32
- [ ] Patch `model.makeCache()` to return TurboQuantKVCache when TQ enabled
- [ ] Detect hybrid models from `layer_types` or `hybrid_override_pattern`
- [ ] Handle MLA models (kv_lora_rank > 0): custom key/value dimensions
- [ ] Gate dequantization for Nemotron-H hybrid SSM models

---

## 6. Continuous Batching Scheduler

### 6.1 SchedulerConfig
- [ ] **Batch sizing:** maxNumSeqs (256), maxBatchedTokens (8192)
- [ ] **Prefill:** prefillBatchSize (8), prefillStepSize (2048)
- [ ] **Decode:** completionBatchSize (32)
- [ ] **Cache flags:** enablePrefixCache, usePagedCache, useMemoryAwareCache
- [ ] **KV quantization:** kvCacheQuantization (none/q4/q8), kvCacheGroupSize (64)
- [ ] **Disk cache:** enableDiskCache, diskCacheDir, diskCacheMaxGB
- [ ] **Block disk cache:** enableBlockDiskCache, blockDiskCacheDir

### 6.2 Scheduler
- [ ] **Request lifecycle:** add -> schedule -> generate -> finish -> cleanup
- [ ] **Scheduling policy:** FCFS with priority support
- [ ] **Waiting queue** (deque) + **Running dict** (active requests)
- [ ] **Cache integration:** CacheCoordinator fetch before batching
- [ ] **Hybrid model detection:** `_isHybrid`, `_hybridKVPositions`, `_hybridNumLayers`
- [ ] **TQ activation:** auto-enable for all MLX models
- [ ] **SSM state cache:** HybridSSMStateCache companion (max 50)

### 6.3 Batch Builder
- [ ] Construct batched input tensors from multiple requests
- [ ] Handle variable-length sequences with padding
- [ ] For hybrid: merge KV caches AND SSM states into batch-aware structures
- [ ] BatchKVCache / BatchMambaCache merge support

### 6.4 MLLM Scheduler
- [ ] Vision-aware: preprocess images/video before scheduling
- [ ] Manage vision embedding cache
- [ ] Handle mixed text+vision batches
- [ ] Forward gen_prompt_len stripping for thinking models

---

## 7. Generation Engine

### 7.1 Sampler
- [ ] Temperature scaling (greedy when T=0)
- [ ] Top-p (nucleus sampling)
- [ ] Top-k filtering
- [ ] Min-p filtering
- [ ] Repetition penalty

### 7.2 Stop Sequence Detector
- [ ] Sliding window matcher
- [ ] Hold back maxStopLen characters, emit safe prefix
- [ ] Cross-token-boundary detection
- [ ] Skip matching inside unclosed `<think>` blocks

### 7.3 Stream Accumulator
- [ ] AsyncSequence: token IDs -> typed events
  - `.tokens(String)` -- normal text output
  - `.toolInvocation(name:argsJSON:)` -- tool call detected
  - `.thinking(String)` -- reasoning content
- [ ] Incremental decode with sliding 8-token context (BPE boundary handling)
- [ ] Tool parser integration
- [ ] Stop sequence detector integration

### 7.4 Generation Engine
- [ ] **Prefill phase:** batched forward pass on uncached tokens
- [ ] **Decode phase:** autoregressive sampling, one token per step per request
- [ ] **Cache reuse:** CacheCoordinator fetch, common prefix detection
- [ ] **TQ recompress:** after prefill, skip SSM layers
- [ ] **Two-phase prefill:** for hybrid models (snapshot at stable boundary)
  - Phase 1: tokens[0:stable_boundary] -> SSM checkpoint
  - Phase 2: tokens[stable_boundary:] (gen_prompt) -> continue
- [ ] **Materialize before store:** force computation on all lazy arrays
- [ ] **Mid-prefill SSM checkpoint:** for thinking models (before gen_prompt_len)

---

## 8. Vision-Language Pipeline

### 8.1 Vision Processor
- [ ] Image: resize max 1024x1024, normalize, CoreImage -> MLXArray
- [ ] Video: smart frame extraction (8-64 frames)
- [ ] PNG/JPEG/WebP support
- [ ] Detail levels: auto/low/high

### 8.2 Vision Embedding Cache
- [ ] Cache preprocessed image embeddings by data hash
- [ ] Avoid re-encoding on cache hit

### 8.3 VLM Model Wrapper
- [ ] Bridge vision encoder + LLM
- [ ] Supported: Qwen-VL, Qwen2.5-VL, Qwen3.5-VL, Pixtral, InternVL, LLaVA, Gemma 3n, Phi-3-Vision
- [ ] Grid THW (temporal, height, width) for variable resolution

---

## 9. Parsers

### 9.1 Tool Call Parsers (14)
- [ ] Protocol: `processToken(_:)` -> text | buffered | toolCall
- [ ] Auto-detect from model name + chat template
- [ ] Implementations:
  - [ ] Qwen (3/2.5/QwQ)
  - [ ] Llama (3/3.1/3.2/3.3/4)
  - [ ] Mistral / Mixtral / Codestral
  - [ ] DeepSeek (V2/V3)
  - [ ] Hermes / NousResearch
  - [ ] Functionary v3
  - [ ] Granite (IBM)
  - [ ] GLM-4.7 / ChatGLM4
  - [ ] MiniMax M2.5
  - [ ] Nemotron
  - [ ] xLAM (Salesforce)
  - [ ] Moonshot Kimi
  - [ ] StepFun Step-3.5
  - [ ] Generic JSON fallback

### 9.2 Reasoning Parsers (4)
- [ ] Extract `<think>...</think>` blocks, separate reasoning from content
- [ ] Implementations:
  - [ ] Qwen3 / Qwen3.5
  - [ ] DeepSeek-R1
  - [ ] GPT-OSS / GLM-4.7
  - [ ] Mistral

---

## 10. Osaurus Integration

### 10.1 ChatMessageMapper
- [ ] Osaurus `ChatMessage` (OpenAI format) -> `InferenceRequest`
- [ ] Handle text, images (base64 data URLs), tool calls, tool results

### 10.2 VMLXRuntime Actor
- [ ] Singleton replacing `ModelRuntime`
- [ ] Owns: model loading, scheduler, cache coordinator, generation engine
- [ ] API: `generateEventStream()`, `respondWithTools()`, `streamWithTools()`

### 10.3 VMLXService
- [ ] Conforms to `ToolCapableService` protocol
- [ ] Drop-in replacement for `MLXService` with id "vmlx"
- [ ] `streamDeltas()`, `generateOneShot()`, `respondWithTools()`, `streamWithTools()`

### 10.4 Osaurus Wiring
- [x] VMLXServiceBridge added to ChatEngine default services (before MLXService for priority)
- [x] Model discovery merges VMLXRuntime models (ModelDetector.scanAvailableModels) with MLXService models
- [x] UI, agents, plugins, server, memory, identity unchanged -- all go through ChatEngine

### 10.5 Integration Map -- How Every Subsystem Connects

All inference in Osaurus flows through `ChatEngine` -> `ModelServiceRouter`. No subsystem
bypasses this path.

**Inference paths:**
- Chat UI -> `ChatView` -> `ChatEngine.streamChat()`
- Sandbox agents -> `HostAPIBridgeServer` -> `ChatEngine.completeChat()`
- Plugins -> `PluginHostAPI` -> `ChatEngine`
- Work mode -> `WorkExecutionEngine` -> `ChatEngine`
- HTTP API -> `HTTPHandler` -> `ChatEngine`
- Memory -> `MemoryService` -> `ModelServiceRouter` directly

**Service priority (ModelServiceRouter.resolve):**
1. Remote providers (matched by prefix, e.g., "openai/gpt-4") -- checked first for non-default models
2. `FoundationModelService` (Apple on-device, macOS 26+) -- handles "default", "foundation", empty
3. `VMLXServiceBridge` (JANG models, "vmlx", "local") -- handles JANG quants via VMLXRuntime
4. `MLXService` (fallback for non-JANG MLX models) -- handles models found in ModelManager

**Model discovery (installedModelsProvider):**
- `VMLXServiceBridge.getAvailableModels()` -- scans `~/.cache/huggingface/hub/`, `~/jang/models/`, `~/models/`, `~/.mlxstudio/models/`, `~/.osaurus/models/`
- `MLXService.getAvailableModels()` -- delegates to `ModelManager.installedModelNames()`
- De-duplicated: VMLX models take priority, MLX models fill gaps

---

## 11. Extended Capabilities (Post-MVP)

### 11.1 Image Generation
- [ ] Flux (Schnell, Dev, Kontext, Krea, Klein)
- [ ] Z-Image (Turbo, 4-bit, 8-bit, full)
- [ ] FIBO / FIBO-Lite
- [ ] Endpoint: `POST /v1/images/generations`

### 11.2 Audio
- [ ] Kokoro TTS -> `POST /v1/audio/speech`
- [ ] Whisper STT -> `POST /v1/audio/transcriptions`

### 11.3 Embeddings & Reranking
- [ ] `POST /v1/embeddings` with dimension control
- [ ] `POST /v1/rerank`

### 11.4 Anthropic Messages API
- [ ] `POST /v1/messages` -- Anthropic format translation
- [ ] Tool calling, thinking blocks, streaming

### 11.5 Model Config Registry
- [ ] 65+ model family configurations
- [ ] Auto-detect reasoning format, tool calling format, tokenizer quirks

---

## 12. Critical Invariants

1. **SSM state is path-dependent** -- never truncate SSM layers. `HybridCache.canTruncate` enforces this.
2. **Materialize before caching** -- `MLXArray` lazy computation must be forced before cache storage, otherwise next access replays the full computation graph.
3. **Pre-serialize on calling thread** -- Metal GPU ops must run on the thread owning the Metal context. Background writers receive pre-serialized data.
4. **TQ recompress skips SSM** -- `TurboQuantConfig.keyBits(forLayer:)` returns nil for SSM layers.
5. **Block ref count on abort** -- `deleteBlockTable()` to prevent OOM from leaked refs.
6. **SSM companion: empty == MISS** -- `[]` is a miss, not just nil.
7. **Two-phase prefill for hybrid** -- if `canTruncate == false`, snapshot at stable boundary.
8. **MLA head count awareness** -- `kv_lora_rank > 0` uses custom key_dim/value_dim.
9. **Mid-prefill SSM checkpoint** -- for thinking models, checkpoint DURING prefill (before gen_prompt_len), not after generation.
10. **Async re-derive = full forward pass** -- SSM layers can't run independently, but cost is amortized via background execution.

---

## Progress Tracker

| Phase | Component | Status | Notes |
|-------|-----------|--------|-------|
| 1.1 | Package Scaffold | DONE | mlx-swift + swift-transformers deps |
| 1.2 | LayerCache + HybridCache | DONE | Hybrid SSM first-class, truncation safety |
| 1.3 | Core Types | DONE | InferenceRequest, SamplingParams, RequestOutput |
| 2.1 | CacheBlock + FreeBlockQueue | DONE | Ref counting, SHA-256 hash chain, O(1) LRU |
| 2.2 | BlockHashMap + BlockTable | DONE | O(1) prefix lookup, request tracking |
| 2.3 | PagedCacheManager | DONE | Block pool, COW, eviction, thread-safe |
| 2.4 | PrefixCache (Trie) | DONE | Exact/shorter/longer prefix, hybrid-safe |
| 2.5 | MemoryCache | DONE | RAM-aware LRU, pressure adaptation, TTL |
| 2.6 | DiskCache (L2 SSD) | DONE | SQLite index, WAL, metadata-only (tensor I/O pending) |
| 2.7 | TQDiskStore | DONE | 26x serialization format, serialize/deserialize roundtrip |
| 2.8 | SSMStateCache + Checkpoint | DONE | Deep-copy, empty==MISS, thinking model checkpoints |
| 2.8b | SSMReDeriver | DONE | Async recovery actor, sync/async decision, dedup |
| 2.9 | CacheCoordinator | DONE | 5-layer orchestration, hybrid-aware fetch/store |
| 3.1 | TurboQuantConfig | DONE | Per-layer bits, critical layers, MLA, hybrid skip |
| 3.2 | EncodedKeys + EncodedValues | DONE | Packed uint32 + float16 norms |
| 3.3 | TurboQuantKVCache | DONE | Two-phase fill/compress lifecycle |
| 3.4 | TurboQuant Encode/Decode | DONE | Interface stubs (Metal kernels deferred) |
| 4.1 | JangLoader | DONE | Auto-detect, config parsing, TQ+hybrid+MLA config |
| 5.1 | SchedulerConfig | DONE | All knobs, RAM auto-detect, cache config bridge |
| 5.2 | RequestQueue | DONE | FCFS, priority, waiting/running lifecycle |
| 5.3 | Scheduler | DONE | Cache-integrated request lifecycle |
| 5.4 | BatchBuilder | DONE | Multi-request tensor batching with padding |
| 5.5 | MLLMScheduler | DONE | Vision-aware, embedding cache, gen_prompt strip |
| 6.1 | Sampler | DONE | Temp/top-p/top-k/min-p/rep penalty (top-p fixed) |
| 6.2 | StopSequenceDetector | DONE | Sliding window, cross-boundary, multi-sequence |
| 6.3 | StreamAccumulator | DONE | Tool+reasoning+stop integration |
| 6.4 | GenerationEngine | DONE | Two-phase prefill, SSM checkpoint, ModelForwardPass protocol |
| 7.1 | VisionProcessor | DONE | CoreImage pipeline, resize, normalize, multi-format |
| 7.2 | VisionEmbeddingCache | DONE | SHA-256 keyed LRU, memory budget |
| 7.3 | VLMModelWrapper | DONE | 7 VLM configs, token strategies, VLMModelProtocol |
| 8.1 | ToolCallParser + Generic | DONE | Protocol + factory auto-detect (fixed) |
| 8.2 | ReasoningParser + ThinkTag | DONE | Protocol + think tag extraction |
| 9.1 | ChatMessageMapper | DONE | OpenAI-compatible types, streaming chunks |
| 9.2 | VMLXRuntime Actor | DONE | Singleton, cache+scheduler, streaming (gen stub) |
| 9.3 | VMLXService | DONE | ToolCapableService protocol, sentinel encoding |
| 9.4 | Osaurus Wiring | DONE | VMLXServiceBridge in OsaurusCore, type mapping |
| 10.5 | ModelConfig Registry | DONE | 30+ families, auto-detect tool/reason/vision/hybrid |
| -- | Audit Fixes | DONE | top-p sampling, parser auto-detect, config mapping |
| -- | JangLoader Rewrite | DONE | Real jang_config.json format, all 7 profiles |
| -- | ModelDetector | DONE | Multi-file detection, 5-dir scanning |
| -- | ModelLoader | DONE | Safetensors sharded loading, tokenizer setup |
| -- | ModelContainer | DONE | Tokenization, chat templates, gen_prompt_len |
| -- | VMLXRuntimeActor Wiring | DONE | Real model loading + tokenization pipeline |
| 13.1 | TransformerModel | DONE | Attention, FFN, RMSNorm, RoPE layer stack, KV cache |
| 13.2 | ModelForwardPass impl | DONE | Prefill + decode via transformer layers, cache connected to CacheCoordinator |
| 13.3 | TQ Metal Kernels | TODO | Codebook quantization via Metal compute |
| 13.4 | Disk Cache Tensor I/O | TODO | Safetensors read/write for MLXArray |
| 13.5 | Video Processing | TODO | AVFoundation frame extraction |

## Known Gaps

1. ~~**Model forward pass**~~ -- DONE. TransformerModel with attention, FFN, RoPE, KV cache. Cache connected to CacheCoordinator via loadCache/exportCache.
2. **TQ Metal kernels** -- TurboQuantEncoder stubs need actual codebook quantization via Metal compute shaders.
3. **Disk cache tensor I/O** -- SQLite index works, safetensors MLXArray serialization pending.
4. **Video processing** -- VisionProcessor.extractFrames() needs AVFoundation.

## Audit Issues Fixed (2026-03-29)

1. **Sampler top-p** -- inverse permutation via argSort was wrong. Replaced with threshold-based approach.
2. **ToolCallParser autoDetect** -- hardcoded GenericToolParser.Type cast. Replaced with factory closure registry.
3. **SchedulerConfig** -- ssmMaxEntries not forwarded to CacheCoordinatorConfig. Added field and mapping.
4. **GenerationEngine** -- missing model parameter. Added `model: (any ModelForwardPass)?`.
5. **VMLXRuntimeActor** -- dead requestQueue + duplicate cacheCoordinator. Now delegates through Scheduler.
6. **JangLoader** -- config structure didn't match real files. Rewritten from actual JANG models on disk.

## Phase 13: Transformer Model (Current Work)

The final piece: building the actual transformer layer stack using mlx-swift's Module building blocks.

### 13.1 Architecture

```
Input token IDs: [Int]
       |
       v
  Embedding (vocab_size x hidden_size)
       |
       v
  For each layer 0..N:
    +-- RMSNorm (pre-attention)
    +-- Attention:
    |     Q = Linear(hidden -> n_heads * head_dim)
    |     K = Linear(hidden -> n_kv_heads * head_dim)
    |     V = Linear(hidden -> n_kv_heads * head_dim)
    |     output = scaledDotProductAttention(Q, K, V, mask)
    |     O = Linear(n_heads * head_dim -> hidden)
    |     + residual connection
    +-- RMSNorm (pre-FFN)
    +-- FFN:
    |     gate = Linear(hidden -> intermediate)
    |     up   = Linear(hidden -> intermediate)
    |     down  = Linear(intermediate -> hidden)
    |     output = down(SiLU(gate) * up)
    |     + residual connection
       |
       v
  RMSNorm (final)
       |
       v
  LM Head: Linear(hidden -> vocab_size)
       |
       v
  Logits: MLXArray [batch, seq_len, vocab_size]
```

### 13.2 Model Families Supported

| Family | Attention | FFN | Norm | Special |
|--------|-----------|-----|------|---------|
| Qwen3.5 | GQA + SSM | SwiGLU | RMSNorm | Hybrid SSM layers |
| Llama 3/4 | GQA | SwiGLU | RMSNorm | Standard transformer |
| Mistral | GQA/MLA | SwiGLU | RMSNorm | Sliding window, MLA |
| Nemotron-H | GQA + SSM | SwiGLU | RMSNorm | Hybrid SSM + MoE |
| MiniMax | GQA + MoE | SwiGLU | RMSNorm | 256 experts |
| DeepSeek | MLA + MoE | SwiGLU | RMSNorm | Latent attention |

### 13.3 KV Cache Integration

During forward pass:
1. **Prefill**: Compute Q, K, V for all tokens. Store K, V in cache.
2. **Decode**: Compute Q for new token only. Concatenate with cached K, V.
3. **TQ**: After prefill, compress cached K, V to 3-bit via TurboQuantKVCache.
4. **Hybrid**: SSM layers update cumulative state (not KV cache).

---

## Code Map -- Every File and Its Role

Total: 42 files, ~8,400 lines.

### Root (1 file)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `VMLXRuntime.swift` | 3 | LIVE | Package version constant |

### Core/ (7 files)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `LayerCache.swift` | 97 | LIVE | KVCacheLayer, SSMStateLayer, LayerCacheEntry enum for unified cache entries |
| `HybridCache.swift` | 169 | LIVE | Multi-layer cache container with truncation safety, materialization, factory |
| `Types.swift` | 211 | LIVE | SamplingParams, InferenceRequest, RequestOutput, FinishReason, CacheDetail |
| `SSMCheckpoint.swift` | 31 | LIVE | SSM state snapshot at stable boundary for thinking model support |
| `ModelConfig.swift` | 225 | LIVE | ModelConfig registry (30+ families), tool/reasoning format detection |
| `ModelContainer.swift` | 152 | LIVE | Wraps LoadedModel with runtime config, tokenization, chat templates |
| `ModelDetector.swift` | 526 | LIVE | Filesystem scanner for model directories, multi-file detection |
| `ModelLoader.swift` | 207 | LIVE | Safetensors sharded loading, tokenizer setup via Hub/Tokenizers |

### Cache/ (12 files)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `CacheBlock.swift` | 53 | LIVE | Fixed-size KV block with ref counting, SHA-256 hash, COW support |
| `BlockHashMap.swift` | 16 | LIVE | Hash-to-block dictionary for O(1) prefix lookup |
| `BlockTable.swift` | 25 | LIVE | Request-to-blockId mapping with token count tracking |
| `FreeBlockQueue.swift` | 45 | LIVE | O(1) doubly-linked list of free blocks (LRU front, MRU back) |
| `PagedCacheManager.swift` | 239 | LIVE | Block pool manager with alloc/free/fork/eviction, thread-safe |
| `PrefixCache.swift` | 202 | LIVE | Trie-based token prefix cache with LRU eviction, hybrid-safe |
| `MemoryCache.swift` | 272 | LIVE | RAM-budget-aware LRU cache with memory pressure adaptation |
| `DiskCache.swift` | 249 | HAS_STUBS | SQLite-indexed L2 cache; index works, tensor I/O pending |
| `TQDiskStore.swift` | 160 | HAS_STUBS | 26x compressed TQ-native safetensors serialization format |
| `SSMStateCache.swift` | 142 | LIVE | LRU companion cache for SSM layer state, deep-copy on fetch |
| `SSMReDeriver.swift` | 165 | HAS_STUBS | Async SSM state recovery actor, dedup, sync/async decision |
| `CacheCoordinator.swift` | 290 | LIVE | Orchestrates fetch/store across all cache layers, hybrid-aware |

### Quantization/ (6 files)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `TurboQuantConfig.swift` | 129 | LIVE | Per-layer bit widths, critical layer overrides, MLA/hybrid skip |
| `EncodedKeys.swift` | 62 | HAS_STUBS | Packed codebook indices + QJL sign bits + norms for compressed keys |
| `EncodedValues.swift` | 48 | HAS_STUBS | Packed codebook indices + norms for compressed values |
| `TurboQuantKVCache.swift` | 165 | HAS_STUBS | Two-phase fill/compress lifecycle for KV cache compression |
| `TurboQuantEncoder.swift` | 80 | HAS_STUBS | Codebook quantization interface; Metal kernel impl deferred |
| `JangLoader.swift` | 397 | LIVE | JANG config parser (7 profiles), hybrid/TQ/MLA auto-detect |

### Generation/ (4 files)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `GenerationEngine.swift` | 232 | LIVE | ModelForwardPass protocol, GenerationConfig/Result, reference generation loop |
| `Sampler.swift` | 143 | LIVE | Token sampling: temperature, top-p, top-k, min-p, repetition penalty |
| `StopSequenceDetector.swift` | 94 | LIVE | Sliding window stop sequence matcher with holdback buffer |
| `StreamAccumulator.swift` | 151 | LIVE | Token stream to typed events (text/thinking/tool) with BPE handling |

### Models/ (1 file)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `TransformerModel.swift` | 509 | LIVE | Full decoder-only transformer: attention, FFN, RoPE, KV cache, forward pass wrapper |

### Scheduler/ (5 files)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `SchedulerConfig.swift` | 214 | LIVE | All scheduling knobs: batch sizes, cache flags, KV quant, disk paths |
| `RequestQueue.swift` | 129 | LIVE | FCFS request lifecycle: waiting -> running -> finished |
| `Scheduler.swift` | 222 | LIVE | Cache-integrated continuous batching scheduler |
| `BatchBuilder.swift` | 163 | HAS_STUBS | Multi-request tensor batching with padding and mask construction |
| `MLLMScheduler.swift` | 180 | HAS_STUBS | Vision-aware scheduler with embedding cache and gen_prompt strip |

### Vision/ (3 files)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `VisionProcessor.swift` | 262 | HAS_STUBS | CoreImage pipeline: resize, normalize, multi-format image preprocessing |
| `VisionEmbeddingCache.swift` | 117 | LIVE | SHA-256 keyed LRU cache for preprocessed vision embeddings |
| `VLMModelWrapper.swift` | 174 | HAS_STUBS | VLM bridge config for 7 architectures, token strategies, grid THW |

### Parsers/ (4 files)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `ToolCallParser.swift` | 66 | LIVE | ToolCallParser protocol, ParsedToolCall, factory auto-detect |
| `ToolParsers/GenericToolParser.swift` | 98 | LIVE | Generic JSON-based tool call parser (fallback for any model) |
| `ReasoningParser.swift` | 41 | LIVE | ReasoningParser protocol, ReasoningResult types |
| `ReasoningParsers/ThinkTagReasoningParser.swift` | 119 | LIVE | Think-tag extraction for Qwen3, DeepSeek-R1, similar models |

### Integration/ (3 files)

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `ChatMessageMapper.swift` | 265 | LIVE | OpenAI-compatible chat types, VMLXChatCompletionRequest, streaming chunks |
| `VMLXRuntimeActor.swift` | 610 | LIVE | Singleton actor: model loading, cache coordination, generation loop |
| `VMLXService.swift` | 228 | LIVE | VMLXModelService protocol, drop-in replacement for MLXService |

---

## Unimplemented Components

| Component | What's Missing | Why | When |
|-----------|---------------|-----|------|
| TurboQuant Metal Kernels | Codebook encode/decode in `TurboQuantEncoder` | Needs Metal compute shaders for performance | Phase 2 |
| Disk Cache Tensor I/O | `DiskCache` safetensors MLXArray read/write | SQLite index works, tensor serialization needs MLX C API | Phase 2 |
| TQ Disk Store Roundtrip | `TQDiskStore` serialize/deserialize with real data | Depends on TQ Metal kernels producing real encoded data | Phase 2 |
| SSM Re-Derivation | `SSMReDeriver` full forward pass for recovery | Needs hybrid model (Mamba layers) in TransformerModel | Phase 3 |
| Video Processing | `VisionProcessor.extractFrames()` | Needs AVFoundation frame extraction | Phase 3 |
| Vision Model Forward Pass | `VLMModelWrapper` actual vision encoder integration | Needs vision encoder weights + embedding merge | Phase 3 |
| BatchBuilder Multi-Request | `BatchBuilder` real multi-sequence padding/merging | Single-request generation works; batching needs scheduler wire-up | Phase 2 |
| MLLM Scheduling | `MLLMScheduler` vision preprocessing pipeline | Needs VisionProcessor + VLMModelWrapper completion | Phase 3 |
