# Remaining Phases: TurboQuant, SSM Re-Derivation, Settings Integration

**Date:** 2026-03-30
**Status:** Planning (not yet implemented)

---

## Phase 5: TurboQuant Integration

### What Exists
- `TurboQuantKVCache.swift` — two-phase cache (fill → compress), encode/decode via TurboQuantEncoder
- `TurboQuantEncoder.swift` — random projection quantization, codebook generation, QJL sign correction
- `EncodedKeys.swift` / `EncodedValues.swift` — compressed data structures
- `TurboQuantConfig.swift` — per-layer bit configuration (keyBits, valueBits)
- `JangLoader.swift` — builds TurboQuantConfig from jang_config.json
- UI toggle in ConfigurationView — "TurboQuant (3-bit KV compression)"
- Settings flow: ConfigurationView → ServerConfiguration → RuntimeConfig → VMLXServiceBridge → VMLXRuntimeActor.scheduler.config.enableTurboQuant

### What's Missing

#### 5.1 VMLXKVCache conformance for TurboQuantKVCache
**Current:** TurboQuantKVCache is standalone. Model forward pass uses VMLXKVCacheSimple.
**Needed:** Either wrap TurboQuantKVCache to conform to VMLXKVCache, or use TurboQuantKVCache as a post-processing layer that wraps the standard cache.

**Recommended approach (post-processing):**
- Keep standard VMLXKVCacheSimple for the forward pass
- After prefill, extract KV state from VMLXKVCacheSimple into TurboQuantKVCache
- Call compress()
- On cache store, store compressed EncodedKeys/EncodedValues (5x smaller)
- On cache fetch, decompress into decoded buffer, load into VMLXKVCacheSimple

**Why not wrap:** TurboQuant decoding is O(n*d^2) per step per layer if done naively. The forward pass needs raw float keys/values for SDPA. Wrapping TurboQuantKVCache around VMLXKVCacheSimple would require decoding on every update() call.

#### 5.2 Decoded buffer cache (post-compress speed fix)
**Problem:** TurboQuantKVCache.getKeys() re-decodes ALL compressed tokens every call.
**Fix:** After compress(), decode once into _decoded_k_buffer and _decoded_v_buffer. Subsequent getKeys()/getValues() reads from these float buffers (no re-decode). New tokens append to a separate float window. Return concat(decoded_buffer, float_window).

**This is the CRITICAL fix from Python VMLX Phase 2.** Without it, post-compress generation drops to ~2.5 tok/s (from 50+ tok/s).

#### 5.3 Integration with cache store/fetch
**Store path:** After extracting HybridCache from cache objects, check if TurboQuant is enabled. If yes, compress attention layers before storing.
**Fetch path:** After reconstructing HybridCache, check if data is compressed. If yes, decompress into decoded buffers before loading into VMLXKVCacheSimple.

**Requires:** New LayerCacheEntry case `.compressedAttention(EncodedKeys, EncodedValues, offset)` to distinguish compressed from float data in HybridCache/CacheBlock storage.

#### 5.4 Per-model-type considerations
| Model Type | TurboQuant Behavior |
|------------|-------------------|
| Standard LLM (Llama, Qwen2) | Compress all attention layers uniformly |
| MoE (MiniMax, Qwen3 MoE) | Compress expert attention. Gate/router stays float. |
| Hybrid SSM (Qwen3.5) | Compress attention layers only. SSM state NOT compressed. |
| MLA (Mistral 4, DeepSeek-V3) | kv_lora_rank > 0: compressed latent, H=1. Need to verify TQ works with H=1 shape. |
| VL (Qwen3.5-VL) | Same as hybrid — vision embeddings are pre-attention, not in KV cache. |

#### 5.5 Sink token preservation
First 4 tokens (BOS/system prompt start) stay at full precision during compression. These are the "attention sinks" that all subsequent tokens attend to heavily. Compressing them degrades quality.

#### 5.6 JANG-gating
TurboQuant only activates when:
1. scheduler.config.enableTurboQuant is true (UI toggle)
2. Model has jang_config.json (JANG model)
3. container.turboQuantConfig is non-nil (JangLoader built config)

Regular MLX models DO NOT get TurboQuant — they use standard KV cache.

### Implementation Steps
```
5.1  [x] Add .compressedAttention case to LayerCacheEntry (+ all 15 switch statements updated)
5.2  [x] Add decoded buffer fields to TurboQuantKVCache (_decodedKeyBuffer, _decodedValueBuffer)
5.3  [x] Update compress() to populate decoded buffers after encoding
5.4  [x] Update getKeys()/getValues() to return decoded buffer (no re-decode)
5.5  [x] Add TurboQuant compress step in VMLXRuntimeActor cache store (gated by enableTurboQuant + turboQuantConfig)
5.6  [x] Add TurboQuant decompress in cache fetch (detect .compressedAttention, decode to float)
5.7  [x] Add sink token preservation (first 4 tokens at full precision during compression)
5.8  [ ] Test: verify 5x memory reduction with MiniMax JANG
5.9  [ ] Test: verify no quality degradation (compare output with/without TQ)
5.10 [ ] Test: verify post-compress decode speed (should match pre-compress)
```

### Files Changed (Phase 5)
- `Core/LayerCache.swift` — `.compressedAttention(EncodedKeys, EncodedValues, Int)` enum case
- `Core/HybridCache.swift` — `materialized()` evals compressed + sink arrays
- `Cache/CacheCoordinator.swift` — Decompress in reconstruct + paged store; seed propagation
- `Cache/DiskCache.swift` — Store/fetch compressed_attention with TQ tensors + sink data + seed
- `Integration/VMLXRuntimeActor.swift` — Compress on store, decompress on fetch, Sampler integration
- `Quantization/TurboQuantEncoder.swift` — Sink token extraction (first N), prepend on decode
- `Quantization/EncodedKeys.swift` — `sinkData` field, `seed` field, updated `estimatedBytes`
- `Quantization/EncodedValues.swift` — `sinkData` field, `seed` field, updated `estimatedBytes`
- `Quantization/TurboQuantKVCache.swift` — Fixed `estimatedBytes` to include all buffers

---

## Phase 6: SSM Async Re-Derivation

### What Exists
- `SSMReDeriver.swift` — actor with requestReDerive(), sync/async decision logic, dedup
- `ModelForwardPass` protocol — for running prefill during re-derivation
- SSMCheckpoint struct — stores SSM states at boundary

### What's Missing

#### 6.1 SSM state extraction from cache objects
**Problem:** SSMReDeriver.requestReDerive() creates checkpoints with `ssmStates: []` (empty).
**Fix:** After running the forward pass (prefill), extract SSM state from the cache objects (VMLXMambaCache) and populate the checkpoint.

**Requires:** The re-deriver needs access to the cache objects after the forward pass. The current ModelForwardPass protocol doesn't return cache state.

#### 6.2 Model wiring
**Problem:** SSMReDeriver.setModel() is never called.
**Fix:** In VMLXRuntimeActor.loadModel(), create SSMReDeriver and wire the model.

#### 6.3 Thinking model constraint
For reasoning models (Qwen3.5 with enable_thinking), SSM state carries reasoning context. Using stale/missing SSM state during thinking produces garbage reasoning. Sync re-derivation is mandatory — must block until SSM state is computed.

For non-thinking generation, async re-derivation is acceptable — the model can start generating with attention-only KV cache while SSM state is computed in the background. Once ready, inject into the cache.

#### 6.4 Per-model-type considerations
| Model Type | Re-Derivation Needed? |
|------------|----------------------|
| Standard LLM (Llama, Qwen2) | NO — no SSM layers, no re-derivation needed |
| MoE (MiniMax) | NO — pure attention, no SSM |
| Hybrid SSM (Qwen3.5, Nemotron) | YES — SSM state is path-dependent |
| MLA (Mistral 4) | NO — MLA is attention-based, no SSM |
| VL+Hybrid (Qwen3.5-VL) | YES — same as hybrid, vision embeddings don't affect SSM |

### Implementation Steps
```
6.1  [x] SSMReDeriver redesigned to use VMLXModelContainer directly
      Removed dead ModelForwardPass protocol. Re-deriver now runs
      container.forward() + extracts SSM states from VMLXMambaCache.
6.2  [x] SSM state extraction from VMLXMambaCache implemented in requestReDerive()
6.3  [x] Wire SSMReDeriver.setModel(container) in loadModel()
6.4  [x] Safe fallback on .partialHit: full prefill re-derives SSM.
      CacheCoordinator.store() saves SSM companion → next turn gets full hit.
6.5  [x] Re-deriver sync path: block until SSM state ready via Task.value
6.6  [x] Re-deriver async path: fire-and-forget Task stores checkpoint when done
6.7  [x] Sync/async decision: shouldSyncReDerive(tokenCount:) < syncThreshold
6.8  [ ] Test: verify Qwen3.5 hybrid cache hit with SSM re-derivation
6.9  [ ] Test: verify thinking quality with re-derived SSM state
```

### Zombie Code Removed (Phase 6)
- `GenerationEngine.swift` — entire file deleted (stub generate(), never called)
- `GenerationEngineTests.swift` — deleted (tested removed code)
- `ModelForwardPass` protocol — deleted (used [MLXArray] cache, incompatible with [VMLXKVCache])
- `TransformerModelForwardPass` class — deleted from TransformerModel.swift
- `HybridTransformerModelForwardPass` class — deleted from HybridTransformerModel.swift
- `GenerationConfig`, `GenerationResult` structs — deleted (self-referencing only)

### SSM Re-Derivation Flow
SSMReDeriver now uses VMLXModelContainer directly:
1. `requestReDerive(tokens:stableBoundary:)` called on .partialHit
2. Runs `container.forward()` on tokens up to stableBoundary
3. Extracts SSM states from VMLXMambaCache objects in resulting cache
4. Stores SSMCheckpoint in SSMStateCache for future hits
5. Primary path: CacheCoordinator.store() saves SSM companion after generation (self-healing)

---

## Phase 7: Settings Integration + Edge Cases

### What Exists (UI → Engine settings flow)
```
ConfigurationView (SwiftUI)
  → ServerConfiguration (UserDefaults, Codable)
    → RuntimeConfig.snapshot() (auto-detection + user overrides)
      → VMLXServiceBridge.applyRuntimeConfig()
        → VMLXService.applyUserConfig()
          → VMLXRuntimeActor.applyUserConfig()
            → SchedulerConfig fields
              → CacheCoordinator (via toCacheCoordinatorConfig())
```

### Settings That Exist in UI
| Setting | UI Control | Config Field | VMLXRuntime Field |
|---------|-----------|-------------|-------------------|
| KV Cache Bits | Stepper (2/4/8/none) | genKVBits | scheduler.config.kvCacheQuantization |
| KV Group Size | Stepper (32/64/128) | genKVGroupSize | scheduler.config.kvCacheGroupSize |
| Max Context | Stepper (2K-64K) | genMaxKVSize | scheduler.config.maxNumBatchedTokens |
| Prefill Step | Stepper (512-8192) | genPrefillStepSize | scheduler.config.prefillStepSize |
| TurboQuant | Toggle | enableTurboQuant | scheduler.config.enableTurboQuant |
| Disk Cache | Toggle | enableDiskCache | scheduler.config.enableDiskCache |
| Memory Budget | Slider (10%-60%) | N/A (tempCacheMemoryPercent) | scheduler.config.cacheMemoryPercent |
| Model Eviction | Segmented picker | N/A | separate (ModelEvictionPolicy) |

### Settings NOT in UI but needed
| Setting | Why Needed | Default |
|---------|-----------|---------|
| Paged Cache Enable | Block-level prefix sharing | false (should be true for multi-turn) |
| Paged Block Size | Tokens per block | 64 |
| Max Cache Blocks | Memory cap for paged cache | 1000 |
| SSM Max Entries | LRU cap for SSM companion | 50 |
| Disk Cache Directory | Where to store L2 files | ~/.osaurus/cache/<model_hash>/ |
| Disk Cache Max Size | GB cap | 10.0 |

### Edge Cases by Model Type

#### Standard LLM (Llama 3, Qwen 2.5, Qwen 3)
- No SSM layers → isHybrid = false
- Cache hit: restore KV → prefill remaining + genSuffix
- Cache store: store all layers (all attention)
- TurboQuant: applicable to all layers
- KV bits: applicable (q4/q8 KV cache quantization)
- Works with all cache tiers (memory, paged, disk)

#### MoE (MiniMax M2.5, Qwen MoE)
- No SSM layers → isHybrid = false
- Large expert count (>=256) → bfloat16 conversion applied
- Float32 gate routing → prevents overflow
- e_score_correction_bias → selection not weighting
- Cache hit: same as standard LLM
- TurboQuant: applicable but gate weights excluded

#### Hybrid SSM (Qwen3.5, Nemotron, Jamba)
- Mixed layers → isHybrid = true
- Cache hit with SSM companion → full restore (KV + SSM)
- Cache hit without SSM companion → partialHit → full prefill (safe) or re-derive (Phase 6)
- SSM companion stored via CacheCoordinator.store (boundary matches KV key)
- Paged blocks: SSM in last block only, nil in non-last (Bug 2 guard)
- TurboQuant: attention layers only, SSM layers excluded
- KV bits: attention layers only

#### MLA (Mistral 4, DeepSeek-V3)
- kv_lora_rank > 0 → compressed latent, n_kv_heads = 1
- Cache block reconstruction: validate head count = 1 (not num_attention_heads)
- Currently: Mistral 3/4 rejected as unsupported (FP8 quantization)
- Future: When FP8 support added, MLA head count needs special handling in paged cache

#### VL (Qwen3.5-VL, Mistral Small 4 VL)
- Vision embeddings processed separately (not in KV cache)
- Chat template may include image placeholders → affects tokenization
- gen_prompt_len stripping still works (strips assistant header, not image tokens)
- Cache key may not match across turns due to VLM tokenizer image placeholder divergence (EXPECTED per Python VMLX S25-10)
- TurboQuant: same as underlying text model

#### Thinking Models (Qwen3 with enable_thinking, DeepSeek-R1)
- gen_prompt_len includes think tag tokens
- SSM re-derivation MUST be sync (thinking context in SSM state)
- thinkingBudget = maxTokens / 2 (caps thinking tokens)
- enable_thinking flag flows through: request.enableThinking → SamplingParams → chat template → generation loop

### Settings Interaction Matrix

| Setting | Standard | MoE | Hybrid | MLA | VL |
|---------|----------|-----|--------|-----|-----|
| KV Bits (q4/q8) | YES | YES | attention only | YES (H=1) | YES |
| TurboQuant | YES | YES (not gate) | attention only | VERIFY | YES |
| Disk Cache | YES | YES | YES (stores SSM too) | YES | YES |
| Memory Budget | YES | YES | YES | YES | YES |
| Paged Cache | YES | YES | YES (Bug 2 guard) | VERIFY (H=1) | YES |
| Prefill Step | YES | YES | YES | YES | YES |
| Max Context | YES | YES | YES | YES | YES |

### applyUserConfig Timing Issue — RESOLVED
**Problem:** `applyRuntimeConfig()` is called AFTER model load. CacheCoordinator built at init with defaults.

**Fix applied:** `Scheduler.rebuildCacheCoordinator()` — called at end of `applyUserConfig()`.
Rebuilds CacheCoordinator from current config, sets hybrid flag. Loses cached data (acceptable —
settings changes typically happen at model load, before any cache is populated).

### Implementation Steps
```
7.1  [x] Fix applyUserConfig timing: rebuildCacheCoordinator() after config changes
7.2  [x] Default paged cache to true (SchedulerConfig.usePagedCache default changed)
7.3  [x] Disk cache directory auto-configuration in applyUserConfig()
7.4  [x] KV bits setting consumed: VMLXQuantizedKVCache(bits:groupSize:) created
      when kvCacheQuantization != "none". Quantizes on update(), dequantizes for SDPA.
      container.newCache(kvBits:kvGroupSize:) replaces VMLXKVCacheSimple with quantized.
7.5  [x] Memory budget: cacheMemoryPercent flows end-to-end from UI to MemoryCache
7.6  [ ] Test: change settings mid-session, verify new settings take effect
7.7  [ ] Test: each model type with each cache setting combination
7.8  [x] nemotron_h added to unsupportedTypes with clear error message
      (Mamba2 architecture, needs dedicated model class — different from GatedDeltaNet)
```

### Files Changed (Phase 7)
- `Scheduler/SchedulerConfig.swift` — Default `usePagedCache = true`
- `Models/Utilities/KVCache.swift` — Added `VMLXQuantizedKVCache` class (q4/q8 KV quantization)
- `Core/ModelContainer.swift` — Added `newCache(kvBits:kvGroupSize:)` factory
- `Integration/VMLXRuntimeActor.swift` — Wire KV quantization from scheduler config; Sampler integration; removed unused vars
- `Integration/VMLXService.swift` — Pass through `cacheMemoryPercent`, `usePagedCache`
- `OsaurusCore/Services/Inference/VMLXServiceBridge.swift` — Pass `cacheMemoryPercent` from ServerConfiguration
- `Models/ModelRegistry.swift` — nemotron_h in unsupportedTypes with Mamba2 error message

### Zombie Code Removed (Phase 7)
- `GenerationEngine.swift` — removed (stub generate(), GenerationConfig, GenerationResult, ModelForwardPass)
- `GenerationEngineTests.swift` — removed
- `TransformerModelForwardPass` — removed from TransformerModel.swift
- `HybridTransformerModelForwardPass` — removed from HybridTransformerModel.swift
- Unused `topP` variable and `repetitionPenalty` TODO — replaced with Sampler integration

### Sampling Now Uses Sampler Class
VMLXRuntimeActor.generateStream() now uses Sampler for all token sampling:
- First token: `Sampler.sample(logits:params:)` (full pipeline: top-p, top-k, min-p, rep penalty)
- Decode loop: inline filtering (`topPFilter`, `topKFilter`, `applyRepetitionPenalty`) preserving
  double-buffered asyncEval pattern (Sampler.sample() would force materialization, breaking pipelining)

---

## Summary: Implementation Status (updated 2026-03-30)

| Phase | Status | Remaining |
|-------|--------|-----------|
| 5. TurboQuant | **ALL CODE DONE** | 3 model-testing items |
| 6. SSM Re-Derivation | **ALL CODE DONE** | 2 model-testing items |
| 7. Settings Integration | **ALL CODE DONE** | 2 model-testing items |

**All code complete. Only model-level functional testing remains (requires running models).**

**Everything wired end-to-end:**
- TurboQuant: encode with sink preservation → store → fetch → decode with seed → prepend sinks
- SSM: store companion → fetch with boundary match → inject on hit → self-heal on miss
- KV quantization: UI stepper → config → VMLXQuantizedKVCache → quantize on update → dequantize for SDPA
- Paged cache: decompress TQ → slice blocks → reconstruct → SSM in last block
- Disk cache: N-1 key fix → serialize all types → deserialize with seed + sink data
- Sampling: Sampler class handles top-p, top-k, min-p, repetition penalty
- Settings: all UI controls wired to engine via rebuildCacheCoordinator()
