# VMLXRuntime Cache Stack — Full Architecture & Integration Plan

**Date:** 2026-03-30
**Goal:** Rebuild VMLXRuntime's generation loop around the 5-layer cache stack so it matches (and exceeds) the Python VMLX engine's cache behavior. Every component must be wired, tested, and working end-to-end.

**Why this matters:** Raw forward pass speed is bounded by mlx-swift's C API overhead (~6x vs Python per-op). We CANNOT beat mlx-swift-lm on raw decode speed. The ONLY way to be faster is through the cache stack: skip prefill via cache hits, compress KV via TurboQuant, persist to disk for instant warm-up. A cache hit on turn 2+ saves 500ms-5s of prefill — that's where we win.

---

## Architecture Overview

```
Request arrives (messages + model)
  |
  +-- Tokenize + apply chat template
  +-- Compute gen_prompt_len (assistant header tokens to strip)
  +-- Strip gen_prompt_len from token sequence for cache key
  |
  +-- CACHE FETCH CASCADE
  |   |
  |   +-- L1a: Paged Block Cache (BlockAwarePrefixCache)
  |   |   +-- Chain-hash block matching -> reconstruct KV slices
  |   |
  |   +-- L1b: Memory Cache (MemoryAwarePrefixCache)
  |   |   +-- Exact/prefix/reverse match -> full KV cache
  |   |
  |   +-- L2: Disk Cache (DiskCacheManager)
  |   |   +-- SQLite index -> load safetensors -> promote to L1
  |   |
  |   +-- MISS -> full prefill
  |
  +-- HYBRID MODEL CHECK (if cache hit)
  |   |
  |   +-- Fetch SSM companion state (SSMStateCache)
  |   |   +-- HIT: inject SSM + KV -> skip prefill entirely
  |   |   +-- MISS: discard KV cache -> full prefill
  |   |       (SSM state is path-dependent, can't skip)
  |   |       OR: async re-derive SSM while using KV for attention
  |   |
  |   +-- _fix_hybrid_cache: expand KV-only cache to full layer count
  |       (insert fresh ArraysCache at SSM positions)
  |
  +-- PREFILL (only uncached tokens)
  |   |
  |   +-- Chunked prefill (prefillStepSize chunks)
  |   +-- After prefill: capture SSM state for companion cache
  |   +-- After prefill: optionally compress via TurboQuant
  |
  +-- DECODE LOOP (token by token, asyncEval double-buffered)
  |   |
  |   +-- Each step: forward(token) -> sample -> yield
  |
  +-- POST-GENERATION
      |
      +-- Extract cache state (KV layers + SSM layers)
      +-- Truncate to prompt_len - 1 (last token re-fed on hit)
      +-- Strip gen_prompt_len from token key
      +-- Optionally TurboQuant compress before storage
      |
      +-- Store L1a: paged blocks (chain-hashed, ref-counted)
      +-- Store L1b: memory cache (LRU, memory-aware)
      +-- Store L2: disk cache (safetensors, background write)
      +-- Store SSM companion (if hybrid model)
      +-- Done
```

---

## Component Mapping

### A. Cache Key Computation (NEW -- not currently implemented)

**What:** Strip gen_prompt_len tokens from end of token sequence before computing cache key.

**Why:** Chat templates append assistant header tokens (like assistant role prefix or think tags) that differ between turns. Without stripping, the cache key changes every turn and never hits.

**Implementation:**
```
File: VMLXModelContainer.swift
Method: computeGenPromptLen(messages:) -- already exists (line 166)

File: VMLXRuntimeActor.swift (generateStream)
Change: After tokenization, compute gen_prompt_len.
        Create cacheKeyTokens = tokens[0 ..< tokens.count - genPromptLen]
        Use cacheKeyTokens for fetch() and store()
        Use full tokens for actual prefill
```

**Python reference:** scheduler.py:2360-2371, mllm_batch_generator.py:1444-1450

**Nuance:** gen_prompt_len = 0 for models without chat templates. Always safe to strip 0.

---

### B. Cache Fetch -- Partial Hit Handling (BROKEN -- treated as miss)

**Current:** VMLXRuntimeActor.swift:430 groups .partialHit with .miss.

**Fix:** Handle partial hits properly:
```swift
case .partialHit(let attentionCache, let remaining, let detail):
    // Attention KV cached but SSM state missing
    if container.isHybrid {
        // Option 1 (safe): discard, full prefill
        // Option 2 (fast): use KV cache for attention, run SSM prefill only
        // Option 3 (async): use KV + stale SSM, re-derive SSM in background
    } else {
        // Non-hybrid: partial hit is a regular prefix hit
        // Restore KV, prefill remaining tokens
    }
```

**Python reference:** mllm_batch_generator.py:1480-1545 -- discards KV if SSM companion missing for hybrid models.

---

### C. Cache Store -- Truncation to prompt_len-1 (WRONG -- currently stores prompt+generated)

**Current:** VMLXRuntimeActor.swift:583 stores tokens + accumulator.generatedTokenIds.

**Fix:** Store only cacheKeyTokens (stripped of gen_prompt_len), truncated to len-1:
```swift
// After generation completes:
let cacheKeyTokens = tokens[0 ..< tokens.count - genPromptLen]
let storeTokens = Array(cacheKeyTokens.dropLast(1))  // prompt_len - 1

// Truncate cache state to match
for c in cache {
    if let kvc = c as? VMLXKVCacheSimple {
        kvc.trim(genPromptLen + generatedTokenCount + 1)
        // Now cache has prompt_len - 1 tokens
    }
}
```

**Why prompt_len-1:** On cache hit, we re-feed the last token to get fresh logits for sampling the first generated token. This ensures correct model state.

**Python reference:** scheduler.py:2376-2432

---

### D. Hybrid Model Support -- _fix_hybrid_cache (NOT IMPLEMENTED)

**What:** When cache hit returns KV-only layers (from paged/memory cache), expand to full layer count by inserting fresh SSM cache objects at SSM positions.

**Implementation:**
```
File: NEW -- Models/Utilities/HybridCacheUtils.swift

func fixHybridCache(
    cache: [any VMLXKVCache],       // KV-only from cache hit
    model: any VMLXNativeModel,     // for template
    kvPositions: [Int]?,            // which layers are attention
    numModelLayers: Int?
) -> [any VMLXKVCache]
```

**Logic:**
1. Get template from model.newCache() -- has correct types per layer
2. If cache.count == numModelLayers: check type mismatches, swap wrong types
3. If cache.count < numModelLayers: expand: place cached KV at kvPositions, fresh SSM elsewhere

**Python reference:** mllm_batch_generator.py:201-288

**Critical:** Only needed for hybrid models (Qwen3.5, Nemotron). Pure transformer models skip this.

---

### E. SSM Companion Cache -- Store and Fetch (STORE EXISTS, FETCH MISSING)

**Current state:**
- SSMStateCache.swift: built, has store() and fetch() methods
- CacheCoordinator stores SSM state at line 199-208
- CacheCoordinator NEVER calls fetch() for SSM companion

**Fix in CacheCoordinator.fetch():**
```swift
// After KV cache hit for hybrid model:
if isHybrid, let cache = kvCache {
    let ssmStates = ssmStateCache?.fetch(tokenHash: hash)
    if let ssmStates {
        return .hit(cache: mergedCache, remaining: remaining, detail: detail)
    } else {
        return .partialHit(attentionCache: cache, remaining: remaining, detail: detail)
    }
}
```

**SSM state extraction after prefill:**
```swift
// In VMLXRuntimeActor, after prefill completes:
if container.isHybrid {
    var ssmLayers: [MLXArray] = []
    for (i, c) in cache.enumerated() {
        if c is VMLXMambaCache {
            ssmLayers.append(contentsOf: (c as! VMLXMambaCache).state)
        }
    }
    if !ssmLayers.isEmpty {
        scheduler.cache.ssmStateCache?.store(
            tokenHash: cacheKeyHash,
            ssmStates: ssmLayers,
            boundary: cacheKeyTokens.count
        )
    }
}
```

**Python reference:** mllm_batch_generator.py:291-346 (HybridSSMStateCache), scheduler.py:2277-2341 (store after generation)

---

### F. SSM Re-Derivation (BUILT, NOT WIRED)

**What:** When cache has KV but no SSM companion, option to re-derive SSM state by running a forward pass on cached tokens through SSM layers only.

**Current:** SSMReDeriver.swift exists as an actor with requestReDerive(). Never called.

**Wiring needed:**
1. In VMLXRuntimeActor.loadModel: create SSMReDeriver, call setModel()
2. In generation loop on .partialHit: call rederiver.requestReDerive()
3. Decide sync vs async:
   - Sync (simple): block until SSM state re-derived, then proceed
   - Async (fast): start generation with KV-only, inject SSM state when ready
   - Recommendation: Start with sync. Async is complex and can produce wrong output.

**Thinking model consideration:** For reasoning models (Qwen3.5 with think tags), SSM state carries reasoning context. Using stale/missing SSM state during thinking will produce garbage reasoning. Sync re-derivation is mandatory for thinking models.

---

### G. Paged Block Cache Integration (BUILT, NOT WIRED)

**Current:** PagedCacheManager exists, initialized in CacheCoordinator, but fetch()/store() never called.

**What needs to happen:**

1. Fetch path in CacheCoordinator:
   - Compute block chain hashes for input tokens
   - Search PagedCacheManager for matching block sequence
   - Reconstruct KV cache by concatenating block tensor slices
   - Return remaining uncached tokens

2. Store path in CacheCoordinator:
   - Split extracted cache into block-sized chunks (64 tokens each)
   - Compute chain hashes (parent_hash + token_content)
   - Dedup: if block hash exists, increment ref count (reuse)
   - New blocks: allocate, store tensor slices
   - Handle cumulative SSM state in last block (Bug 2 fix from Python)

3. Block structure:
   - Each block = 64 tokens of KV cache per layer
   - Chain hash = SHA256(parent_hash + tokens)
   - Ref counting for shared prefix blocks
   - LRU eviction when max blocks exceeded

**Python reference:** paged_cache.py (BlockAwarePrefixCache), prefix_cache.py:590-900+

**Nuance -- cumulative SSM in last block:**
- Non-last blocks tag SSM layers as "skip"
- Last block tags SSM layers as "cumulative" (stores full state)
- On block reuse for last position: check if block has cumulative data
- If not: allocate new block instead of reusing (Bug 2 from hybrid-ssm-cache-bugs.md)

---

### H. TurboQuant Integration (BUILT, NOT WIRED)

**What:** 3-bit KV cache compression. 5x memory reduction. Enables caching 5x more conversations.

**When triggered:** After prefill, before cache storage. Configurable via enableTurboQuant.

**Integration points:**

1. Cache construction (VMLXModelContainer.newCache()):
   - If TurboQuant enabled: wrap each KVCacheSimple in TurboQuantKVCache
   - TurboQuantKVCache starts in "fill" phase (stores float16)

2. After prefill (VMLXRuntimeActor):
   - Call turboQuantCache.compress() on each layer
   - This encodes float to 3-bit, creates decoded buffer for fast read
   - Subsequent decode steps read from decoded buffer (no re-decode)

3. Cache store:
   - Store compressed data (EncodedKeys/EncodedValues) instead of float
   - 5x smaller in memory cache, 5x smaller on disk

4. Cache fetch:
   - Load compressed data
   - Decode once into float buffer
   - Use decoded buffer for attention

**Sink tokens:** First 4 tokens (BOS/system) stay at full precision. Never compressed.

**Post-compress speed fix:** Decoded buffer cached after compress(). Each decode step reads concat(decoded_buffer, float_window) -- both float arrays, standard SDPA at full Metal speed. O(1) per step, no re-decode.

**Python reference:** scheduler.py:2185-2192, jang/docs/plans/2026-03-24-turboquant-integration.md

---

### I. Disk Cache L2 (BUILT, PARTIALLY WIRED)

**Current:** DiskCache.swift exists with SQLite index + safetensors storage. Wired in CacheCoordinator but disabled by default.

**Gaps:**
1. Background write doesn't verify file exists before declaring hit
2. No L2 to L1 promotion (load from disk into memory cache)
3. Must pre-materialize tensors on calling thread before background write (Metal thread safety -- Bug 1 from Python)

**Fix:**
1. In fetchCache(): verify safetensors file exists at path before returning
2. After disk fetch hit: also store into memory cache (L2 to L1 promotion)
3. In storeCache(): materialize all cache arrays before dispatching to background thread

**Python reference:** disk_cache.py, Bug 1 in hybrid-ssm-cache-bugs.md

---

## Implementation Order

Each item builds on the previous. Do not skip ahead.

### Phase 1: Cache Key and Store Fix (foundation -- everything depends on this) -- DONE 2026-03-30
1. [x] Implement gen_prompt_len stripping in cache key computation
2. [x] Fix cache store: truncate to prompt_len-1, use stripped key
3. [x] Fix cache fetch: handle partial hits properly (not as miss)
4. [x] Fix full hit input to include gen_prompt_len suffix tokens
5. [ ] Test: verify turn-2 cache hit with simple Llama/Qwen model

### Phase 2: Hybrid Model Support (required for JANG SSM models) -- DONE 2026-03-30
5. [x] Pass SSMCheckpoint through CacheFetchResult.hit (was checked but not returned)
6. [x] Merge SSM checkpoint into VMLXMambaCache objects on cache hit
7. [x] SSM companion stored via CacheCoordinator.store (boundary matches KV store key)
8. [x] BUG FIX: removed duplicate direct SSM store that had wrong boundary (N vs N-1)
9. [x] _fix_hybrid_cache: DEFERRED to Phase 3 (only needed for paged block cache KV-only storage)
10. [x] SSMReDeriver wiring: DEFERRED to Phase 6 (needs protocol additions, current safe fallback works)
11. [x] Updated Scheduler.swift pattern match for new CacheFetchResult.hit(4 params)
12. [ ] Test: verify Qwen3.5 hybrid model cache hit with SSM companion

### Phase 3: Paged Block Cache (prefix sharing, memory efficiency) -- DONE 2026-03-30
10. [x] CacheBlock.cacheData changed to [LayerCacheEntry?]? for SSM + future TurboQuant support
11. [x] Added import MLX to CacheCoordinator.swift
12. [x] Wire PagedCacheManager fetch into CacheCoordinator (_fetchFromPagedCache)
13. [x] Wire PagedCacheManager store into CacheCoordinator (_storeToPagedCache)
14. [x] Block reconstruction: concatenate KV slices across blocks per layer (_reconstructFromBlocks)
15. [x] Handle cumulative SSM in last block: .ssm in last block, nil in non-last (Bug 2 prevention)
16. [x] Block dedup: check chain hash, increment ref count on match
17. [x] Bug 2 guard: _blockLacksCumulativeSSM — skip reuse if last block needs SSM but has none
18. [x] Paged fetch runs BEFORE memory cache in cascade (priority order: paged -> memory -> prefix -> disk)
19. [x] Prefix cache disabled when paged cache is active (store: pagedCache == nil check)
20. [ ] Test: verify block sharing across conversations with same system prompt

### Phase 4: Disk Cache L2 (persistence across sessions)
15. [ ] Fix pre-materialize on calling thread (Metal thread safety)
16. [ ] Add file existence check in fetch
17. [ ] Implement L2 to L1 promotion
18. [ ] Test: verify cold start loads from disk, second turn hits L1

### Phase 5: TurboQuant (5x memory compression)
19. [ ] Wire TurboQuantKVCache into cache construction
20. [ ] Call compress() after prefill
21. [ ] Store compressed data in memory/disk cache
22. [ ] Implement decoded buffer cache (post-compress speed fix)
23. [ ] Test: verify 5x memory reduction, no quality degradation

### Phase 6: SSM Async Re-Derivation (advanced -- thinking models)
24. [ ] Wire SSMReDeriver actor into generation loop
25. [ ] Implement async path: generate with KV-only while SSM re-derives
26. [ ] Handle thinking model constraint (must be sync for reasoning)
27. [ ] Test: verify reasoning quality with re-derived SSM state

---

## Known Bugs to Avoid (from Python VMLX audit)

| Bug | What Went Wrong | How to Avoid |
|-----|----------------|--------------|
| Metal crash (Bug 1) | Background thread called MLX GPU ops | Pre-materialize all tensors on calling thread before background write |
| Layer count mismatch (Bug 2) | Block reuse didn't update cumulative SSM state | Check if last block needs cumulative data; if reused block lacks it, allocate new |
| Broadcast shape error (Bug 3) | Reconstructed cache offset vs attention mask mismatch | Ensure cache.offset is set correctly after reconstruction |
| gen_prompt_len not stripped (S25-7) | Cache key included assistant tokens | Always strip gen_prompt_len from cache key |
| ndim crash (S25-1) | dict.keys() matched as cache .keys attribute | Type-check before accessing .ndim |
| MLA head count (prefix_cache) | Assumed num_attention_heads, but MLA uses H=1 | Check kv_lora_rank; if > 0, n_kv_heads = 1 |

---

## Files to Create/Modify

### New Files
- Models/Utilities/HybridCacheUtils.swift -- _fix_hybrid_cache, hybrid detection
- Cache/BlockHashComputer.swift -- chain hash computation for paged cache

### Major Modifications
- Integration/VMLXRuntimeActor.swift -- generation loop restructured around cache
- Cache/CacheCoordinator.swift -- full fetch/store cascade with all layers
- Core/ModelContainer.swift -- gen_prompt_len computation

### Minor Modifications
- Cache/DiskCache.swift -- pre-materialize fix, file verification, L2-to-L1 promotion
- Cache/SSMStateCache.swift -- wire fetch into coordinator
- Cache/PagedCacheManager.swift -- wire into coordinator fetch/store
- Quantization/TurboQuantKVCache.swift -- wire into cache construction
- Scheduler/SchedulerConfig.swift -- expose all cache config knobs

---

## Audit Checklist — Component Status (updated 2026-03-30)

### Generation Loop (VMLXRuntimeActor.swift)
| Item | Status | Line | Notes |
|------|--------|------|-------|
| gen_prompt_len computation | DONE | 376 | container.computeGenPromptLen() |
| cacheKeyTokens stripping | DONE | 377-382 | drops genPromptLen from end |
| Cache fetch uses cacheKeyTokens | DONE | 407 | not raw tokens |
| .hit: restore KV state | DONE | 412-425 | iterates cachedHybrid.layers |
| .hit: inject SSM from checkpoint | DONE | 428-438 | iterates cache for VMLXMambaCache |
| .hit: genSuffix computation | DONE | 441-442 | tokens.suffix(genPromptLen) |
| .hit: remaining + genSuffix input | DONE | 444-457 | feeds remaining + suffix |
| .partialHit hybrid: discard KV | DONE | 461-467 | full prefill, SSM path-dependent |
| .partialHit non-hybrid: use KV | DONE | 469-497 | restore + prefill remaining |
| .miss: full prefill | DONE | 499-501 | all tokens |
| Chunked prefill | DONE | 510-528 | prefillStepSize chunks |
| SSM companion extraction after prefill | DONE | via CacheCoordinator.store | boundary matches KV store key (storeTokens.count) |
| Double-buffered asyncEval | DONE | 569-581 | overlaps CPU/GPU |
| Cache store: use storeTokens | DONE | 677-720 | cacheKeyTokens.dropLast(1) |
| Cache store: trim KV to targetOffset | DONE | 692-699 | kvc.trim(offset - target) |
| Cache store: SSM stored as-is | DONE | 703-704 | cumulative, not truncated |
| Cache store: materialized() before store | DONE | 715 | forces lazy eval |
| Cancellation clears cache | DONE | 724-725 | prevents stale partial |

### CacheCoordinator (CacheCoordinator.swift)
| Item | Status | Line | Notes |
|------|--------|------|-------|
| CacheFetchResult.hit has ssmCheckpoint | DONE | 48 | Optional SSMCheckpoint parameter |
| fetch cascade: paged -> memory -> prefix -> disk | DONE | 132-190 | 4-layer cascade |
| _fetchFromPagedCache: chain hash walk | DONE | 197-228 | walks tokens in blockSize chunks |
| _reconstructFromBlocks: concat KV slices | DONE | 234-290 | per-layer concat + SSM from last block |
| _resolveHybridFetch returns checkpoint | DONE | 395 | passes SSMCheckpoint in .hit |
| store cascade: paged + memory + prefix + disk + SSM | DONE | 295-340 | all layers stored to all tiers |
| _storeToPagedCache: split + chain hash + dedup | DONE | 345-400 | block allocation + Bug 2 guard |
| _blockLacksCumulativeSSM: Bug 2 prevention | DONE | 405-412 | checks for .ssm entries in block |
| SSMStateCache.store called with ssmLayers | DONE | 316-325 | extracted from HybridCache |
| Prefix cache disabled when paged active | DONE | 307 | pagedCache == nil check |
| L2 disk to L1 promotion | NOT DONE | -- | Phase 4 |

### SSMStateCache (SSMStateCache.swift)
| Item | Status | Line | Notes |
|------|--------|------|-------|
| store() by hash + boundary | DONE | 51-55 | convenience method |
| fetch() with deep copy | DONE | 59-85 | array * 1 forces copy |
| Empty ssmStates = MISS | DONE | 71 | bug fix from VMLX ba07392 |
| LRU eviction | DONE | 44-46 | oldest first |
| Token hashing (SHA-256) | DONE | 112-117 | JSON of token array |

### MemoryCache (MemoryCache.swift)
| Item | Status | Notes |
|------|--------|-------|
| Exact match | DONE | O(1) lookup |
| Forward prefix match | DONE | cached is prefix of request |
| Reverse prefix match | DONE | request is prefix of cached, truncates |
| Memory pressure monitoring | DONE | checks every 60s |
| LRU eviction | DONE | oldest first when over limit |
| TTL expiration | DONE | configurable, default off |

### Speed Optimizations (StandardModel.swift, Qwen35Model.swift, etc.)
| Item | Status | File | Notes |
|------|--------|------|-------|
| Compiled SwiGLU | DONE | StandardModel.swift:209 | compile(shapeless: true) |
| Compiled SwiGLU (Qwen35) | DONE | Qwen35Model.swift:229 | same compiledSwiGLU |
| Compiled computeGatedDeltaG | DONE | GatedDelta.swift:17 | fused exp+softplus+mul |
| Compiled gatedDeltaStepOps | DONE | GatedDelta.swift:29-77 | with/without mask |
| RMSNormGated uses compiledSwiGLU | DONE | Qwen35Model.swift:211 | fused silu*x |
| Float32 MoE gate routing | DONE | StandardModel.swift:264 | prevents overflow |
| e_score_correction_bias | DONE | StandardModel.swift:269 | selection not weighting |
| argPartition sign convention | DONE | StandardModel.swift:272 | -scores, kth=k-1 |
| Score normalization epsilon | DONE | StandardModel.swift:275 | + 1e-20 |
| Symbolic .causal mask for prefill | DONE | KVCache.swift:239 | no array materialization |
| bfloat16 for large MoE | DONE | ModelRegistry.swift:130 | >= 256 experts |
| Scalar Q/K scaling (Qwen35) | DONE | Qwen35Model.swift:334 | Float * MLXArray |
| Memory.clearCache every 256 | DONE | VMLXRuntimeActor.swift | matches Python |

### NOT YET WIRED (Phase 3-6)
| Component | Phase | Status | Notes |
|-----------|-------|--------|-------|
| CacheBlock.cacheData type | 3 | DONE | Changed from [(keys,values)]? to [LayerCacheEntry?]? for SSM+TQ support |
| PagedCacheManager fetch | 3 | DONE | _fetchFromPagedCache: chain hash walk + block matching |
| PagedCacheManager store | 3 | DONE | _storeToPagedCache: split into blocks, chain hash, dedup |
| Block chain hash computation | 3 | DONE | CacheBlock.computeBlockHash with SHA-256 + parent chain |
| Block reconstruction (concat KV slices) | 3 | DONE | _reconstructFromBlocks: concat per-layer across blocks |
| Cumulative SSM in last block | 3 | DONE | .ssm in last block, nil in non-last + Bug 2 guard |
| _fix_hybrid_cache expansion | 3 | DEFERRED | Not needed when paged store writes ALL layers (KV + SSM) |
| DiskCache pre-materialize | 4 | NOT DONE | Metal thread safety (Bug 1) |
| DiskCache file verification | 4 | NOT DONE | check file exists before hit |
| DiskCache L2-to-L1 promotion | 4 | NOT DONE | load from disk into memory |
| TurboQuantKVCache wrapping | 5 | NOT WIRED | built, never instantiated |
| TurboQuant compress() after prefill | 5 | NOT DONE | encoder exists, never called |
| TurboQuant decoded buffer cache | 5 | NOT DONE | post-compress speed fix |
| SSMReDeriver wiring | 6 | NOT WIRED | actor exists, requestReDerive never called |
| SSMReDeriver state extraction | 6 | NOT DONE | creates empty checkpoints |
| Async SSM re-derivation | 6 | NOT DONE | thinking model constraint |

---

## Change Log — Files Modified (2026-03-30 session)

### Speed Optimizations (before cache stack work)
| File | Changes |
|------|---------|
| `Models/StandardModel.swift` | compiledSwiGLU, float32 MoE gate routing, e_score_correction_bias fix, argPartition convention, epsilon guard |
| `Models/Qwen35Model.swift` | compiledSwiGLU for MLP + RMSNormGated, scalar Q/K scaling (Float * MLXArray) |
| `Models/Utilities/GatedDelta.swift` | compiled computeGatedDeltaG, compiled gatedDeltaStepOps (with/without mask) |
| `Models/Utilities/KVCache.swift` | symbolic .causal mask for fresh prefill |
| `Models/WeightLoader.swift` | no changes (was read-only audit) |
| `Models/ModelRegistry.swift` | bfloat16 conversion for large MoE (>= 256 experts) |

### Cache Stack Integration (Phase 1-3)
| File | Changes |
|------|---------|
| `Integration/VMLXRuntimeActor.swift` | gen_prompt_len computation + cacheKeyTokens stripping, cache fetch uses cacheKeyTokens, .hit with SSM checkpoint injection + genSuffix computation, .partialHit handling (hybrid discard vs non-hybrid use), .miss full prefill, cache store truncated to prompt_len-1 with stripped key, KV trim to targetOffset, double-buffered asyncEval, chunked prefill, Memory.clearCache every 256 |
| `Cache/CacheCoordinator.swift` | import MLX added, CacheFetchResult.hit gains ssmCheckpoint param, fetch cascade: paged -> memory -> prefix -> disk, _fetchFromPagedCache (chain hash walk), _reconstructFromBlocks (per-layer concat), store cascade: paged + memory + prefix + disk + SSM, _storeToPagedCache (split + chain hash + dedup + Bug 2 guard), _blockLacksCumulativeSSM, _resolveHybridFetch returns SSMCheckpoint, prefix cache disabled when paged active |
| `Cache/CacheBlock.swift` | cacheData type changed from [(keys: MLXArray, values: MLXArray)]? to [LayerCacheEntry?]? for SSM + future TurboQuant |
| `Scheduler/Scheduler.swift` | pattern match updated for CacheFetchResult.hit(4 params) |

### Documentation
| File | Contents |
|------|---------|
| `docs/plans/2026-03-30-cache-stack-architecture.md` | Full architecture, 9 components (A-I), 6 phases, known bugs, audit checklist |
| `docs/plans/2026-03-30-phase1-cache-key-store-fix.md` | Phase 1 detailed implementation plan with exact code |

### Bugs Found and Fixed During Development
| Bug | Found In | Fix |
|-----|----------|-----|
| SSM boundary mismatch | Phase 2 cross-check | Removed duplicate direct SSM store (boundary N vs N-1). CacheCoordinator.store handles it correctly with boundary = storeTokens.count |
| CacheFetchResult pattern match | Phase 2 build | Updated Scheduler.swift and VMLXRuntimeActor.swift to match new 4-param .hit enum |
| MLX.concatenated doesn't exist | Phase 3 build | Changed to concatenated() free function |
| Missing import MLX | Phase 3 build | Added to CacheCoordinator.swift |

### Function Call Traces (verified correct)
| Call Site | Function | Signature | Verified |
|-----------|----------|-----------|----------|
| VMLXRuntimeActor:376 | container.computeGenPromptLen(messages:) | (messages: [VMLXChatMessage]) -> Int | YES |
| VMLXRuntimeActor:407 | scheduler.cache.fetch(tokens:) | (tokens: [Int], tokenHash: String?) -> CacheFetchResult | YES |
| VMLXRuntimeActor:716 | scheduler.cache.store(tokens:cache:) | (tokens: [Int], cache: HybridCache, tokenHash: String?) | YES |
| CacheCoordinator:284 | ssmCache.fetch(tokenHash:boundary:) | (tokenHash: String, boundary: Int) -> SSMCheckpoint? | YES |
| CacheCoordinator:320 | ssmCache.store(ssmStates:tokens:boundary:) | (ssmStates: [SSMStateLayer], tokens: [Int], boundary: Int) | YES |
| CacheCoordinator:137 | _fetchFromPagedCache(tokens:pagedCache:) | private, returns (HybridCache, [Int])? | YES |
| CacheCoordinator:345 | _storeToPagedCache(tokens:cache:pagedCache:) | private, splits into blocks | YES |
| VMLXRuntimeActor:452 | kvc.trim(_:) | (Int) -> Int, decrements offset | YES |
| VMLXRuntimeActor:435 | mambaCache.state = checkpoint.ssmStates[ssmIdx].state | [MLXArray] setter on VMLXArraysCache | YES |
| SSMStateLayer init | SSMStateLayer(state:) | (state: [MLXArray], isCumulative: Bool = true) | YES |
| HybridCache init | HybridCache(layers:) | (layers: [LayerCacheEntry]) | YES |
| KVCacheLayer init | KVCacheLayer(keys:values:offset:) | (keys: MLXArray, values: MLXArray, offset: Int) | YES |
| CacheBlock.computeBlockHash | (parentHash:tokenIds:) | (BlockHash?, [Int]) -> BlockHash | YES |
| pagedCache.findCachedBlock(hash:) | (hash: BlockHash) -> CacheBlock? | YES |
| pagedCache.allocateBlock() | () -> CacheBlock? | YES |
| pagedCache.markCached(block:hash:) | (block: CacheBlock, hash: BlockHash) | YES |
