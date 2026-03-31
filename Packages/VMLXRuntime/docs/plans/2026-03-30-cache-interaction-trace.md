# Cache Interaction Trace: TurboQuant × SSM × All Cache Layers

**Date:** 2026-03-30
**Purpose:** Trace every code path where TurboQuant, SSM re-derivation, and cache layers interact.
**Verified against:** Current code in `feature/vmlx` branch.

---

## Function Call Map

```
VMLXRuntimeActor.generateStream()
├── container.applyChatTemplate() → tokens
├── container.computeGenPromptLen() → genPromptLen
├── cacheKeyTokens = tokens.dropLast(genPromptLen)
│
├── FETCH: scheduler.cache.fetch(tokens: cacheKeyTokens)
│   └── CacheCoordinator.fetch()
│       ├── Layer 0: PagedCacheManager
│       │   ├── _fetchFromPagedCache() → walk chain hashes
│       │   └── _reconstructFromBlocks() → concat per-layer KV slices
│       │       ├── .attention(kv) → append keys/values
│       │       ├── .compressedAttention(ek,ev) → decode → append (safety fallback only)
│       │       └── .ssm(ssm) → last block only
│       ├── Layer 1: MemoryCache.fetch() → prefix match
│       ├── Layer 2: PrefixCache.fetch() → trie match
│       ├── Layer 3: DiskCache.fetchCache() → safetensors load
│       │   └── L2→L1 promotion: MemoryCache.store(cache)
│       │
│       └── _resolveHybridFetch() (if isHybrid)
│           ├── boundary = tokens.count - remaining.count
│           ├── SSMStateCache.fetch(tokenHash, boundary)
│           ├── SSM found → .hit(cache, remaining, detail, ssmCheckpoint)
│           └── SSM missing → .partialHit(attentionCache, remaining, detail)
│
├── HIT HANDLER (line 436-495):
│   ├── .hit + layerCount matches:
│   │   ├── Restore KV: .attention → VMLXKVCacheSimple.state = [keys, values]
│   │   ├── Restore TQ: .compressedAttention → decode → VMLXKVCacheSimple.state
│   │   ├── Restore SSM: .ssm → VMLXMambaCache.state
│   │   ├── Inject SSM checkpoint (overwrites, belt-and-suspenders)
│   │   ├── Compute genSuffix = tokens.suffix(genPromptLen)
│   │   ├── If remaining empty: trim cache by 1, re-feed last token + genSuffix
│   │   └── If remaining non-empty: prefill remaining + genSuffix
│   │
│   ├── .partialHit + hybrid:
│   │   └── Discard KV, full prefill (SSM re-derived as side effect)
│   │       → CacheCoordinator.store() saves SSM companion → next turn gets .hit
│   │
│   ├── .partialHit + non-hybrid:
│   │   ├── Restore KV: .attention or .compressedAttention → decode → load
│   │   └── Compute remaining + genSuffix → prefill
│   │
│   └── .miss or layer count mismatch:
│       └── Full prefill all tokens
│
├── PREFILL (line 559-588):
│   ├── Chunked: process in prefillStepSize chunks with eval + clearCache
│   └── Last chunk → logits → first token (greedy or sampled)
│
├── DECODE LOOP (line 594-702):
│   ├── Double-buffered: build graph for N+1 while GPU evaluates N
│   ├── EOS/stop detection
│   ├── Thinking budget enforcement
│   └── StreamAccumulator for tool/reasoning parsing
│
└── STORE (line 704-761):
    ├── storeTokens = cacheKeyTokens.dropLast(1)
    ├── Trim KV cache to targetOffset = storeTokens.count
    ├── Build LayerCacheEntry array:
    │   ├── VMLXMambaCache → .ssm(SSMStateLayer)
    │   ├── VMLXKVCacheSimple + TQ enabled → TurboQuantEncoder.encode → .compressedAttention
    │   └── VMLXKVCacheSimple + TQ disabled → .attention(KVCacheLayer)
    │
    └── CacheCoordinator.store(tokens: storeTokens, cache: hybridCache)
        ├── _storeToPagedCache():
        │   ├── .attention → slice to block range → store
        │   ├── .compressedAttention → DECOMPRESS → slice → store as .attention
        │   │   (paged cache needs positional slicing; compressed indices are per-vector)
        │   └── .ssm → last block only, nil in non-last
        ├── MemoryCache.store() → stores as-is (compressed or float)
        ├── PrefixCache.store() → (only if memory+paged both off)
        ├── DiskCache.storeCache() → safetensors with type metadata
        │   ├── "attention" → keys/values tensors
        │   ├── "compressed_attention" → TQ index/norm/qjl tensors
        │   └── "ssm" → state arrays
        └── SSMStateCache.store() → ssmLayers extracted, boundary = tokens.count
```

---

## Interaction Matrix: Model Type × Cache Layer × TurboQuant

### Standard LLM (Llama, Qwen2, Qwen3) — No SSM

| Operation | TQ Off | TQ On |
|-----------|--------|-------|
| Cache store entries | `.attention` per layer | `.compressedAttention` per layer |
| Paged cache store | slice float KV per block | decompress → slice float KV per block |
| Memory cache store | float HybridCache | compressed HybridCache (5x smaller) |
| Disk cache store | attention type (safetensors) | compressed_attention type (5x smaller) |
| SSM companion | N/A (not hybrid) | N/A |
| Cache hit restore | float → VMLXKVCacheSimple | decode → VMLXKVCacheSimple |
| Layer count check | `layerCount == cache.count` ✓ | `layerCount == cache.count` ✓ |

### Hybrid SSM (Qwen3.5-A3B, Nemotron-H, Jamba) — Mixed Layers

| Operation | TQ Off | TQ On |
|-----------|--------|-------|
| Cache store entries | `.attention` + `.ssm` | `.compressedAttention` + `.ssm` |
| Paged cache store | attention sliced, SSM in last block | decompress+slice attention, SSM in last block |
| Memory cache store | float + SSM | compressed + SSM |
| Disk cache store | attention + ssm types | compressed_attention + ssm types |
| SSM companion store | `cache.ssmLayers` → SSMStateCache | same (`.ssm` entries unaffected by TQ) |
| SSM companion fetch | `_resolveHybridFetch` → boundary match | same |
| Cache hit + SSM found | `.hit(ssmCheckpoint)` → inject both | same + decode compressed |
| Cache hit + SSM missing | `.partialHit` → full prefill | same (SSM re-derived on prefill) |
| SSM self-healing | store() saves SSM companion → next turn .hit | same |

### MoE (MiniMax M2.5, Qwen3 MoE) — No SSM, Large Expert Count

| Operation | TQ Off | TQ On |
|-----------|--------|-------|
| Behavior | Same as Standard LLM | Same as Standard LLM |
| Note | Gate weights are float32 (bfloat16 model) | TQ compresses attention KV, not gate |
| `keyBits(forLayer:)` | All layers return bits | `.expert` layers return nil → skip TQ |

### VL (Qwen3.5-VL) — Vision + Text

| Operation | TQ Off | TQ On |
|-----------|--------|-------|
| Behavior | Same as underlying model type | Same |
| Note | Vision embeddings are pre-attention, not in KV cache | TQ compresses text KV only |

---

## SSM Companion Boundary Verification

**Critical invariant:** SSM companion store boundary must match fetch boundary.

```
STORE PATH:
  storeTokens = cacheKeyTokens.dropLast(1)    // length = N-1
  CacheCoordinator.store(tokens: storeTokens)
  └── SSMStateCache.store(boundary: storeTokens.count)  // boundary = N-1

FETCH PATH:
  CacheCoordinator.fetch(tokens: cacheKeyTokens)        // length = N
  ├── MemoryCache.fetch(tokens: cacheKeyTokens)
  │   └── prefix match on storeTokens → remaining = [last token]
  │       remaining.count = 1
  └── _resolveHybridFetch:
      boundary = N - 1                                   // N - remaining.count = N - 1
      SSMStateCache.fetch(boundary: N-1)                 // ✓ MATCHES
```

**Example:**
- cacheKeyTokens = [10, 20, 30, 40, 50] → N = 5
- storeTokens = [10, 20, 30, 40] → stored with boundary = 4
- Fetch: prefix matches [10, 20, 30, 40], remaining = [50]
- boundary = 5 - 1 = 4 → matches store boundary ✓

---

## TurboQuant Compression/Decompression Points

```
COMPRESS (store-time, VMLXRuntimeActor line 744-752):
  Input:  VMLXKVCacheSimple.state = [keys_float16, values_float16]
  Output: .compressedAttention(EncodedKeys, EncodedValues, offset)
  Guard:  scheduler.config.enableTurboQuant
          && container.turboQuantConfig != nil
          && keyBits(forLayer:) != nil (not SSM layer)

DECOMPRESS (fetch-time, multiple locations):
  1. VMLXRuntimeActor .hit handler (line 447-453):
     .compressedAttention → TurboQuantEncoder.decodeKeys/Values → VMLXKVCacheSimple.state
  2. VMLXRuntimeActor .partialHit non-hybrid (line 521-526):
     Same as above
  3. CacheCoordinator._storeToPagedCache (line 403-413):
     .compressedAttention → decode → slice → store as .attention
  4. CacheCoordinator._reconstructFromBlocks (line 264-270):
     Safety fallback: .compressedAttention → decode → append slices
  5. HybridTransformerModel.loadCache (line 323-329):
     .compressedAttention → decode → KVCache
  6. TransformerModel.loadCache (line 437-441):
     .compressedAttention → decode → KVCache
```

---

## DiskCache Serialization Format

```
attention type:
  metadata: __layer_{i}_type__ = "attention"
            __layer_{i}_offset__ = "{offset}"
  arrays:   layer_{i}_keys, layer_{i}_values

compressed_attention type:
  metadata: __layer_{i}_type__ = "compressed_attention"
            __layer_{i}_offset__ = "{offset}"
            __layer_{i}_index_bits__ = "{bits}"
            __layer_{i}_shape__ = "{b},{h},{t},{d}"
            __layer_{i}_value_shape__ = "{b},{h},{t},{d}"
  arrays:   layer_{i}_ek_indices (uint32, packed codebook indices)
            layer_{i}_ek_qjl (uint32, QJL sign bits)
            layer_{i}_ek_residual (float16, residual norms)
            layer_{i}_ek_norms (float16, vector norms)
            layer_{i}_ev_indices (uint32, value codebook indices)
            layer_{i}_ev_norms (float16, value vector norms)

ssm type:
  metadata: __layer_{i}_type__ = "ssm"
            __layer_{i}_state_count__ = "{n}"
  arrays:   layer_{i}_state_{0..n}
```

---

## Paged Cache Block Layout (Hybrid + TurboQuant)

```
Block 0 (non-last):
  layer 0: .attention(keys[0:64], values[0:64])    ← decompressed from TQ
  layer 1: nil                                      ← SSM position, skip
  layer 2: .attention(keys[0:64], values[0:64])    ← decompressed from TQ
  layer 3: nil                                      ← SSM position, skip

Block 1 (last):
  layer 0: .attention(keys[64:100], values[64:100]) ← decompressed from TQ
  layer 1: .ssm(cumulative_state)                    ← SSM in last block only
  layer 2: .attention(keys[64:100], values[64:100]) ← decompressed from TQ
  layer 3: .ssm(cumulative_state)                    ← SSM in last block only
```

**Key invariant:** Paged cache ALWAYS stores float `.attention` entries.
Compressed data is decompressed during `_storeToPagedCache` before slicing.
This is necessary because TQ codebook indices are per-vector, not positional.

---

## Settings Flow: UI → Engine

```
ConfigurationView (SwiftUI)
  ├── cacheMemoryPercent slider
  ├── enableDiskCache toggle
  ├── enableTurboQuant toggle
  ├── kvBits stepper
  └── kvGroupSize stepper
      ↓
ServerConfiguration (UserDefaults, Codable)
      ↓
VMLXServiceBridge.applyRuntimeConfig()
  ├── RuntimeConfig.snapshot() → kvBits, kvGroup, maxKV, prefillStep
  └── ServerConfiguration → enableDiskCache, enableTurboQuant, cacheMemoryPercent
      ↓
VMLXService.applyUserConfig()
      ↓
VMLXRuntimeActor.applyUserConfig()
  ├── scheduler.config.kvCacheQuantization = "q{bits}" or "none"
  ├── scheduler.config.kvCacheGroupSize = kvGroupSize
  ├── scheduler.config.maxNumBatchedTokens = maxKV
  ├── scheduler.config.prefillStepSize = prefillStep
  ├── scheduler.config.enableDiskCache = enableDiskCache
  ├── scheduler.config.diskCacheDir = auto or explicit
  ├── scheduler.config.enableTurboQuant = enableTurboQuant
  ├── scheduler.config.cacheMemoryPercent = cacheMemoryPercent
  ├── scheduler.config.usePagedCache = usePagedCache
  └── scheduler.rebuildCacheCoordinator()
      └── Creates new CacheCoordinator from updated config
          ├── PagedCacheManager (if usePagedCache)
          ├── MemoryCache (if useMemoryAwareCache, with cacheMemoryPercent)
          ├── PrefixCache (if enablePrefixCache && !usePagedCache)
          ├── DiskCache (if enableDiskCache && diskCacheDir set)
          └── SSMStateCache (always, maxEntries from config)
```

**Note:** `kvCacheQuantization` ("q4"/"q8") is plumbed but NOT consumed in the forward pass.
KV cache quantization during attention requires changes to VMLXKVCacheSimple (future work).
This is separate from TurboQuant (post-prefill compression for storage).

---

## Bugs Found and Fixed

### Bug #1: Paged cache stored full compressed data in every block
**Found:** `_storeToPagedCache` stored `.compressedAttention` as-is in each block.
Compressed data is the FULL sequence — not a block-sized slice.
If 10 blocks, the full data was duplicated 10x.
On reconstruct, each block decoded the full sequence → 10x actual tokens.

**Fix:** Decompress `.compressedAttention` to float first, then slice per block.
Paged cache now always stores `.attention` (float). Memory/disk cache keep compressed.

### Bug #4: Disk cache NEVER hits due to hash mismatch (found during audit)
**Found:** `CacheCoordinator.store()` stores disk cache with `storeTokens` (N-1 tokens).
`CacheCoordinator.fetch()` queries disk cache with `cacheKeyTokens` (N tokens).
DiskCache uses exact hash matching → `hash(N-1 tokens) ≠ hash(N tokens)` → always misses.

Memory cache and paged cache don't have this problem because they do prefix matching.
SSM companion doesn't have this problem because it hashes only first `boundary` tokens.

**Fix:** CacheCoordinator.fetch() now tries disk cache with both `tokens` (exact) and
`tokens.dropLast(1)` (standard store path). L2→L1 promotion stores with truncated key
so memory cache prefix matching works correctly.

### Bug #5: TurboQuant seed not passed to decode calls (found during audit)
**Found:** All 8 `decodeKeys`/`decodeValues` call sites used default seed (42).
But `encodeKeys`/`encodeValues` used `tq.seed` from TurboQuantConfig.
If `tq.seed != 42`, the codebook won't match → decoded output is garbage.
Currently dormant (default seed IS 42) but would break silently on config change.

**Fix:** Added `seed` field to `EncodedKeys` and `EncodedValues` structs.
Encoder stores the seed. All 8 decoder call sites now use `ek.seed` / `ev.seed`.
DiskCache and TQDiskStore serialize/deserialize the seed in metadata.

### Previous bugs (from earlier phases):
- SSM boundary mismatch (direct SSM store used boundary=N, CacheCoordinator uses N-1) — fixed
- MLX.concatenated() → free function concatenated() — fixed
- Thread-unsafe refCount++ in block dedup → forkBlock() under lock — fixed
- _blockLacksCumulativeSSM optional unwrap — fixed

---

## Audit Checklist (completed 2026-03-30)

| File | Status | Notes |
|------|--------|-------|
| `LayerCache.swift` | ✅ | `.compressedAttention` case added, all properties handle it |
| `HybridCache.swift` | ✅ | `materialized()`, `canTruncate`, `isAttention` updated |
| `CacheBlock.swift` | ✅ | Uses `[LayerCacheEntry?]?`, compatible |
| `CacheCoordinator.swift` | ✅ | Bugs #1, #4 fixed. All switch statements handle `.compressedAttention` |
| `PagedCacheManager.swift` | ✅ | Lock-protected, LRU eviction, fork COW correct |
| `MemoryCache.swift` | ✅ | Prefix matching works with TQ (forward match). `canTruncate` blocks reverse match for TQ — acceptable |
| `PrefixCache.swift` | ✅ | Same as MemoryCache. Only created when paged=false |
| `DiskCache.swift` | ✅ | Store/fetch compressed_attention with seed. Bug #4 fixed in coordinator |
| `SSMStateCache.swift` | ✅ | Empty states = MISS guard. Deep copy on fetch. Boundary math verified |
| `SSMReDeriver.swift` | ✅ | Wired in loadModel(). Safe fallback (full prefill) self-heals |
| `FreeBlockQueue.swift` | ✅ | O(1) linked list, sentinel nodes, count tracked |
| `BlockHashMap.swift` | ✅ | Simple dict, pop checks blockId |
| `BlockTable.swift` | ✅ | Simple struct |
| `BlockDiskStore.swift` | ✅ | Uses old KV tuple format (not LayerCacheEntry) — separate concern |
| `TQDiskStore.swift` | ✅ | Seed added to metadata. serialize/deserialize updated |
| `TurboQuantKVCache.swift` | ✅ | Decoded buffer cache, estimatedBytes fixed |
| `TurboQuantEncoder.swift` | ✅ | Seed stored in EncodedKeys/Values |
| `EncodedKeys.swift` | ✅ | Seed field added |
| `EncodedValues.swift` | ✅ | Seed field added |
| `VMLXRuntimeActor.swift` | ✅ | All decode calls use ek.seed/ev.seed. TQ store gated correctly |
| `TransformerModel.swift` | ✅ | loadCache handles .compressedAttention |
| `HybridTransformerModel.swift` | ✅ | loadCache handles .compressedAttention |
