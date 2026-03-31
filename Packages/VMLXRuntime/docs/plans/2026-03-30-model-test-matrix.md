# Model Test Matrix: Cache Integration Verification

**Date:** 2026-03-30
**Purpose:** Verify every cache path works correctly across all model types on disk.

---

## Available Models on Disk

### Testable Models (smallest first — for quick iteration)

| # | Model | Type | Arch | Layers | Size | Path |
|---|-------|------|------|--------|------|------|
| 1 | Qwen3.5-4B-JANG_4S | qwen3_5 | Hybrid SSM | 32 (24 SSM + 8 attn) | ~4B | ~/jang/models/ |
| 2 | Qwen3.5-9B-JANG_4S | qwen3_5 | Hybrid SSM | 32 | ~9B | ~/jang/models/ |
| 3 | MiniMax-M2.5-JANG_2L | minimax_m2 | MoE (256 exp) | 62 | ~41B | ~/jang/models/ |
| 4 | Qwen3.5-VL-9B-8bit | qwen3_5 | Hybrid SSM + VL | 32 | ~9B | ~/.mlxstudio/models/ |
| 5 | Mistral-Small-4-119B-JANG_2L | mistral3 | MLA | 36 | ~119B | ~/models/ |
| 6 | Nemotron-Cascade-2-30B-A3B-JANG_2L | nemotron_h | Hybrid SSM | 52 | ~30B | ~/.mlxstudio/models/ |

### Not Loadable (blocked in ModelRegistry)

| Model | Reason |
|-------|--------|
| Mistral-Small-4 (mistral3) | FP8 quantization unsupported |
| Mistral-Small-4 (mistral4 text_config) | FP8 quantization unsupported |
| Nemotron (nemotron_h) | Not in standardTransformerTypes, not qwen3_5 — falls to default |

**ISSUE: nemotron_h is not handled by the registry switch statement.** It falls to the default branch
which tries `standardTransformerTypes.contains("nemotron_h")` → false → goes to the else branch
which tries to load as StandardTransformerModel. This will fail because Nemotron-H has a
completely different architecture (Mamba + attention hybrid, not standard transformer).

---

## Model Type → Cache Behavior Matrix

### 1. Qwen3.5 Hybrid SSM (JANG) — Primary Test Target
**Model:** Qwen3.5-4B-JANG_4S
**Config:** model_type=qwen3_5, full_attention_interval=4, 32 layers
**Layer pattern:** [SSM, SSM, SSM, Attn, SSM, SSM, SSM, Attn, ...] (24 SSM + 8 attention)

| Cache Path | Expected Behavior |
|------------|-------------------|
| **newCache()** | 32 caches: 24 VMLXMambaCache + 8 VMLXKVCacheSimple |
| **Cache store** | HybridCache with 24 `.ssm` + 8 `.attention` (or `.compressedAttention` if TQ on) |
| **Cache store + TQ** | 24 `.ssm` + 8 `.compressedAttention` (SSM never compressed) |
| **Paged store** | Attention sliced per block, SSM in last block only, nil in non-last |
| **Paged store + TQ** | TQ decompressed → sliced → stored as `.attention` in blocks |
| **Memory store** | Full HybridCache stored as-is (compressed or float) |
| **Disk store** | attention/compressed_attention + ssm types in safetensors |
| **SSM companion store** | `cache.ssmLayers` = 24 SSMStateLayer entries → SSMStateCache |
| **Cache fetch (full hit)** | `.hit` with ssmCheckpoint → inject both KV + SSM state |
| **Cache fetch (SSM evicted)** | `.partialHit` → full prefill → SSM re-derived as side effect |
| **Cache fetch + TQ** | `.compressedAttention` decoded via TurboQuantEncoder.decodeKeys/Values |
| **gen_prompt_len strip** | Working — multi-turn cache key reuse |
| **Trim on full hit** | Only attention (VMLXKVCacheSimple) trimmed, SSM untouched |
| **isHybrid** | true → SSM companion stored, _resolveHybridFetch called |

### 2. MiniMax M2.5 MoE (JANG) — Large MoE Test
**Model:** MiniMax-M2.5-JANG_2L
**Config:** model_type=minimax_m2, 62 layers, 256 experts × 8 active
**Architecture:** Pure attention (no SSM), MoE routing, bfloat16 conversion

| Cache Path | Expected Behavior |
|------------|-------------------|
| **newCache()** | 62 VMLXKVCacheSimple (all attention) |
| **Cache store** | HybridCache with 62 `.attention` entries |
| **Cache store + TQ** | 62 `.compressedAttention` (all layers compressible) |
| **isHybrid** | false → no SSM companion, no _resolveHybridFetch |
| **bfloat16** | Model loaded with _convertToBFloat16 (≥256 experts) |
| **Cache hit restore** | Standard KV restoration, no SSM injection |

### 3. Qwen3.5-VL (Vision-Language Hybrid)
**Model:** Qwen3.5-VL-9B-8bit
**Config:** model_type=qwen3_5, has vision_config + preprocessor_config

| Cache Path | Expected Behavior |
|------------|-------------------|
| **newCache()** | Same as Qwen3.5 (hybrid SSM pattern) |
| **Vision embeddings** | Pre-attention, NOT in KV cache |
| **Cache key** | May not match across turns (VLM tokenizer image placeholder divergence) |
| **TurboQuant** | Same as text model — compresses text KV, vision not in cache |

### 4. Mistral Small 4 (MLA)
**Status:** BLOCKED — FP8 quantization rejected by ModelRegistry
**Future:** When FP8 support added, MLA needs n_kv_heads=1 handling

### 5. Nemotron-H (Hybrid SSM)
**Status:** BLOCKED — nemotron_h not in registry, falls to StandardTransformerModel (wrong)
**Future:** Needs dedicated Nemotron model class or registry entry

---

## Test Plan: Phase 8 — Functional Verification

### Prerequisites
- Build passes (SPM + Xcode) ✅
- Smallest testable model: Qwen3.5-4B-JANG_4S (~4B, fits in 16GB RAM)

### Test 1: Basic Load + Single Token Forward Pass
**Model:** Qwen3.5-4B-JANG_4S
**Goal:** Verify model loads, cache objects created correctly, forward pass produces logits

```
1. Load model from ~/jang/models/Qwen3.5-4B-JANG_4S
2. Verify container.isHybrid == true
3. Verify container.layerPattern has 24 SSM + 8 attention
4. Verify newCache() returns 32 caches (24 VMLXMambaCache + 8 VMLXKVCacheSimple)
5. Forward pass: single token [1] → logits shape [1, 1, vocab_size]
6. Verify no crash, logits are finite
```

### Test 2: Two-Turn Cache Hit (Non-TQ)
**Model:** Qwen3.5-4B-JANG_4S
**Goal:** Verify gen_prompt_len stripping, cache store/fetch, SSM companion

```
1. Turn 1: Generate with messages=[{role:"user", content:"Hi"}]
2. Verify cache stored (NSLog "[Gen] Stored cache:")
3. Turn 2: Generate with messages=[{role:"user", content:"Hi"}, {role:"assistant", content:"Hello!"}, {role:"user", content:"How?"}]
4. Verify cache HIT (NSLog "[Gen] Cache HIT:")
5. Verify SSM companion injected (NSLog "[Gen] Injected N SSM companion states")
6. Verify remaining tokens < total tokens (prefix hit working)
```

### Test 3: TurboQuant Compress + Fetch
**Model:** Qwen3.5-4B-JANG_4S
**Goal:** Verify TQ compression on store, decompression on fetch

```
1. Enable TQ: scheduler.config.enableTurboQuant = true
2. Turn 1: Generate → verify cache stored with compressed attention
3. Turn 2: Generate → verify cache HIT with decompressed attention
4. Verify output quality not degraded (compare first 5 tokens with TQ off)
```

### Test 4: MoE Model (MiniMax)
**Model:** MiniMax-M2.5-JANG_2L (WARNING: 41B, needs ~24GB+ RAM)
**Goal:** Verify bfloat16 conversion, MoE routing, cache store/fetch

```
1. Load model → verify bfloat16 conversion log
2. Verify isHybrid == false
3. Turn 1: Generate → verify cache stored
4. Turn 2: Generate → verify cache HIT (no SSM companion needed)
```

### Test 5: Disk Cache Round-Trip
**Model:** Qwen3.5-4B-JANG_4S
**Goal:** Verify disk cache store/fetch with TQ

```
1. Enable disk cache: applyUserConfig(enableDiskCache: true)
2. Turn 1: Generate → verify disk cache stored (safetensors file created)
3. Clear memory cache: scheduler.cache.clearAll()
4. Turn 2: Generate → verify disk cache HIT → L2→L1 promotion
5. Verify SSM companion still works after disk round-trip
```

### Test 6: Paged Cache Block Sharing
**Model:** Qwen3.5-4B-JANG_4S
**Goal:** Verify paged blocks share prefix across turns

```
1. Turn 1: Long prompt (>128 tokens) → verify multiple blocks stored
2. Turn 2: Same prefix + new tokens → verify block reuse (COW fork)
3. Verify SSM in last block only, nil in non-last
```

---

## Registry Gaps Found

| Issue | Impact | Fix |
|-------|--------|-----|
| nemotron_h not registered | Nemotron models fail to load | Add Nemotron model class or map to Qwen35 if compatible |
| mistral3/mistral4 FP8 blocked | Mistral Small 4 can't load | Implement FP8 weight support |
| No MLA attention implementation | MLA models (DeepSeek-V3) unsupported | Future: add MLA attention module |
| VL forward pass not implemented | Vision-language models load but can't process images | Future: add vision encoder integration |

---

## Implementation Order (recommended)

1. **Test 1-2**: Basic load + cache hit (Qwen3.5-4B) — validates core cache stack
2. **Test 3**: TurboQuant — validates Phase 5 changes
3. **Test 5**: Disk cache round-trip — validates Bug #4 fix
4. **Test 6**: Paged blocks — validates Bug #1 fix
5. **Test 4**: MoE (MiniMax) — validates bfloat16 + non-hybrid path
6. Registry fix: Add nemotron_h support (if needed)
