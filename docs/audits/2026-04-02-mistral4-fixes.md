# Mistral 4 Small — Fixes & Audit (2026-04-02)

Branch: `feature/vmlx`

## Commits

| Commit | Files Changed | Description |
|--------|--------------|-------------|
| `f70eb83` | ModelLoader.swift, Mistral4Model.swift | EOS detection for nested text_config + FP8 metadata cleanup |
| `47d4497` | ModelContainer.swift | Auto-map enable_thinking to reasoning_effort for Mistral 4 |
| `b841a0f` | ModelDetector.swift, ModelContainer.swift | Add vHeadDim detection, wire to TQ config for MLA models |

---

## Fix 1: EOS Token Detection for Nested `text_config` (CRITICAL)

**File:** `Packages/VMLXRuntime/Sources/VMLXRuntime/Core/ModelLoader.swift:67-85`

**Problem:** `ModelLoader.eosTokenIds` only checked `config["eos_token_id"]` at the top level. For Mistral 4 (and other VLM models), `eos_token_id` is nested inside `text_config`:

```json
{
  "model_type": "mistral3",
  "text_config": {
    "eos_token_id": 2
  }
}
```

The top level has no `eos_token_id`. Result: empty EOS set, generation never stops at `</s>`, model outputs garbled text after the natural end.

**Fix:** Added fallback to `text_config`:

```swift
// Fallback to text_config (VLM models nest eos_token_id there)
if ids.isEmpty, let tc = config["text_config"] as? [String: Any] {
    if let eosIds = tc["eos_token_id"] as? [Int] { ids.formUnion(eosIds) }
    else if let eosId = tc["eos_token_id"] as? Int { ids.insert(eosId) }
}
```

**Models affected:**
- All Mistral-Small-4-119B variants (eos_token_id=2 only in text_config)
- Qwen3.5-35B-A3B-FP8 variants (eos_token_id=248044 only in text_config)

**Models NOT affected (already have top-level EOS):**
- Nemotron Cascade/Super (eos=11/2 at top level)
- MiniMax M2.5 (eos=200020 at top level)
- All JANG Qwen3.5 models (eos=248046 at both levels)
- Gemma4 (eos=[1,106] at top level)

---

## Fix 2: FP8 Metadata Cleanup in Sanitize

**File:** `Packages/VMLXRuntime/Sources/VMLXRuntime/Models/Mistral4Model.swift:687-690`

**Problem:** JANG quantized Mistral 4 models retain leftover FP8 metadata keys from the original HF checkpoint:
- `mlp.experts.down_proj_activation_scale`
- `mlp.experts.down_proj_scale_inv`
- `mlp.experts.gate_up_proj_activation_scale`
- `mlp.experts.gate_up_proj_scale_inv`

These didn't match the existing filters (`hasSuffix(".activation_scale")` vs `_activation_scale`).

**Fix:** Broadened the filter to `contains("activation_scale") || contains("scale_inv")`.

**Impact:** Harmless (unmatched keys were silently ignored by `model.update(verify: [])`), but cleaner weight loading.

---

## Fix 3: Auto-Map `enable_thinking` to `reasoning_effort` (CRITICAL)

**File:** `Packages/VMLXRuntime/Sources/VMLXRuntime/Core/ModelContainer.swift:173-191`

**Problem:** Mistral 4's chat template uses `reasoning_effort` ("none" or "high"), not `enable_thinking`. The code only passed `enable_thinking` to the template context. Mistral 4's Jinja template ignored this, defaulting to `reasoning_effort: "none"` (no thinking).

**Evidence:** Python VMLX server.py lines 2070-2076 and 3657-3667 explicitly auto-map:
```python
# Auto-map enable_thinking -> reasoning_effort for Mistral 4
if request.enable_thinking is True and "reasoning_effort" not in _ct_kwargs:
    _ct_kwargs["reasoning_effort"] = "high"
```

**Fix:** Added matching logic in `applyChatTemplate`:
```swift
if context["reasoning_effort"] == nil {
    if enableThinking { context["reasoning_effort"] = "high" }
    else { context["reasoning_effort"] = "none" }
}
```

**Safety:** Jinja2 templates silently ignore unknown variables. Qwen/Gemma/MiniMax templates that use `enable_thinking` are unaffected by the extra `reasoning_effort` context var.

---

## Fix 4: Add `vHeadDim` to TurboQuant Config for MLA Models

**Files:**
- `Packages/VMLXRuntime/Sources/VMLXRuntime/Core/ModelDetector.swift` — added `vHeadDim: Int?` field
- `Packages/VMLXRuntime/Sources/VMLXRuntime/Core/ModelContainer.swift` — wire to TQ config

**Problem:** TurboQuantConfig had `mlaKeyDim` set from `qkNopeHeadDim + qkRopeHeadDim = 128` but `mlaValueDim` was never set. The JANG path (`JangLoader.buildTQConfig`) accepted `vHeadDim` but it was never passed from `ModelContainer`.

For Mistral 4: keyDim == valueDim == 128, so TQ's asymmetric guard (`keyDim == valueDim`) passes and encoding works. But the config was incomplete.

**Fix:**
1. Added `vHeadDim` to `DetectedModel` struct
2. Read `v_head_dim` from config.json (top-level and text_config fallback)
3. Set `defaultTQ.mlaValueDim = model.detected.vHeadDim` in non-JANG path
4. Pass `vHeadDim: model.detected.vHeadDim` to `JangLoader.buildTQConfig` in JANG path

**Models affected:** Only MLA models (Mistral4 with kvLoraRank=256). All other models have `kvLoraRank=nil` and skip this code path.

---

## Verification Audit

### Architecture Correctness (vs Python mlx-lm mistral4.py)

| Component | Status | Notes |
|-----------|--------|-------|
| MLA Attention (Q LoRA, KV LoRA, RoPE) | Correct | Matches Python line-by-line |
| YaRN RoPE frequencies | Correct | Same harmonic mean formula as Python |
| MoE gate (softmax routing, top-4) | Correct | Same argPartition + normalized weights |
| MoE experts (VMLXSwitchGLU) | Correct | Same gatherQuantizedMM path as Python |
| Shared expert | Correct | Same add-to-MoE-output |
| FP8 dequantization | Correct | Block-scaled with block_size=128 |
| Weight key mapping (sanitize) | Correct | language_model. prefix stripped, experts pre-stacked |
| KV cache (VMLXKVCacheSimple) | Correct | Stores decompressed KV, same as Python |
| bfloat16 conversion | Correct | Triggered by kvLoraRank > 0 |

### Pipeline Integration

| Component | Status | Notes |
|-----------|--------|-------|
| EOS detection | Fixed | Now reads from text_config |
| Chat template (reasoning) | Fixed | reasoning_effort auto-mapped |
| TurboQuant config | Fixed | mlaValueDim now set |
| TurboQuant encode/decode | OK | keyDim==valueDim==128, symmetric path works |
| KV quantization (kvBits) | OK | Separate from TQ, works with MLA shapes |
| CacheCoordinator | OK | No MLA-specific issues, isHybrid=false |
| Streaming (parsers) | OK | MistralReasoningParser handles [THINK]/[/THINK] |
| Tool parsing | OK | MistralToolParser handles [TOOL_CALLS] |
| Sampler | OK | compiledCategoricalSample used (non-hybrid) |
| Prefill chunking | OK | Adaptive step size, clearCache after prefill |

### Model Config Audit (All Local Models)

| Model | model_type | EOS Location | MLA? | Status |
|-------|-----------|-------------|------|--------|
| Mistral-Small-4-119B (4 variants) | mistral3 | text_config ONLY | Yes (kvlr=256) | Fixed |
| Nemotron-Cascade-30B (2 variants) | nemotron_h | top-level | No | OK |
| Nemotron-Super-120B | nemotron_h | top-level | No | OK |
| MiniMax-M2.5 | minimax_m2 | top-level | No | OK |
| Qwen3.5-4B (3 variants) | qwen3_5 | both | No | OK |
| Qwen3.5-9B (3 variants) | qwen3_5 | both | No | OK |
| Qwen3.5-27B | qwen3_5 | both | No | OK |
| Qwen3.5-35B (6 variants) | qwen3_5_moe | both (JANG) / text_config only (FP8) | No | Fixed (FP8) |
| Qwen3.5-122B (2 variants) | qwen3_5_moe | both | No | OK |
| Gemma4-26B (2 variants) | gemma4 | top-level | No | OK |
