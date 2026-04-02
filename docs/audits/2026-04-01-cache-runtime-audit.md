# Cache Runtime Audit

Date: 2026-04-01

Scope:
- VMLX prefix cache, paged cache, memory/L2 cache, SSM companion cache, SSM async re-derive
- JANG / hybrid model handling for Nemotron H, Cascade-class Nemotron H configs, Qwen 3.5
- Mistral Small 4 MLA path
- TurboQuant wiring
- MLX cache path vs VMLX cache path
- UI/settings wiring into actual runtime behavior

Status legend:
- `Working`: implemented and used on the hot path
- `Partial`: implemented but limited, fallback-heavy, or only works on some paths
- `Miswired`: code exists but important connections are wrong or missing
- `Broken`: active behavior is directly harmful or invalidates expected state
- `Not Applicable`: this runtime uses a different design, so the named subsystem does not exist there
- `Dead`: infrastructure exists but is not used by the current runtime path

Cache-level terminology used in this audit:
- `L1`: in-process RAM cache layers such as VMLX `MemoryCache` or MLX hot session cache
- `L2`: local persistent cache layers such as VMLX `DiskCache` / `TQDiskStore` or MLX SSD cache

## Executive Summary

The main problem is not one missing feature. It is that the codebase has two different local-runtime cache systems with different capabilities:
- VMLX has `CacheCoordinator`, paged block cache, memory cache, SSM companion cache, and disk cache.
- MLX has `KVCacheStore`, session cache, SSD cache, and a background-built prefix warm cache.

The UI presents several cache controls as if they apply to "local inference" broadly, but some of them only affect VMLX. Separately, VMLX has several places where the intended advanced behavior exists as infrastructure but is not actually used on the live generation path.

There is also a routing mismatch:
- `ChatEngine` tries `VMLXServiceBridge` before `MLXService`.
- `VMLXService.handles(...)` accepts almost any non-remote model string.
- `VMLXServiceBridge` only forces MLX routing for `model_type` values listed in `VMLXModelRegistry.mlxServiceOnlyTypes`, but that set is currently empty.

So in practice local-model routing is often "try VMLX first, then fall back on failure" rather than an explicit up-front runtime choice.

## Runtime Coverage Matrix

This audit explicitly covers both runtime families and both model families the app exposes locally.

### VMLXRuntime / VMLXService path

Applies to:
- JANG models discovered by `ModelDetector` and loaded through `VMLXServiceBridge`
- VMLX-native non-JANG local models that still route through `VMLXService`

Cache-stack layers expected on this path:
- KV cache objects from `VMLXModelContainer.newCache(...)`
- VMLX L1 memory cache via `MemoryCache`
- VMLX prefix cache or paged block cache via `CacheCoordinator`
- VMLX L2 disk cache via `DiskCache`
- Hybrid SSM companion cache via `SSMStateCache`
- Optional KV quantization via `VMLXQuantizedKVCache`
- Optional TurboQuant policy derived from JANG/config metadata

Model families explicitly audited on this path:
- Nemotron H / Cascade-style Nemotron H configs
- Qwen 3.5 hybrid models
- Mistral Small 4

### MLX / ModelRuntime path

Applies to:
- MLXService-routed local models handled by `ModelRuntime`
- MLX hybrid models using `MambaCache`-based reuse
- MLX standard transformer models using normal session/prefix KV reuse

Cache-stack layers expected on this path:
- per-session hot cache in `KVCacheStore`
- SSD cache in `KVCacheStore`
- background-built prefix warm cache in `KVCacheStore`
- hybrid stable-boundary snapshot path in `MLXGenerationEngine`
- optional KV quantization via `RuntimeConfig.kvBits`

Cache-stack layers not present on this path:
- no VMLX `CacheCoordinator`
- no VMLX paged cache
- no VMLX SSM companion cache
- no VMLX TurboQuant path

### UI / Routing / Service layer

This audit also covers the glue between the two engines:
- [ChatEngine.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/Chat/ChatEngine.swift)
- [VMLXServiceBridge.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/Inference/VMLXServiceBridge.swift)
- [MLXService.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/Inference/MLXService.swift)
- [ConfigurationView.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Views/Settings/ConfigurationView.swift)

That means the checklist is intended to be complete across:
- runtime selection
- per-runtime cache-stack behavior
- model-specific cache behavior
- UI/settings wiring into those codepaths

## Progress Update

Implemented in this pass:
- `Working` Chat stop now propagates into the live VMLX generation producer instead of only cancelling the outer UI task.
  Evidence: [VMLXService.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXService.swift) now attaches `continuation.onTermination` to cancel the producer task in both `streamDeltas(...)` and `streamWithTools(...)`. This was the missing link behind stop/unload getting stuck on in-flight VMLX generations such as Mistral 4.

- `Working` Chat-model context menu only offers `Unload Model` when the selected model is actually the currently loaded local runtime model.
  Evidence: [FloatingInputCard.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Views/Chat/FloatingInputCard.swift) now resolves the selected model against the real VMLX and MLX loaded-state sources and performs targeted unload instead of blindly clearing both runtimes.

- `Working` Parser selections are now normalized from one shared source across the three UI surfaces:
  - chat model options popover
  - downloaded-model detail sheet
  - Settings -> Local Inference -> Parsers
  Evidence: [ModelOptions.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Models/Configuration/ModelOptions.swift), [ModelDetailView.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Views/Model/ModelDetailView.swift), and [ConfigurationView.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Views/Settings/ConfigurationView.swift) now use the same parser option definitions and normalize older saved aliases like `think_tags`.

- `Working` Chat-surface parser changes now persist instead of being session-only.
  Evidence: [FloatingInputCard.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Views/Chat/FloatingInputCard.swift) now normalizes and saves `activeModelOptions` changes back to `ModelOptionsStore`.

- `Working` Tool parser override remains a VMLX runtime concern and is still applied on the live request path.
  Evidence: [VMLXServiceBridge.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/Inference/VMLXServiceBridge.swift) resolves per-model then global tool-parser overrides and passes them into `SamplingParams`, which are consumed by [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift).

- `Working` VMLX reasoning-parser override is now honored on the live runtime path instead of being hardwired off.
  Evidence: [ReasoningParser.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Parsers/ReasoningParser.swift) now exposes `reasoningParserForFormat(...)`, and [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift) now resolves per-request overrides or family config into the active `StreamAccumulator` reasoning parser.

- `Working` Reasoning parser selection is now applied on the chat/work UI parsing path instead of being a dead setting.
  Evidence: [StreamingMiddleware.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Utils/StreamingMiddleware.swift), [StreamingDeltaProcessor.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Utils/StreamingDeltaProcessor.swift), [ChatView.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Views/Chat/ChatView.swift), and [WorkSession.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Views/Work/WorkSession.swift) now resolve the effective reasoning parser from per-model plus global settings and apply the matching preprocessing for:
  - `<think>` models
  - Mistral `[THINK]` models
  - GPT-OSS / Harmony channel-tag models

- `Working` VMLX reasoning chunks are bridged back into the app's standard `<think>` stream shape, so runtime-side parser selection and UI thinking rendering no longer fight each other.
  Evidence: [VMLXService.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXService.swift) now wraps `.thinking(...)` events with `<think>` / `</think>` markers before yielding string deltas to Osaurus.

- `Working` VMLX generation stats now expose live cache bytes so cache footprint is visible in the chat stats row.
  Evidence: [GenerationStats.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Models/Chat/GenerationStats.swift), [ModelService.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/Inference/ModelService.swift), [VMLXService.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXService.swift), and [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift) now stream `KV x.xx GB` alongside TTFT / PP / TG stats.

Still intentionally true after this pass:
- Tool-parser override is still not a first-class MLX override path; MLX continues to rely on its upstream model/config-driven tool-call format.
- Reasoning-parser override is now honored on the VMLX runtime path and on the UI parsing path. MLX still uses the UI parsing path only; it did not gain VMLX's runtime parser stack.
- Cache-footprint stats are currently sourced from the VMLX live generation cache path; MLX does not yet emit the same cache-byte stat.
- Targeted VMLX parser test execution is still blocked by the long-standing unrelated [ModelConfigTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Core/ModelConfigTests.swift) compile failures, because SwiftPM still compiles that broken file before running filtered parser suites.

Implemented in the latest hybrid/TQ cache pass:
- `Working` VMLX prefix cache is now populated even when the memory tier is enabled, as long as prefix cache itself exists.
  Evidence: [CacheCoordinator.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift) now stores into `prefixCache` whenever that tier is configured, instead of silently skipping it whenever memory cache is on.

- `Working` `PrefixCache.clear()` is now lock-protected and resets its own counters.
  Evidence: [PrefixCache.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/PrefixCache.swift) now wraps `clear()` in the same lock used by the rest of the type and resets `hits` / `misses` along with trie and LRU state.

- `Working` TurboQuant policy on the live actor path now resolves from the loaded model's `TurboQuantConfig` instead of a hardcoded `3-bit / seed 42` fallback.
  Evidence: [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift) now gates the active TQ path on `container.turboQuantConfig` and encodes eligible layers through `TurboQuantLayerCache.encodeAttentionLayer(...)`.

- `Partial` Cross-turn cache store no longer always degrades attention back to float on cache-hit turns.
  Evidence: [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift) now reuses a compressed restored prefix when available, encodes only the fresh tail for eligible layers, and stores `.compressedAttention` back into `scheduler.cache.store(...)` when that merge is valid.

- `Partial` Paged cache no longer eagerly destroys compressed attention on the normal block-store/fetch path.
  Evidence: [CacheCoordinator.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift) now slices `.compressedAttention` per block and attempts to merge block slices back into `.compressedAttention` on fetch. Sink-only edge slices still fall back to float for correctness.

- `Working` Guardrail tests now cover the newly-wired prefix and TurboQuant cache behavior.
  Evidence:
  - [CacheCoordinatorTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/CacheCoordinatorTests.swift)
  - [TurboQuantLayerCacheTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Quantization/TurboQuantLayerCacheTests.swift)

Implemented in the current pass:
- `Working` App-side model-option family matching is no longer one broad "Qwen-ish thinking models" bucket.
  Evidence: [ModelOptions.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Models/Configuration/ModelOptions.swift) now separates:
  - Qwen / QwQ reasoning models
  - MiniMax
  - DeepSeek
  - Mistral / Mixtral
  - Phi reasoning
  This keeps MiniMax distinct from Qwen, exposes a proper Mistral thinking toggle, and intentionally still does not invent a Nemotron/Cascade thinking toggle that the runtime family config does not declare.

- `Partial` Large JANG MoE prefills now use an adaptive chunk cap instead of always pushing the full configured prefill step through one expert pass.
  Evidence: [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift) now lowers the effective prefill chunk size for JANG models with `numExperts >= 256`, with explicit logging when the adaptive cap is active.
  Reason: live process sampling during a stuck `Qwen3.5-35B-A3B-JANG_4K` request showed the runtime burning inside:
  - `VMLXRuntimeActor.generateStream(...)`
  - `Qwen35TopLevelModel.callAsFunction(...)`
  - `Qwen35SparseMoeBlock.callAsFunction(...)`
  - `VMLXSwitchGLU.callAsFunction(...)`
  - `VMLXQuantizedSwitchLinear.callAsFunction(...)`
  during chunked prefill, before the first hybrid boundary snapshot.
  This confirms the immediate issue is giant- MoE prefill cost on the quantized expert path, not a UI spinner bug or cache-restore deadlock.

## Checklist

- `Miswired` VMLX "continuous batching" is not active on the real generation path.
  Evidence: [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L424) cancels any in-flight generation before starting another one, while [Scheduler.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Scheduler/Scheduler.swift#L18) only defines scheduler/request-queue infrastructure.

- `Partial` VMLX paged cache is a block-level prefix reconstruction layer, not a live paged KV runtime buffer.
  Evidence: [CacheCoordinator.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift#L331) stores completed caches into paged blocks, but [PagedCacheManager.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/PagedCacheManager.swift#L110) allocation/table APIs are otherwise unused by the live generation path.

- `Working` VMLX prefix cache is populated when configured, including the common "paged off, memory on" configuration.
  Evidence: [CacheCoordinator.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift#L92) still constructs `PrefixCache` only when paged cache is off, and [CacheCoordinator.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift#L355) now stores into it whenever the tier exists instead of skipping it when memory cache is enabled.

- `Working` VMLX memory cache `clear()` now fully resets byte accounting and counters.
  Evidence: [MemoryCache.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/MemoryCache.swift#L115) now resets `currentMemory`, `effectiveMemoryLimit`, memory-pressure bookkeeping, and stats instead of only dropping `entries`.

- `Working` VMLX cancellation and generic request failure no longer wipe the entire cache stack.
  Evidence: [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift) now uses request-scoped invalidation plus `clearVolatile()` on the narrowed failure paths, and [CacheCoordinator.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift#L380) now exposes targeted invalidation and volatile-layer clearing instead of forcing full L2 destruction for every request problem.

- `Working` VMLX disk cache now guards against stale-file resurrection after clear/remove.
  Evidence: [DiskCache.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/DiskCache.swift#L35) now versions invalidations, cancels pending writes, and only commits a background temp-file write if the entry is still current at commit time.

- `Partial` VMLX SSM re-derive is now on the live hybrid recovery path, but large or unavailable recoveries still intentionally full-prefill the current request.
  Evidence: [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L635) now requests boundary-aligned re-derive for exact hybrid hit replay, and [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L723) does the same for hybrid `.partialHit`. [SSMReDeriver.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/SSMReDeriver.swift#L92) now decides sync-vs-async using `stableBoundary`, which matches the actual re-derive work size.

- `Partial` Hybrid partial-hit recovery now uses real boundary-aligned SSM re-derive when a checkpoint can be returned in time, but still falls back to full prefill for the current request when recovery is pending or unavailable.
  Evidence: [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L705) computes the matched hybrid boundary, normalizes exact-hit replay to `N-1` when needed, and restores attention KV only after a checkpoint is available. The fallback path remains in [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L741).

- `Working` Hybrid cache store now captures SSM state at the real `storeTokens` boundary, including restored cache-hit turns, so the companion checkpoint refresh can stay aligned with the attention cache being written.
  Evidence: [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L781) captures a boundary snapshot either from restored cache state or exactly when chunked prefill crosses the store boundary, and [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L1117) writes those SSM layers back alongside the stored attention layers. [CacheCoordinator.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift#L352) then refreshes the companion cache from the stored hybrid entry.

- `Working` Qwen 3.5 hybrid cache object layout is correct at the model level.
  Evidence: [Qwen35Model.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Models/Qwen35Model.swift#L639) uses `VMLXMambaCache` for linear-attention layers and `VMLXKVCacheSimple` for full-attention layers.

- `Partial` Qwen 3.5 35B JANG load stalls are currently traced to quantized MoE prefill cost, not to the cache-stack restore path.
  Evidence: a live `sample` of the app process while `Qwen3.5-35B-A3B-JANG_4K` was "loading forever" showed the hot stack in `Qwen35SparseMoeBlock -> VMLXSwitchGLU -> VMLXQuantizedSwitchLinear` during `_chunkedPrefill`, before any `[Gen] SSM snapshot` or `[Gen] Final prefill token` log line appeared. This is now partially mitigated by adaptive prefill chunking, but it still needs a dedicated MoE-path optimization audit.

- `Working` Nemotron H hybrid cache object layout is correct at the model level, including Cascade-style configs that differ in MoE internals.
  Evidence: [NemotronHModel.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Models/NemotronHModel.swift#L522) maps `M` to `VMLXMambaCache`, `*` to `VMLXKVCacheSimple`, and expert-only layers to a zero-sized placeholder. The model code also documents Super-vs-Cascade latent-projection differences in the mixer/MoE implementation.

- `Miswired` TurboQuant on the live VMLX path is only "compress then immediately decode once back to float," not a persistent compressed runtime KV cache.
  Evidence: [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L755) encodes prefill KV, decodes it immediately, and writes decoded float buffers back into the normal cache objects.

- `Partial` The live in-memory TurboQuant transform still runs only after uncached prefill, but cache-hit turns no longer force cross-turn cache storage back to float.
  Evidence: [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L947) still keeps the decode-once in-memory TQ step behind `cachedTokenCount == 0`, while the later store path in [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L1237) now reuses compressed restored prefixes and encodes fresh tails for eligible layers on cache-hit turns.

- `Dead` `TurboQuantKVCache` exists but is not used on the active runtime path.
  Evidence: the class is implemented in [TurboQuantKVCache.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/TurboQuantKVCache.swift), but repo usage search does not show any construction or integration from the active generation path.

- `Dead` TQ-native disk serialization exists but is not used.
  Evidence: [TQDiskStore.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/TQDiskStore.swift) is present, but repo usage search finds no callers.

- `Partial` Cross-turn VMLX cache store can now preserve `.compressedAttention`, but only for layer shapes/policies that `TurboQuantLayerCache` can safely encode.
  Evidence: [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L1237) now builds compressed prefix/tail entries and stores merged `.compressedAttention` where valid, while unsupported/asymmetric shapes still deliberately fall back to float.

- `Partial` Paged cache now preserves compressed attention on the normal block-store/fetch path, with a narrow float fallback for sink-only edge slices.
  Evidence: [CacheCoordinator.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift#L398) now slices `.compressedAttention` directly per block, and [CacheCoordinator.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift#L271) now attempts to merge compressed block slices back into one compressed layer on fetch.

- `Miswired` KV quantization and TurboQuant are two different mechanisms, but the UI copy does not clearly separate them.
  Evidence: VMLX runtime cache creation swaps `VMLXKVCacheSimple` to `VMLXQuantizedKVCache` in [ModelContainer.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Core/ModelContainer.swift#L201), while TurboQuant is applied later in the actor hot path in [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L755).

- `Miswired` Mistral Small 4 is not using the latent MLA cache path even though generic MLA latent-cache infrastructure exists.
  Evidence: generic latent cache exists in [MLAAttention.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Models/MLAAttention.swift#L72) and is used by the generic MLA attention module, but [Mistral4Model.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Models/Mistral4Model.swift#L664) still allocates plain `VMLXKVCacheSimple` per layer.

- `Partial` Mistral Small 4 decode hot path is improved but still structurally heavier than it should be.
  Evidence: it is still doing full-key/full-value cache storage through standard KV cache objects instead of storing latent MLA state. The broadcast fix for `kPe` reduces one obvious waste, but the model remains on the full-KV path.

- `Miswired` "Local Inference" cache-stack toggles are VMLX-only in practice, but the UI presents them as global local-runtime controls.
  Evidence: the UI presents TurboQuant, Disk Cache, and Memory Cache Budget under [ConfigurationView.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Views/Settings/ConfigurationView.swift#L438), but MLX-side runtime config in [RuntimeConfig.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime/RuntimeConfig.swift#L10) only includes `topP`, `kvBits`, `kvGroup`, `quantStart`, `maxKV`, and `prefillStep`. Those cache-stack toggles are only forwarded through the VMLX bridge in [VMLXServiceBridge.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/Inference/VMLXServiceBridge.swift#L81).

- `Working` MLX has its own separate hot RAM + SSD cache path.
  Evidence: [KVCacheStore.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime/KVCacheStore.swift) manages hot session cache plus SSD persistence and is used by [ModelRuntime.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime.swift).

- `Partial` MLX prefix cache is only a background-built warm cache for stable system/tools content, not the same thing as VMLX paged/prefix cache.
  Evidence: [ModelRuntime.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime.swift#L366) builds prefix cache using a synthetic `system + user("Hi")` prompt after generation finishes.

- `Working` MLX hybrid reuse uses a stable-boundary two-phase snapshot for non-trimmable Mamba caches.
  Evidence: [MLXGenerationEngine.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime/MLXGenerationEngine.swift#L264) explicitly implements a two-phase prefill so hybrid models can snapshot before the generation prefix and avoid Mamba trim failures on later turns.

- `Not Applicable` MLX does not have a separate async SSM re-derive / companion-cache subsystem.
  Evidence: OsaurusCore search does not show an MLX analogue to `SSMReDeriver`; the MLX path relies on direct cache reuse plus stable-boundary snapshotting in [MLXGenerationEngine.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime/MLXGenerationEngine.swift#L264), not a background SSM recovery path.

- `Partial` MLX L2 cache has the same detached-write eventual-consistency risk as VMLX.
  Evidence: [KVCacheStore.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime/KVCacheStore.swift#L211) writes SSD entries in `Task.detached`, while [KVCacheStore.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime/KVCacheStore.swift#L536) clears files synchronously without cancelling pending writers. A clear/unload can therefore be followed by a stale file being recreated.

- `Miswired` MLX prefix-cache store/promotion bypasses hot-tier budget enforcement.
  Evidence: [KVCacheStore.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime/KVCacheStore.swift#L452) stores prefix caches via `putCache(...)` and immediate SSD save without a matching `ensureBudget(...)`, and [KVCacheStore.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime/KVCacheStore.swift#L364) promotes SSD-loaded prefix caches into RAM by incrementing `totalHotBytes` directly. This means warm-prefix activity can grow the hot tier past the intended budget until some later session-cache write triggers eviction.

- `Miswired` UI "Always-On Layers" copy is inaccurate.
  Evidence: [ConfigurationView.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Views/Settings/ConfigurationView.swift#L465) says memory cache, prefix cache, and SSM companion cache are always on, but actual VMLX behavior depends on `usePagedCache`, `useMemoryAwareCache`, and whether the runtime path ever stores into those layers.

- `Working` VMLX prefix cache clear path is now lock-protected.
  Evidence: [PrefixCache.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/PrefixCache.swift#L48) now clears trie/LRU state and stats under the same lock used for normal cache operations.

- `Miswired` Local-model routing is optimistic and failure-driven rather than explicit.
  Evidence: [ChatEngine.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/Chat/ChatEngine.swift#L18) tries `VMLXServiceBridge` before `MLXService`; [VMLXService.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXService.swift#L60) accepts almost any local model string; and [ModelRegistry.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Models/ModelRegistry.swift#L64) currently defines an empty `mlxServiceOnlyTypes` set. That means many models are routed to MLX only after VMLX load failure instead of by an up-front capability decision.

- `Working` KV quantization and TurboQuant both correctly skip SSM caches at the configuration level.
  Evidence: [ModelContainer.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Core/ModelContainer.swift#L201) only replaces `VMLXKVCacheSimple` with `VMLXQuantizedKVCache`, and [TurboQuantConfig.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/TurboQuantConfig.swift#L57) returns `nil` bit widths for SSM layers when `layerPattern` marks them as `.ssm`.

- `Partial` JANG-to-TurboQuant config split is conceptually correct, but the active runtime path does not preserve a real TurboQuant cache object after prefill.
  Evidence: [JangLoader.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/JangLoader.swift#L371) builds per-layer KV bit policy and supports hybrid layer patterns, while [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L755) still encodes and immediately decodes back to float instead of keeping `TurboQuantKVCache` live.

## Model-Specific Notes

- Nemotron H / Cascade:
  The model-level hybrid cache split is correct. The main remaining issues are shared runtime concerns, not model-construction bugs:
  - hybrid recovery is still fallback-heavy when re-derive is async or unavailable
  - the live runtime still degrades back to float-KV state after the decode-once TurboQuant step
  - cross-turn compressed-cache preservation now exists, but it is not yet the same thing as a live persistent `TurboQuantKVCache`

- Qwen 3.5:
  The model-level hybrid split is correct. On VMLX, hybrid SSM recovery is now targeted but still fallback-heavy for large boundaries and the 35B JANG path still needs a dedicated MoE prefill optimization pass. On MLX, there is no async SSM companion system; reuse depends on stable-boundary snapshotting and non-trimmable-cache handling.

- Mistral Small 4:
  It is not hybrid, so SSM companion logic does not apply. The bigger problems are MLA cache shape choice and TurboQuant/runtime-cache wiring. The generic latent MLA cache machinery exists but Mistral 4 is still on plain KV caches, so even a correct KV-only TurboQuant split does not give the full MLA memory/runtime benefit.

## Remaining Fix Order Recommendation

1. Decide whether VMLX paged cache is meant to stay a block-level prefix store or become a real live paged runtime buffer, then align code/UI/docs to that choice.
2. Decide whether TurboQuant remains a decode-once memory reduction path or becomes a true persistent compressed runtime cache object.
3. Finish the remaining hybrid fallback policy work so large-boundary re-derive stops feeling like an almost-always-prefill path.
4. Fix MLX hot-tier prefix-budget enforcement so MLX cache growth matches the configured limits.
5. Rework Mistral 4 onto latent MLA cache if the target is real MLA memory/runtime efficiency.
6. Split UI labels so MLX-only and VMLX-only controls are explicit.

## Exact Work Needed

This section is the concrete implementation ledger: what code has to change for the system to actually match the intended design.

### 1. Stop VMLX from nuking L2 on cancellation/error

Status:
- Completed for the request-cancellation / generic-failure path.
- Full user-triggered unload/reset still intentionally uses explicit full-clear behavior.

Files:
- [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L1072)
- [CacheCoordinator.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift#L331)

Exact changes needed:
- Remove unconditional `scheduler.cache.clearAll()` from the general cancellation path.
- Split cache invalidation APIs into at least:
  - volatile-only clear (`MemoryCache`, in-flight block/prefix state if needed)
  - targeted key invalidation for one token sequence
  - explicit full clear for user-triggered unload/reset
- On generation failure, only invalidate the cache key involved in the failing request if the failure is cache-corruption-related. Do not drop the whole disk cache.

Done means:
- Canceling a generation does not delete unrelated L2 entries.
- One malformed cache entry can be invalidated without destroying the rest of the cache stack.

### 2. Fix `MemoryCache.clear()` so unload/clear actually resets L1 state

Status:
- Completed.

Files:
- [MemoryCache.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/MemoryCache.swift#L115)

Exact changes needed:
- Reset `currentMemory` to `0`.
- Reset `effectiveMemoryLimit` back to `baseMemoryLimit`.
- Reset memory-pressure bookkeeping such as `lastPressureCheck`.
- Decide whether hit/miss/eviction counters should reset on clear; either choice is fine, but it must be intentional.

Done means:
- After unload/clear, the next store behaves like an empty cache instead of a full one.

### 3. Make hybrid partial-hit recovery use real SSM re-derive instead of always full-prefill

Status:
- Partially completed.
- Sync-capable boundaries now use real re-derive; large/unavailable recoveries still intentionally fall back for the current request.

Files:
- [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L639)
- [SSMReDeriver.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/SSMReDeriver.swift#L92)

Exact changes needed:
- On `.partialHit` for hybrid models, compute the matched stable boundary from the cached prefix length.
- Call `SSMReDeriver.requestReDerive(...)` with that boundary.
- If the request is below the sync threshold, wait for the checkpoint, inject both attention cache and SSM state, and continue as a real prefix hit.
- If above the threshold, start async re-derive and choose the fallback explicitly:
  - either full prefill for the current request, or
  - background recovery for the next request only.
- Keep the current full-prefill path only as an intentional fallback, not as the only path.

Done means:
- Small hybrid partial hits no longer throw away usable attention KV.
- Large hybrid partial hits at least schedule re-derive for a later hit instead of staying permanently degraded.

### 4. Refresh SSM companion state on cache-hit turns too

Status:
- Completed for the current VMLX generation/store path.

Files:
- [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L709)
- [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L1038)
- [CacheCoordinator.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift#L352)

Exact changes needed:
- Remove the `cachedTokenCount == 0` restriction for SSM snapshotting.
- Capture SSM state at the actual `storeTokens` boundary, not only on uncached full prefill.
- Ensure the stored `HybridCache` for hybrid models includes both:
  - attention layers for the stored boundary
  - companion SSM layers for the same boundary
- If the generation path cannot naturally produce the boundary-aligned SSM state, trigger a targeted re-derive before store.

Done means:
- A cache-hit turn can refresh the hybrid companion checkpoint instead of storing attention-only state and poisoning the next turn into another partial hit.

### 5. Decide what paged cache actually is, then make code/UI/docs match

Status:
- Not completed.
- The code is clearer now, but the architectural decision is still open.

Files:
- [PagedCacheManager.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/PagedCacheManager.swift)
- [CacheCoordinator.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift#L371)

Exact changes needed:
- Choose one of two directions:
  - keep it as a block-hashed prefix store, and rename/document it that way, or
  - implement real live paged runtime allocation/table management and wire the generation loop to it
- Remove unused APIs if the first option is chosen.
- Stop describing the current implementation like a live paged attention allocator if it is not one.

Done means:
- The name, codepath, and UI/docs all describe the same thing.

### 6. Make TurboQuant a real runtime/storage path or narrow the feature claim

Status:
- Partially completed.
- Cache store/fetch preservation is now real, but the live runtime cache object is still decode-once float rather than persistent `TurboQuantKVCache`.

Files:
- [VMLXRuntimeActor.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift#L755)
- [TurboQuantKVCache.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/TurboQuantKVCache.swift)
- [TQDiskStore.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/TQDiskStore.swift)
- [JangLoader.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/JangLoader.swift#L371)

Exact changes needed:
- If the intent is a real TurboQuant path:
  - keep attention KV in a TurboQuant cache object after prefill instead of decoding immediately back to float
  - apply it on cache-hit turns too, not only `cachedTokenCount == 0`
  - store/fetch `.compressedAttention` on normal cross-turn cache paths
  - either integrate `TQDiskStore` or delete it
- If the intent is only “decode-once memory reduction,” then the toggle, UI text, and docs must say exactly that.

Done means:
- TurboQuant either survives past prefill as the active cache representation, or it is explicitly scoped down and no longer described like a persistent compressed cache system.

### 7. Keep the KV/SSM split explicit for JANG hybrid models

Status:
- Partially completed.
- The runtime now consults `TurboQuantConfig` on the live path, but model-family guardrail tests still need to be expanded.

Files:
- [TurboQuantConfig.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/TurboQuantConfig.swift#L57)
- [ModelContainer.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Core/ModelContainer.swift#L201)

Exact changes needed:
- Preserve the current rule that SSM layers are never quantized/compressed as KV.
- When wiring real TurboQuant, ensure hybrid layer patterns from JANG/config are always consulted before wrapping caches.
- Add tests proving that attention layers are wrapped and SSM layers are left alone for Nemotron H and Qwen 3.5 JANG models.

Done means:
- “TurboQuant only on KV, never on SSM” is enforced by tests, not just comments.

### 8. Rework Mistral Small 4 onto a latent MLA cache path

Files:
- [Mistral4Model.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Models/Mistral4Model.swift#L368)
- [Mistral4Model.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Models/Mistral4Model.swift#L664)
- [MLAAttention.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Models/MLAAttention.swift#L72)
- [KVCache.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Models/Utilities/KVCache.swift#L18)

Exact changes needed:
- Stop storing full decompressed K/V in `VMLXKVCacheSimple` for Mistral 4.
- Either:
  - switch Mistral 4 over to the generic MLA module that already consumes `MLALatentCache`, or
  - add a VMLX-native latent-cache type plus attention interface changes so the Mistral 4 attention block can cache latent state instead of full KV
- Only after that does TurboQuant-on-KV become a meaningful optimization for Mistral 4.

Done means:
- Mistral 4 cache growth scales with latent MLA state, not full `num_heads * head_dim` KV tensors.

### 9. Make local runtime routing explicit instead of failure-driven

Files:
- [ChatEngine.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/Chat/ChatEngine.swift#L18)
- [VMLXService.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXService.swift#L60)
- [ModelRegistry.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Models/ModelRegistry.swift#L64)

Exact changes needed:
- Stop having VMLX claim almost every local model up front.
- Route based on an explicit capability decision before load:
  - supported natively by VMLX
  - MLX-only
  - remote/non-local
- Populate and maintain the unsupported/MLX-only sets intentionally instead of leaving `mlxServiceOnlyTypes` empty.

Done means:
- A model lands on the intended runtime by policy, not by VMLX failing first.

### 10. Split UI/runtime controls so they match the actual engines

Files:
- [ConfigurationView.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Views/Settings/ConfigurationView.swift#L438)
- [RuntimeConfig.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime/RuntimeConfig.swift#L10)
- [VMLXServiceBridge.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/Inference/VMLXServiceBridge.swift#L81)

Exact changes needed:
- Separate MLX settings from VMLX settings in the UI copy and grouping.
- Remove or rewrite “always on” claims that are not true on the hot path.
- Make it obvious which toggles affect:
  - VMLX only
  - MLX only
  - both

Done means:
- A user can tell from the UI which cache/quant setting actually reaches the runtime they are using.

### 11. Fix MLX hot-tier accounting for prefix caches

Files:
- [KVCacheStore.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime/KVCacheStore.swift#L327)
- [KVCacheStore.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime/KVCacheStore.swift#L452)
- [ModelRuntime.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime.swift#L263)

Exact changes needed:
- After prefix-cache store and SSD promotion, run the same budget enforcement path used for session caches.
- Avoid incrementing `totalHotBytes` for promoted prefix caches without a corresponding eviction check.
- If needed, thread the current budget down from `ModelRuntime` so `KVCacheStore` can enforce it consistently.

Done means:
- Heavy prefix warming no longer lets MLX hot cache exceed budget silently.

### 12. Fix stale detached-write races on both VMLX and MLX L2

Files:
- [DiskCache.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/DiskCache.swift)
- [KVCacheStore.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Services/ModelRuntime/KVCacheStore.swift#L211)

Exact changes needed:
- Track pending disk-write tasks.
- Cancel or generation-tag them on clear/unload so an old write cannot recreate a deleted cache file.
- Only expose a cache entry as durable after its write has committed, or version the entry so stale writers cannot resurrect old state.

Done means:
- After clear/unload, old cache files do not reappear later from detached background writes.

## Execution Sequence

This is the implementation order that keeps both runtimes stable while we fix the cache stack.

### Current Progress

Implemented in this branch so far:
- `MemoryCache.clear()` now resets L1 bookkeeping instead of leaving stale usage behind.
- `CacheCoordinator` now has request-scoped `invalidate(tokens:)` and `clearVolatile()` paths so VMLX request failures no longer have to wipe L2.
- `VMLXRuntimeActor` cancellation/error handling was narrowed so cancellation no longer calls the full cache wipe path, and generic errors now use request-scoped invalidation plus volatile-layer clearing.
- `DiskCache` background writes now commit through a temp-file path with invalidation-aware checks, reducing stale-file resurrection after clear/remove.
- `KVCacheStore` background SSD writes now use the same invalidation-aware temp-file commit approach.
- `VMLXRuntimeActor` now calls `SSMReDeriver` on hybrid `.partialHit` and on exact hybrid hit replay-boundary recovery, instead of always discarding reusable attention KV.
- `SSMReDeriver` now bases sync-vs-async behavior on `stableBoundary` rather than total conversation length.
- Hybrid SSM snapshotting is now boundary-aligned to `storeTokens.count`, including cache-hit turns that start from restored attention/SSM state.
- `PrefixCache.clear()` now takes the cache lock and resets local stats.
- VMLX prefix cache is now populated in the real "paged off, memory on" configuration instead of being silently bypassed.
- `TurboQuantLayerCache` now preserves `.compressedAttention` through paged block slicing/fetch reconstruction where the block slice still contains compressed tail tokens.
- VMLX store-time TurboQuant now reuses compressed restored prefixes and encodes fresh tails on cache-hit turns instead of always writing float `.attention`.
- The live actor TurboQuant path now honors `container.turboQuantConfig` instead of a hardcoded bit-width fallback.
- Guardrail tests were added for:
  - `MemoryCache.clear()` bookkeeping
  - `CacheCoordinator` targeted invalidation / volatile-only clear behavior
  - stale background write prevention in both VMLX `DiskCache` and MLX `KVCacheStore`
  - prefix-cache population when memory cache is on
  - paged compressed-attention slice/merge preservation

Validation status:
- `swift build --package-path Packages/VMLXRuntime` passed.
- `swift build --package-path Packages/OsaurusCore` passed.
- `swift test` for both packages is currently blocked by unrelated pre-existing test compile failures outside this cache work, so the new tests could not be executed end-to-end in-package yet.

Remaining Phase 2 caveats after these changes:
- Exact hybrid hits are now made safe by requesting an `N-1` checkpoint for replay; if that checkpoint is not immediately available, the current request still intentionally falls back to full prefill.
- Hybrid partial hits are no longer "always full prefill," but large-boundary recovery is still fallback-heavy for the current request because the re-derive result may arrive asynchronously for the next hit instead.
- These changes are runtime-level and intentionally do not collapse model-specific cache layouts. Nemotron H, Cascade-style Nemotron H variants, and Qwen 3.5 still keep their own layer-pattern/model-construction differences; the shared fix here is only the VMLX hybrid cache/recovery behavior that sits under them.

### Phase 0. Guardrails before behavior changes

Goal:
- Extend tests first where current coverage is missing, so later runtime changes can be made safely.

Tests to add or extend:
- [MemoryCacheTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/MemoryCacheTests.swift)
  Add a test that `clear()` resets effective capacity/accounting.
- [CacheCoordinatorTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/CacheCoordinatorTests.swift)
  Add tests for:
  - prefix-cache store behavior when memory cache is on
  - targeted invalidation vs full clear
- [SSMReDeriverTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/SSMReDeriverTests.swift)
  Add an integration-style test for partial-hit recovery flow once wired.
- [KVCacheStoreTests.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Tests/Service/KVCacheStoreTests.swift)
  Add tests for prefix-cache budget enforcement and detached-write invalidation.

### Phase 1. Fix destructive invalidation first

Items:
1. VMLX cancel/error should not call global `clearAll()`.
2. `MemoryCache.clear()` must fully reset L1 bookkeeping.
3. VMLX and MLX detached L2 writes must not resurrect stale files after clear/unload.

Why first:
- These are correctness hazards that can invalidate every later measurement.

Verification:
- VMLX unit tests:
  - [MemoryCacheTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/MemoryCacheTests.swift)
  - [DiskCacheTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/DiskCacheTests.swift)
  - [CacheCoordinatorTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/CacheCoordinatorTests.swift)
- MLX unit tests:
  - [KVCacheStoreTests.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Tests/Service/KVCacheStoreTests.swift)

### Phase 2. Make hybrid VMLX cache correctness real

Items:
4. Store SSM companion state on cache-hit turns too.
5. Wire `SSMReDeriver` into hybrid partial-hit handling.
6. Keep hybrid attention/SSM boundary alignment correct when storing and restoring.

Why before TurboQuant:
- Hybrid correctness has to work in plain float state first, otherwise compressed-cache work just hides a broken base path.

Verification:
- [SSMReDeriverTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/SSMReDeriverTests.swift)
- [CacheIntegrationTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/CacheIntegrationTests.swift)
- [IntegrationTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/IntegrationTests.swift)
- [main.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXSmokeTest/main.swift)
  Expect hybrid cache fetch to stop falling back to `partialHit` for the normal healthy path.

### Phase 3. Normalize cache-layer semantics

Items:
7. Decide and enforce what “paged cache” means.
8. Fix prefix-cache semantics so VMLX prefix cache is either truly used or removed.
9. Fix MLX prefix-cache budget handling so warm-prefix activity obeys the same hot-tier limits.

Why here:
- This phase makes the cache stack itself coherent before model-specific optimization work.

Verification:
- VMLX:
  - [CacheCoordinatorTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/CacheCoordinatorTests.swift)
  - [CacheIntegrationTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/CacheIntegrationTests.swift)
- MLX:
  - [KVCacheStoreTests.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Tests/Service/KVCacheStoreTests.swift)
  - [ModelRuntimePrefixTests.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Tests/Model/ModelRuntimePrefixTests.swift)

### Phase 4. Make TurboQuant match its intended scope

Items:
10. Preserve the rule “TurboQuant applies to KV only, never SSM.”
11. Decide whether TurboQuant is a real persistent compressed cache path or only a decode-once memory reduction path.
12. Make cache-hit turns follow the same TurboQuant policy as uncached prefill turns.

Why after hybrid correctness:
- TurboQuant depends on the underlying boundary/store/fetch logic being trustworthy.

Verification:
- [TurboQuantConfigTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Quantization/TurboQuantConfigTests.swift)
- [TurboQuantKVCacheTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Quantization/TurboQuantKVCacheTests.swift)
- [CacheIntegrationTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/CacheIntegrationTests.swift)
- [TQDiskStoreTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/TQDiskStoreTests.swift) if TQ-native disk remains part of the design
- [JangLoaderTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Quantization/JangLoaderTests.swift)

### Phase 5. Fix Mistral Small 4 structurally

Items:
13. Rework Mistral 4 from full KV caching to latent MLA caching.
14. Only after that, reevaluate whether TurboQuant on Mistral 4 is still needed and where it should apply.

Why late:
- This is a larger architectural change and should be done after the generic cache stack is stable.

Verification:
- [MLAAttentionTests.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Models/MLAAttentionTests.swift)
- add dedicated Mistral 4 cache-shape tests near [Mistral4Model.swift](/Users/eric/osa-jang/Packages/VMLXRuntime/Sources/VMLXRuntime/Models/Mistral4Model.swift)
- run a live decode-speed sanity check after build because unit tests alone will not prove token/s improvement

### Phase 6. Fix routing and UI truthfulness

Items:
15. Make runtime routing explicit instead of failure-driven.
16. Split UI/settings so MLX-only and VMLX-only cache controls are obvious.

Why last:
- These are product-surface fixes; they should reflect the already-correct engine behavior, not paper over broken internals.

Verification:
- service/router unit coverage where available
- manual UI verification in:
  - [ConfigurationView.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Views/Settings/ConfigurationView.swift)
  - [ModelCacheInspectorView.swift](/Users/eric/osa-jang/Packages/OsaurusCore/Views/Model/ModelCacheInspectorView.swift)

## Cross-Runtime Acceptance Criteria

Every change set should be checked against all of these, not just the runtime being edited:

1. VMLX hybrid JANG models:
   Nemotron H / Cascade and Qwen 3.5 must preserve correct attention-vs-SSM cache separation.
2. VMLX non-hybrid models:
   standard transformer cache hits must not regress while hybrid logic changes.
3. VMLX Mistral 4:
   MLA-specific work must not be conflated with hybrid SSM logic.
4. MLX hybrid models:
   stable-boundary snapshot behavior must remain intact and must not be rewritten to imitate VMLX SSM companion logic.
5. MLX non-hybrid models:
   session cache, SSD cache, and prefix warm cache must still obey budget/invalidation rules.
6. Service layer:
   `ChatEngine`, `VMLXServiceBridge`, `VMLXService`, and `MLXService` must still route models deterministically after any capability/routing change.

## Current Branch Context

Already fixed during this branch before this audit:
- unload button visibility/runtime truth in the model cache inspector
- VMLX unload clearing scheduler cache state
- one Mistral 4 MLA decode inefficiency in `kPe` expansion

Those fixes do not change the audit conclusions above.
