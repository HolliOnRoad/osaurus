# Phase 1: Cache Key & Store Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the three foundational cache bugs that prevent any cache hit from working correctly on multi-turn conversations: wrong cache key, wrong store contents, and partial hit mishandling.

**Architecture:** The generation loop in VMLXRuntimeActor needs three changes: (1) compute gen_prompt_len and strip it from the cache key, (2) store cache truncated to prompt_len-1 using the stripped key (not prompt+generated), (3) handle partial hits for hybrid models by checking SSM companion and falling back to full prefill when missing.

**Tech Stack:** Swift, MLX, VMLXRuntime

---

### Task 1: Add gen_prompt_len to generation loop and strip cache key

**Files:**
- Modify: `Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift:359-396`

**What changes:**
After tokenization (line 364-371), compute gen_prompt_len using the already-existing `container.computeGenPromptLen()`. Create a `cacheKeyTokens` array by stripping gen_prompt_len tokens from the end. Use `cacheKeyTokens` for cache fetch instead of `tokens`.

**Step 1: Add gen_prompt_len computation after tokenization**

Find this block (lines 362-396):
```swift
        let tokens: [Int]
        do {
            tokens = try container.applyChatTemplate(
                messages: request.messages,
                addGenerationPrompt: true,
                enableThinking: enableThinking
            )
        } catch {
            throw VMLXRuntimeError.tokenizationFailed
        }

        let promptTokenCount = tokens.count
```

Replace with:
```swift
        let tokens: [Int]
        do {
            tokens = try container.applyChatTemplate(
                messages: request.messages,
                addGenerationPrompt: true,
                enableThinking: enableThinking
            )
        } catch {
            throw VMLXRuntimeError.tokenizationFailed
        }

        // Compute gen_prompt_len: number of assistant header tokens appended by chat template.
        // Strip these from cache key so multi-turn conversations hit the same prefix.
        // e.g., "<|im_start|>assistant\n" or "<think>\n" — these change per turn.
        let genPromptLen = container.computeGenPromptLen(messages: request.messages)
        let cacheKeyTokens: [Int]
        if genPromptLen > 0 && genPromptLen < tokens.count {
            cacheKeyTokens = Array(tokens.dropLast(genPromptLen))
        } else {
            cacheKeyTokens = tokens
        }

        let promptTokenCount = tokens.count
```

**Step 2: Use cacheKeyTokens for cache fetch**

Find (line 396):
```swift
                    let fetchResult = self.scheduler.cache.fetch(tokens: tokens)
```

Replace with:
```swift
                    let fetchResult = self.scheduler.cache.fetch(tokens: cacheKeyTokens)
```

**Step 3: Build and verify**

Run: `cd Packages/VMLXRuntime && swift build`
Expected: Build succeeds with no errors.

**Step 4: Commit**

```bash
git add Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift
git commit -m "feat(cache): strip gen_prompt_len from cache key for multi-turn hits"
```

---

### Task 2: Fix cache store — truncate to prompt_len-1, use stripped key

**Files:**
- Modify: `Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift:583-604`

**What changes:**
Currently stores `tokens + accumulator.generatedTokenIds` as the cache key. This means:
1. Generated tokens are in the key — never matches on next turn (different generation)
2. gen_prompt_len tokens are in the key — changes per turn
3. Full prompt+generated KV is stored — wastes memory

Fix: Store only `cacheKeyTokens` (stripped), truncated to `len-1`. Trim the KV cache to match.

**Step 1: Replace the cache store block**

Find this block (lines 583-604):
```swift
                    // Store cache for future turn reuse
                    NSLog("[Gen] Storing cache: \(tokens.count) prompt + \(accumulator.generatedTokenIds.count) generated = \(tokens.count + accumulator.generatedTokenIds.count) total tokens")
                    let allTokens = tokens + accumulator.generatedTokenIds
                    if !allTokens.isEmpty {
                        var layers: [LayerCacheEntry] = []
                        for c in cache {
                            if let mc = c as? VMLXMambaCache {
                                layers.append(.ssm(SSMStateLayer(state: mc.state)))
                            } else if let kvc = c as? VMLXKVCacheSimple {
                                let s = kvc.state
                                if s.count == 2 {
                                    layers.append(.attention(KVCacheLayer(
                                        keys: s[0], values: s[1], offset: kvc.offset)))
                                }
                            }
                        }
                        if !layers.isEmpty {
                            let hybridCache = HybridCache(layers: layers)
                            hybridCache.materialized()
                            self.scheduler.cache.store(tokens: allTokens, cache: hybridCache)
                        }
                    }
```

Replace with:
```swift
                    // Store cache for future turn reuse.
                    // Key: cacheKeyTokens (stripped of gen_prompt_len), truncated to len-1.
                    // Why len-1: on cache hit, we re-feed the last token to get fresh logits.
                    // This matches Python VMLX's _truncate_cache_to_prompt_length().
                    let storeTokens: [Int]
                    if cacheKeyTokens.count > 1 {
                        storeTokens = Array(cacheKeyTokens.dropLast(1))
                    } else {
                        storeTokens = cacheKeyTokens
                    }

                    if !storeTokens.isEmpty {
                        // Trim KV cache to match storeTokens length.
                        // Current cache has: prompt tokens + generated tokens.
                        // We want: prompt tokens - gen_prompt_len - 1.
                        let trimCount = generatedTokenCount + genPromptLen + 1
                        for c in cache {
                            if let kvc = c as? VMLXKVCacheSimple {
                                kvc.trim(trimCount)
                            }
                            // Note: SSM (Mamba) state is cumulative, not positional.
                            // We store the SSM state as-is (it represents state after
                            // processing all prompt tokens including gen_prompt_len).
                            // This is correct because SSM state at boundary N contains
                            // information from tokens 0..N. On cache hit, the SSM state
                            // for the stripped prefix is what we want.
                        }

                        var layers: [LayerCacheEntry] = []
                        for c in cache {
                            if let mc = c as? VMLXMambaCache {
                                layers.append(.ssm(SSMStateLayer(state: mc.state)))
                            } else if let kvc = c as? VMLXKVCacheSimple {
                                let s = kvc.state
                                if s.count == 2 {
                                    layers.append(.attention(KVCacheLayer(
                                        keys: s[0], values: s[1], offset: kvc.offset)))
                                }
                            }
                        }
                        if !layers.isEmpty {
                            let hybridCache = HybridCache(layers: layers)
                            hybridCache.materialized()
                            self.scheduler.cache.store(tokens: storeTokens, cache: hybridCache)
                            NSLog("[Gen] Stored cache: \(storeTokens.count) tokens (stripped \(genPromptLen) gen_prompt + \(generatedTokenCount) generated + 1 last)")
                        }
                    }
```

**Step 2: Build and verify**

Run: `cd Packages/VMLXRuntime && swift build`
Expected: Build succeeds.

**Step 3: Commit**

```bash
git add Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift
git commit -m "fix(cache): store truncated to prompt_len-1 with stripped key"
```

---

### Task 3: Handle partial cache hits for hybrid models

**Files:**
- Modify: `Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift:430-434`

**What changes:**
Currently `.partialHit` is grouped with `.miss` — always does full prefill. For hybrid models, a partial hit means attention KV is cached but SSM companion is missing. The Python engine discards the KV cache in this case (SSM state is path-dependent, can't skip SSM layers). For non-hybrid models, there's no SSM so partial hit = regular prefix hit.

**Step 1: Replace the partial hit handling**

Find this block (lines 430-434):
```swift
                    case .partialHit(_, _, _), .miss, .hit(_, _, _):
                        // No usable cache hit — prefill all tokens
                        NSLog("[Gen] Cache MISS: prefilling \(tokens.count) tokens")
                        inputTokens = MLXArray(tokens.map { Int32($0) })
```

Replace with:
```swift
                    case .partialHit(let attentionCache, let remaining, let detail):
                        if container.isHybrid {
                            // Hybrid model: SSM companion missing. SSM state is path-dependent —
                            // we can't use KV cache without matching SSM state.
                            // Discard the attention cache and do full prefill.
                            // (Matches Python VMLX: "no SSM companion, full prefill")
                            NSLog("[Gen] Cache PARTIAL HIT (hybrid, SSM missing): discarding KV, full prefill \(tokens.count) tokens")
                            inputTokens = MLXArray(tokens.map { Int32($0) })
                        } else {
                            // Non-hybrid: no SSM layers, partial hit = prefix hit.
                            // Use attention KV cache, prefill only remaining tokens.
                            NSLog("[Gen] Cache PARTIAL HIT (non-hybrid): \(attentionCache.layerCount) layers, \(remaining.count) remaining, detail=\(detail)")
                            for (i, entry) in attentionCache.layers.enumerated() {
                                guard i < cache.count else { break }
                                if case .attention(let kv) = entry {
                                    if let kvSimple = cache[i] as? VMLXKVCacheSimple {
                                        kvSimple.state = [kv.keys, kv.values]
                                    }
                                }
                            }
                            cachedTokenCount = tokens.count - remaining.count
                            if remaining.isEmpty {
                                for c in cache {
                                    if let kvc = c as? VMLXKVCacheSimple {
                                        kvc.trim(1)
                                    }
                                }
                                cachedTokenCount -= 1
                                inputTokens = MLXArray([Int32(tokens.last!)])
                            } else {
                                inputTokens = MLXArray(remaining.map { Int32($0) })
                            }
                        }

                    case .miss, .hit(_, _, _):
                        // Complete miss or layer count mismatch — full prefill
                        NSLog("[Gen] Cache MISS: prefilling \(tokens.count) tokens")
                        inputTokens = MLXArray(tokens.map { Int32($0) })
```

**Step 2: Build and verify**

Run: `cd Packages/VMLXRuntime && swift build`
Expected: Build succeeds.

**Step 3: Commit**

```bash
git add Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift
git commit -m "fix(cache): handle partial hits — use KV for non-hybrid, discard for hybrid"
```

---

### Task 4: Fix full cache hit to use cacheKeyTokens for remaining computation

**Files:**
- Modify: `Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift:398-429`

**What changes:**
The cache hit block computes `cachedTokenCount = tokens.count - remaining.count` but the fetch was done with `cacheKeyTokens` (which is shorter than `tokens` by `genPromptLen`). The remaining tokens are relative to `cacheKeyTokens`, not `tokens`. When there's no remaining (full cache hit), we need to feed ALL of `tokens[cacheKeyTokens.count...]` (the gen_prompt_len tokens) plus re-feed the last cached token.

Actually, on closer look: the fetch is done with `cacheKeyTokens`. So `remaining` contains the tokens from `cacheKeyTokens` that weren't cached. But we need to prefill with `remaining + genPromptLen tokens` because the actual prompt includes the gen_prompt_len suffix.

Wait — let me think about this more carefully.

`cacheKeyTokens` = tokens stripped of gen_prompt_len = tokens[0..<tokens.count - genPromptLen]
We store cache for `cacheKeyTokens.dropLast(1)` = tokens[0..<tokens.count - genPromptLen - 1]
On fetch, we search with `cacheKeyTokens` = tokens[0..<tokens.count - genPromptLen]

If cache hit with remaining=[]: all of cacheKeyTokens matched. But the stored cache only has prompt_len-1 tokens. So remaining should be `[cacheKeyTokens.last!]` (the last stripped token). Plus we need to process the gen_prompt_len tokens after that.

So the input to prefill should be: `[lastCacheKeyToken] + genPromptLenTokens`

**Step 1: Fix the full hit and prefix hit input computation**

Find the `.hit` case (lines 398-429). Replace the input token computation:

```swift
                    case .hit(let cachedHybrid, let remaining, let detail)
                        where cachedHybrid.layerCount == cache.count:
                        NSLog("[Gen] Cache HIT: \(cachedHybrid.layerCount) layers, \(remaining.count) remaining tokens, detail=\(detail)")
                        // Restore cached KV state into the VMLXKVCache objects
                        for (i, entry) in cachedHybrid.layers.enumerated() {
                            guard i < cache.count else { break }
                            switch entry {
                            case .attention(let kv):
                                if let kvSimple = cache[i] as? VMLXKVCacheSimple {
                                    kvSimple.state = [kv.keys, kv.values]
                                }
                            case .ssm(let ssm):
                                if let mambaCache = cache[i] as? VMLXMambaCache {
                                    mambaCache.state = ssm.state
                                }
                            }
                        }

                        // Compute actual input tokens to prefill.
                        // Cache was stored with cacheKeyTokens.dropLast(1).
                        // remaining = cacheKeyTokens tokens not in cache.
                        // We also need to process the gen_prompt_len suffix.
                        let genSuffix = genPromptLen > 0
                            ? Array(tokens.suffix(genPromptLen))
                            : [Int]()

                        if remaining.isEmpty {
                            // Full cache hit on cacheKeyTokens. Re-feed last cache key token
                            // to get fresh logits, plus gen_prompt_len suffix.
                            cachedTokenCount = cacheKeyTokens.count - 1
                            let refeedTokens = [cacheKeyTokens.last!] + genSuffix
                            inputTokens = MLXArray(refeedTokens.map { Int32($0) })
                        } else {
                            // Prefix hit: cache covers some of cacheKeyTokens.
                            cachedTokenCount = cacheKeyTokens.count - remaining.count
                            let allRemaining = remaining + genSuffix
                            inputTokens = MLXArray(allRemaining.map { Int32($0) })
                        }
```

**Step 2: Build and verify**

Run: `cd Packages/VMLXRuntime && swift build`
Expected: Build succeeds.

**Step 3: Commit**

```bash
git add Sources/VMLXRuntime/Integration/VMLXRuntimeActor.swift
git commit -m "fix(cache): correctly compute remaining tokens including gen_prompt_len suffix"
```

---

### Task 5: Verify end-to-end with a build

**Step 1: Full SPM build**

Run: `cd Packages/VMLXRuntime && swift build`
Expected: Build succeeds with no errors.

**Step 2: Full Xcode build**

Run: `cd /Users/eric/osa-jang && xcodebuild -scheme osaurus -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 3: Commit all if not already committed**

---

## How to Test (Manual)

After building, load any model (e.g., Qwen2.5-3B-4bit) and:

1. **Turn 1:** Send "Hello, what is 2+2?" — observe: Cache MISS, full prefill, stores cache
2. **Turn 2:** Send "And what is 3+3?" — observe: Cache HIT (or prefix hit), skips most prefill
3. **Verify in /tmp/vmlx_debug.log:** Look for "Cache HIT" on turn 2 with cached token count > 0

For hybrid models (Qwen3.5), verify that partial hit correctly falls back to full prefill when SSM companion is missing.

---

## What This Fixes

| Before | After |
|--------|-------|
| Cache key includes generated tokens — never hits on turn 2 | Cache key = prompt tokens stripped of gen_prompt_len |
| Cache stores prompt + generated — wastes memory, wrong key | Cache stores prompt_len-1 tokens with stripped key |
| Partial hit treated as miss | Partial hit: use KV for non-hybrid, discard for hybrid |
| Full hit doesn't account for gen_prompt_len | Full hit feeds remaining + gen_prompt_len suffix |

## What This Does NOT Fix (Phase 2+)

- SSM companion not extracted after prefill (Phase 2)
- Paged block cache not wired (Phase 3)
- Disk cache L2 gaps (Phase 4)
- TurboQuant not wired (Phase 5)
- SSM async re-derivation (Phase 6)
