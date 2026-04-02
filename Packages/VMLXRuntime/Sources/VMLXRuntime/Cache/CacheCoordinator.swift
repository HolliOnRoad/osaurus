import Foundation
import MLX
import os

// MARK: - Configuration

public struct CacheCoordinatorConfig: Sendable {
    public var enablePrefixCache: Bool
    public var usePagedCache: Bool
    public var useMemoryAwareCache: Bool
    public var enableDiskCache: Bool
    public var pagedBlockSize: Int
    public var maxCacheBlocks: Int
    public var cacheMemoryPercent: Float
    public var diskCacheMaxGB: Float
    public var diskCacheDir: URL?
    public var ssmMaxEntries: Int

    public init(
        enablePrefixCache: Bool = true,
        usePagedCache: Bool = true,
        useMemoryAwareCache: Bool = true,
        enableDiskCache: Bool = false,
        pagedBlockSize: Int = 64,
        maxCacheBlocks: Int = 1000,
        cacheMemoryPercent: Float = 0.30,
        diskCacheMaxGB: Float = 10.0,
        diskCacheDir: URL? = nil,
        ssmMaxEntries: Int = 50
    ) {
        self.enablePrefixCache = enablePrefixCache
        self.usePagedCache = usePagedCache
        self.useMemoryAwareCache = useMemoryAwareCache
        self.enableDiskCache = enableDiskCache
        self.pagedBlockSize = pagedBlockSize
        self.maxCacheBlocks = maxCacheBlocks
        self.cacheMemoryPercent = cacheMemoryPercent
        self.diskCacheMaxGB = diskCacheMaxGB
        self.diskCacheDir = diskCacheDir
        self.ssmMaxEntries = ssmMaxEntries
    }
}

// MARK: - Fetch Result

public enum CacheFetchResult: Sendable {
    /// Full cache hit — both attention KV and SSM state (if hybrid) available.
    /// ssmCheckpoint is non-nil for hybrid models when SSM companion was found.
    case hit(cache: HybridCache, remainingTokens: [Int], detail: CacheDetail, ssmCheckpoint: SSMCheckpoint? = nil)

    /// Partial hit — attention KV found but SSM state missing (hybrid model only).
    /// Caller must decide: sync re-derive, async re-derive, or full prefill.
    case partialHit(attentionCache: HybridCache, remainingTokens: [Int], detail: CacheDetail)

    /// Complete miss — no cached data found.
    case miss
}

// MARK: - Cache Coordinator

/// Orchestrates all cache layers into a unified fetch/store interface.
/// Fetch cascade: paged -> memory -> prefix -> disk -> MISS
/// Store cascade: paged + memory + prefix + disk + SSM companion
public final class CacheCoordinator: @unchecked Sendable {

    public let config: CacheCoordinatorConfig

    // Cache layers (initialized based on config)
    public let pagedCache: PagedCacheManager?
    public let prefixCache: PrefixCache?
    public let memoryCache: MemoryCache?
    public let diskCache: DiskCache?
    public let ssmStateCache: SSMStateCache?

    // Whether the current model is hybrid (has SSM layers)
    private var _isHybrid: Bool = false
    private let lock = OSAllocatedUnfairLock()

    public init(config: CacheCoordinatorConfig = CacheCoordinatorConfig()) {
        self.config = config

        // Initialize cache layers based on config
        if config.usePagedCache {
            self.pagedCache = PagedCacheManager(
                blockSize: config.pagedBlockSize,
                maxBlocks: config.maxCacheBlocks
            )
        } else {
            self.pagedCache = nil
        }

        // Prefix cache is only useful when paged cache is off (paged has its own hash prefix)
        if config.enablePrefixCache && !config.usePagedCache {
            self.prefixCache = PrefixCache(maxEntries: 100)
        } else {
            self.prefixCache = nil
        }

        if config.useMemoryAwareCache {
            self.memoryCache = MemoryCache(config: MemoryCacheConfig(
                maxMemoryPercent: config.cacheMemoryPercent
            ))
        } else {
            self.memoryCache = nil
        }

        if config.enableDiskCache, let dir = config.diskCacheDir {
            self.diskCache = DiskCache(cacheDir: dir, maxSizeGB: config.diskCacheMaxGB)
        } else {
            self.diskCache = nil
        }

        self.ssmStateCache = SSMStateCache(maxEntries: config.ssmMaxEntries)
    }

    /// Set whether the current model is hybrid (has SSM layers).
    /// Call this after model loading.
    public func setHybrid(_ isHybrid: Bool) {
        lock.withLock { _isHybrid = isHybrid }
    }

    public var isHybrid: Bool {
        lock.withLock { _isHybrid }
    }

    // MARK: - Fetch Cascade

    /// Fetch cached state for a token sequence.
    /// Tries each cache layer in order: paged -> memory -> prefix -> disk -> MISS.
    /// For hybrid models, also fetches SSM companion state.
    public func fetch(tokens: [Int], tokenHash: String? = nil) -> CacheFetchResult {
        let hash = tokenHash ?? SSMStateCache.hashTokens(tokens, count: tokens.count)

        // Layer 0: Paged block cache (block-level prefix matching with chain hashing)
        if let pagedCache = pagedCache {
            if let (cache, remaining) = _fetchFromPagedCache(tokens: tokens, pagedCache: pagedCache) {
                if isHybrid {
                    return _resolveHybridFetch(
                        cache: cache, remaining: remaining,
                        tokens: tokens, tokenHash: hash, detail: .paged
                    )
                }
                return .hit(cache: cache, remainingTokens: remaining, detail: .paged)
            }
        }

        // Layer 1: Memory-aware cache (RAM-budget LRU with prefix matching)
        if let memoryCache = memoryCache {
            let (cache, remaining) = memoryCache.fetch(tokens: tokens)
            if let cache = cache {
                if isHybrid {
                    return _resolveHybridFetch(
                        cache: cache, remaining: remaining,
                        tokens: tokens, tokenHash: hash, detail: .memory
                    )
                }
                return .hit(cache: cache, remainingTokens: remaining, detail: .memory)
            }
        }

        // Layer 2: Prefix cache (trie-based, when paged cache is off)
        if let prefixCache = prefixCache {
            let (cache, remaining) = prefixCache.fetch(tokens: tokens)
            if let cache = cache {
                if isHybrid {
                    return _resolveHybridFetch(
                        cache: cache, remaining: remaining,
                        tokens: tokens, tokenHash: hash, detail: .prefix
                    )
                }
                return .hit(cache: cache, remainingTokens: remaining, detail: .prefix)
            }
        }

        // Layer 3: Disk cache (L2 SSD) — load tensors from safetensors.
        // Disk cache uses exact token hash (no prefix matching).
        // Cache store uses storeTokens = cacheKeyTokens.dropLast(1), so we must
        // try both the full key and the N-1 key to find disk entries.
        if let diskCache = diskCache {
            // Try exact match first (e.g., if stored by external tool)
            if let cache = diskCache.fetchCache(tokens: tokens) {
                if let memoryCache = memoryCache {
                    _ = memoryCache.store(tokens: tokens, cache: cache)
                }
                if isHybrid {
                    return _resolveHybridFetch(
                        cache: cache, remaining: [],
                        tokens: tokens, tokenHash: hash, detail: .disk
                    )
                }
                return .hit(cache: cache, remainingTokens: [], detail: .disk)
            }

            // Try N-1 match (standard store path uses cacheKeyTokens.dropLast(1))
            if tokens.count > 1 {
                let truncatedTokens = Array(tokens.dropLast(1))
                if let cache = diskCache.fetchCache(tokens: truncatedTokens) {
                    // L2→L1 promotion: store with truncated key so memory cache
                    // prefix matching works correctly (remaining = [lastToken])
                    if let memoryCache = memoryCache {
                        _ = memoryCache.store(tokens: truncatedTokens, cache: cache)
                    }
                    let remaining = [tokens.last!]
                    if isHybrid {
                        return _resolveHybridFetch(
                            cache: cache, remaining: remaining,
                            tokens: tokens, tokenHash: hash, detail: .disk
                        )
                    }
                    return .hit(cache: cache, remainingTokens: remaining, detail: .disk)
                }
            }
        }

        return .miss
    }

    // MARK: - Paged Cache Fetch

    /// Search paged cache for matching block chain prefix.
    /// Returns (HybridCache, remainingTokens) if blocks found, nil on miss.
    private func _fetchFromPagedCache(
        tokens: [Int], pagedCache: PagedCacheManager
    ) -> (HybridCache, [Int])? {
        let blockSize = pagedCache.blockSize

        // Walk tokens in block-sized chunks, computing chain hashes.
        // Each block's hash depends on its parent hash + token content.
        var parentHash: BlockHash? = nil
        var matchedBlocks: [CacheBlock] = []
        var matchedTokens = 0

        var offset = 0
        while offset < tokens.count {
            let chunkEnd = min(offset + blockSize, tokens.count)
            let chunk = Array(tokens[offset..<chunkEnd])
            let blockHash = CacheBlock.computeBlockHash(parentHash: parentHash, tokenIds: chunk)

            if let cachedBlock = pagedCache.findCachedBlock(hash: blockHash) {
                matchedBlocks.append(cachedBlock)
                matchedTokens += cachedBlock.tokenCount
                parentHash = blockHash
                offset = chunkEnd
            } else {
                break  // Chain broken — no more prefix match
            }
        }

        guard !matchedBlocks.isEmpty else { return nil }

        // Reconstruct HybridCache by concatenating block tensor slices.
        guard let reconstructed = _reconstructFromBlocks(matchedBlocks) else { return nil }

        let remaining = Array(tokens[matchedTokens...])
        return (reconstructed, remaining)
    }

    /// Reconstruct a HybridCache from matched paged blocks.
    /// Concatenates per-layer KV slices across blocks. SSM state comes from last block only.
    private func _reconstructFromBlocks(_ blocks: [CacheBlock]) -> HybridCache? {
        guard let firstBlock = blocks.first,
              let firstData = firstBlock.cacheData else { return nil }

        let numLayers = firstData.count

        // Validate all blocks have same layer count
        for block in blocks {
            guard let data = block.cacheData, data.count == numLayers else { return nil }
        }

        var layers: [LayerCacheEntry] = []

        for layerIdx in 0..<numLayers {
            // Collect KV slices for this layer across all blocks.
            var attentionEntries: [LayerCacheEntry] = []
            var keySlices: [MLXArray] = []
            var valueSlices: [MLXArray] = []
            var lastSSM: SSMStateLayer? = nil

            for (blockIdx, block) in blocks.enumerated() {
                guard let data = block.cacheData, layerIdx < data.count else { continue }
                guard let entry = data[layerIdx] else { continue }  // nil = skip (SSM in non-last block)

                switch entry {
                case .attention(let kv):
                    attentionEntries.append(.attention(kv))
                case .compressedAttention(let ek, let ev, let offset):
                    attentionEntries.append(.compressedAttention(ek, ev, offset))
                case .ssm(let ssm):
                    // Only use SSM from the LAST block (cumulative state)
                    if blockIdx == blocks.count - 1 {
                        lastSSM = ssm
                    }
                }
            }

            if !attentionEntries.isEmpty {
                let allCompressed = attentionEntries.allSatisfy { entry in
                    if case .compressedAttention = entry { return true }
                    return false
                }
                if allCompressed,
                   let mergedCompressed = TurboQuantLayerCache.mergeCompressedAttention(attentionEntries) {
                    layers.append(mergedCompressed)
                    continue
                }

                for entry in attentionEntries {
                    switch entry {
                    case .attention(let kv):
                        keySlices.append(kv.keys)
                        valueSlices.append(kv.values)
                    case .compressedAttention(let ek, let ev, _):
                        let decodedKeys = TurboQuantEncoder.decodeKeys(ek, seed: ek.seed)
                        let decodedValues = TurboQuantEncoder.decodeValues(ev, seed: ev.seed)
                        keySlices.append(decodedKeys)
                        valueSlices.append(decodedValues)
                    case .ssm:
                        break
                    }
                }
            }

            if !keySlices.isEmpty {
                // Concatenate KV slices along sequence dimension (axis 2 for [B,H,T,D])
                let concatKeys: MLXArray
                let concatValues: MLXArray
                if keySlices.count == 1 {
                    concatKeys = keySlices[0]
                    concatValues = valueSlices[0]
                } else {
                    concatKeys = concatenated(keySlices, axis: 2)
                    concatValues = concatenated(valueSlices, axis: 2)
                }
                let totalTokens = concatKeys.dim(2)
                layers.append(.attention(KVCacheLayer(
                    keys: concatKeys, values: concatValues, offset: totalTokens)))
            } else if let ssm = lastSSM {
                layers.append(.ssm(ssm))
            } else {
                // Layer has no data across all blocks — shouldn't happen
                return nil
            }
        }

        return HybridCache(layers: layers)
    }

    // MARK: - Store Cascade

    /// Store cache state after generation.
    /// Writes to multiple layers: paged + memory + prefix + disk + SSM companion.
    public func store(tokens: [Int], cache: HybridCache, tokenHash: String? = nil) {
        // Paged block cache (block-level storage with chain hashing)
        if let pagedCache = pagedCache {
            _storeToPagedCache(tokens: tokens, cache: cache, pagedCache: pagedCache)
        }

        // Memory cache (hot tier)
        if let memoryCache = memoryCache {
            _ = memoryCache.store(tokens: tokens, cache: cache)
        }

        // Prefix cache remains useful as a secondary hot tier when paged cache is off.
        if let prefixCache = prefixCache {
            prefixCache.store(tokens: tokens, cache: cache)
        }

        // Disk cache (L2 — serialize tensors to safetensors)
        if let diskCache = diskCache {
            diskCache.storeCache(tokens: tokens, cache: cache)
        }

        // SSM companion (for hybrid models)
        if isHybrid, let ssmCache = ssmStateCache {
            let ssmLayers = cache.ssmLayers
            if !ssmLayers.isEmpty {
                ssmCache.store(
                    ssmStates: ssmLayers,
                    tokens: tokens,
                    boundary: tokens.count
                )
            }
        }
    }

    // MARK: - Paged Cache Store

    /// Split cache into block-sized chunks and store with chain-hashed keys.
    /// Each block stores per-layer KV slices for its token range.
    /// Last block stores cumulative SSM state for hybrid models.
    /// Deduplicates blocks by chain hash (increments ref count on match).
    private func _storeToPagedCache(
        tokens: [Int], cache: HybridCache, pagedCache: PagedCacheManager
    ) {
        let blockSize = pagedCache.blockSize
        guard !tokens.isEmpty, !cache.layers.isEmpty else { return }

        let numBlocks = (tokens.count + blockSize - 1) / blockSize

        var parentHash: BlockHash? = nil

        for blockIdx in 0..<numBlocks {
            let tokenStart = blockIdx * blockSize
            let tokenEnd = min(tokenStart + blockSize, tokens.count)
            let blockTokens = Array(tokens[tokenStart..<tokenEnd])
            let isLastBlock = (blockIdx == numBlocks - 1)

            // Compute chain hash
            let blockHash = CacheBlock.computeBlockHash(
                parentHash: parentHash, tokenIds: blockTokens)

            // Check dedup: does a block with this hash already exist?
            if let existing = pagedCache.findCachedBlock(hash: blockHash) {
                if isLastBlock && isHybrid && _blockLacksCumulativeSSM(existing) {
                    // Fall through to allocate new block with SSM data
                } else {
                    pagedCache.forkBlock(existing, hash: blockHash)
                    parentHash = blockHash
                    continue
                }
            }

            guard let block = pagedCache.allocateBlock() else {
                break
            }

            block.tokenCount = blockTokens.count
            block.blockHash = blockHash

            // Slice layer data for this block's token range.
            var blockData: [LayerCacheEntry?] = []
            for entry in cache.layers {
                switch entry {
                case .attention(let kv):
                    let slicedKeys = kv.keys[.ellipsis, tokenStart..<tokenEnd, 0...]
                    let slicedValues = kv.values[.ellipsis, tokenStart..<tokenEnd, 0...]
                    let sliceOffset = tokenEnd - tokenStart
                    blockData.append(.attention(KVCacheLayer(
                        keys: slicedKeys, values: slicedValues, offset: sliceOffset)))

                case .ssm(let ssm):
                    if isLastBlock {
                        blockData.append(.ssm(ssm))
                    } else {
                        blockData.append(nil)
                    }

                case .compressedAttention(let ek, let ev, _):
                    if let slicedCompressed = TurboQuantLayerCache.sliceCompressedAttention(
                        ek,
                        ev,
                        range: tokenStart..<tokenEnd
                    ) {
                        blockData.append(slicedCompressed)
                    } else {
                        // Sink-only edge case: materialize just this slice to float.
                        let decodedKeys = TurboQuantEncoder.decodeKeys(ek, seed: ek.seed)
                        let decodedValues = TurboQuantEncoder.decodeValues(ev, seed: ev.seed)
                        let slicedKeys = decodedKeys[.ellipsis, tokenStart..<tokenEnd, 0...]
                        let slicedValues = decodedValues[.ellipsis, tokenStart..<tokenEnd, 0...]
                        blockData.append(.attention(KVCacheLayer(
                            keys: slicedKeys,
                            values: slicedValues,
                            offset: tokenEnd - tokenStart
                        )))
                    }
                }
            }

            block.cacheData = blockData

            pagedCache.markCached(block: block, hash: blockHash)

            parentHash = blockHash
        }
    }

    /// Check if a block lacks cumulative SSM data (Bug 2 prevention).
    /// Returns true if the block has SSM layer positions but they're all nil (skip).
    private func _blockLacksCumulativeSSM(_ block: CacheBlock) -> Bool {
        guard let data = block.cacheData else { return true }
        // Check if any entry is .ssm — if yes, it has cumulative data
        for entry in data {
            if let entry = entry, case .ssm = entry { return false }
        }
        // No .ssm entries found — either all attention (fine) or SSM positions are nil (lacks data)
        return true
    }

    /// Store SSM checkpoint at a stable boundary (for thinking models).
    /// Call this during prefill Phase 1, before gen_prompt_len tokens.
    public func storeSSMCheckpoint(_ checkpoint: SSMCheckpoint) {
        ssmStateCache?.store(checkpoint: checkpoint)
    }

    // MARK: - Cache Warming

    /// Pre-warm cache with a list of prompts.
    /// Useful for pre-populating system prompts that will be reused.
    public func warmCache(tokenSequences: [[Int]], caches: [HybridCache]) {
        for (tokens, cache) in zip(tokenSequences, caches) {
            store(tokens: tokens, cache: cache)
        }
    }

    /// Returns the total number of entries across all active cache layers.
    public func warmCacheCount() -> Int {
        var count = 0
        if let mc = memoryCache { count += mc.count }
        if let pc = prefixCache { count += pc.count }
        if let dc = diskCache { count += dc.entryCount }
        return count
    }

    // MARK: - Cache Management

    /// Clear volatile caches while preserving persistent L2 entries.
    /// Used for request-scoped recovery without destroying the disk cache.
    public func clearVolatile() {
        ssmStateCache?.clear()
        pagedCache?.clear()
        prefixCache?.clear()
        memoryCache?.clear()
    }

    /// Invalidate request-related cache keys without wiping the full cache stack.
    /// Includes both the exact token key and the standard N-1 store key.
    public func invalidate(tokens: [Int]) {
        let candidateKeys: [[Int]] = {
            guard tokens.count > 1 else { return [tokens] }
            let truncated = Array(tokens.dropLast(1))
            return truncated == tokens ? [tokens] : [tokens, truncated]
        }()

        for key in candidateKeys {
            memoryCache?.invalidate(tokens: key)
            prefixCache?.invalidate(tokens: key)
            diskCache?.remove(tokens: key)

            let tokenHash = SSMStateCache.hashTokens(key, count: key.count)
            ssmStateCache?.invalidate(tokenHash: tokenHash, boundary: key.count)
        }

        // Paged cache does not support exact-key invalidation today.
        // Clear only the volatile paged layer instead of nuking L2.
        pagedCache?.clear()
    }

    /// Clear all caches. Called when switching models to prevent stale KV data.
    public func clearAll() {
        ssmStateCache?.clear()
        pagedCache?.clear()
        prefixCache?.clear()
        memoryCache?.clear()
        diskCache?.clear()
    }

    /// Get aggregate stats across all cache layers.
    public var stats: CacheCoordinatorStats {
        CacheCoordinatorStats(
            memoryCacheHits: memoryCache?.hits ?? 0,
            memoryCacheMisses: memoryCache?.misses ?? 0,
            prefixCacheHits: prefixCache?.hits ?? 0,
            prefixCacheMisses: prefixCache?.misses ?? 0,
            diskCacheHits: diskCache?.hits ?? 0,
            diskCacheMisses: diskCache?.misses ?? 0,
            ssmCacheHits: ssmStateCache?.hits ?? 0,
            ssmCacheMisses: ssmStateCache?.misses ?? 0,
            pagedCacheStats: pagedCache?.stats
        )
    }

    // MARK: - Private

    /// For hybrid models, check if we also have SSM companion state.
    /// If SSM state is found at the matching boundary, return a full hit.
    /// Otherwise return a partial hit (attention KV found but SSM missing).
    private func _resolveHybridFetch(
        cache: HybridCache, remaining: [Int],
        tokens: [Int], tokenHash: String, detail: CacheDetail
    ) -> CacheFetchResult {
        guard let ssmCache = ssmStateCache else {
            return .partialHit(
                attentionCache: cache,
                remainingTokens: remaining,
                detail: detail
            )
        }

        // How many tokens the cache covers
        let boundary = tokens.count - remaining.count
        let boundaryHash = SSMStateCache.hashTokens(tokens, count: boundary)

        if let checkpoint = ssmCache.fetch(tokenHash: boundaryHash, boundary: boundary) {
            // Full hit: have both attention KV and SSM state.
            // Pass checkpoint to generation engine for merging into cache objects.
            return .hit(cache: cache, remainingTokens: remaining, detail: detail, ssmCheckpoint: checkpoint)
        }

        // Partial hit: attention KV found but SSM state is missing
        return .partialHit(
            attentionCache: cache,
            remainingTokens: remaining,
            detail: detail
        )
    }
}

// MARK: - Aggregate Stats

public struct CacheCoordinatorStats: Sendable {
    public let memoryCacheHits: Int
    public let memoryCacheMisses: Int
    public let prefixCacheHits: Int
    public let prefixCacheMisses: Int
    public let diskCacheHits: Int
    public let diskCacheMisses: Int
    public let ssmCacheHits: Int
    public let ssmCacheMisses: Int
    public let pagedCacheStats: CacheStats?
}
