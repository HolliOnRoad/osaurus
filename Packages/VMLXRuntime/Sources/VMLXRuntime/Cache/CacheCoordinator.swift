import Foundation
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
    case hit(cache: HybridCache, remainingTokens: [Int], detail: CacheDetail)

    /// Partial hit — attention KV found but SSM state missing (hybrid model only).
    /// Caller must decide: sync re-derive, async re-derive, or full prefill.
    case partialHit(attentionCache: HybridCache, remainingTokens: [Int], detail: CacheDetail)

    /// Complete miss — no cached data found.
    case miss
}

// MARK: - Cache Coordinator

/// Orchestrates all cache layers into a unified fetch/store interface.
/// Fetch cascade: memory -> prefix -> disk -> MISS
/// Store cascade: memory + disk + paged hashes + SSM companion
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
    /// Tries each cache layer in order: memory -> prefix -> disk -> MISS.
    /// For hybrid models, also fetches SSM companion state.
    public func fetch(tokens: [Int], tokenHash: String? = nil) -> CacheFetchResult {
        let hash = tokenHash ?? SSMStateCache.hashTokens(tokens, count: tokens.count)

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

        // Layer 3: Disk cache (L2 SSD) — metadata lookup only for now
        if let diskCache = diskCache {
            let result = diskCache.fetch(tokens: tokens)
            if result != nil {
                // TODO: Load actual tensor data from safetensors file
                // For now, record the hit but can't return cache data
                return .miss  // Will be .hit once tensor I/O is implemented
            }
        }

        return .miss
    }

    // MARK: - Store Cascade

    /// Store cache state after generation.
    /// Writes to multiple layers: memory + prefix + disk + SSM companion.
    public func store(tokens: [Int], cache: HybridCache, tokenHash: String? = nil) {
        let hash = tokenHash ?? SSMStateCache.hashTokens(tokens, count: tokens.count)

        // Memory cache (hot tier)
        if let memoryCache = memoryCache {
            _ = memoryCache.store(tokens: tokens, cache: cache)
        }

        // Prefix cache (used only when memory cache is off)
        if memoryCache == nil, let prefixCache = prefixCache {
            prefixCache.store(tokens: tokens, cache: cache)
        }

        // Disk cache (L2 — metadata only for now)
        if let diskCache = diskCache {
            _ = diskCache.store(
                tokens: tokens,
                numTokens: tokens.count,
                fileSize: cache.estimatedBytes
            )
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

    /// Store SSM checkpoint at a stable boundary (for thinking models).
    /// Call this during prefill Phase 1, before gen_prompt_len tokens.
    public func storeSSMCheckpoint(_ checkpoint: SSMCheckpoint) {
        ssmStateCache?.store(checkpoint: checkpoint)
    }

    // MARK: - Cache Management

    /// Clear all caches.
    public func clearAll() {
        ssmStateCache?.clear()
        // Note: pagedCache, prefixCache, memoryCache, diskCache don't have
        // bulk clear methods yet. Individual entries are managed via LRU eviction.
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

        if ssmCache.fetch(tokenHash: boundaryHash, boundary: boundary) != nil {
            // Full hit: have both attention KV and SSM state
            // SSM checkpoint merge into HybridCache happens in the generation engine
            return .hit(cache: cache, remainingTokens: remaining, detail: detail)
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
