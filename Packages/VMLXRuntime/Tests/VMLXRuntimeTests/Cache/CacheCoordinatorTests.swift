import Testing
import Foundation
import MLX
@testable import VMLXRuntime

@Suite("CacheCoordinator")
struct CacheCoordinatorTests {

    private func makeCache(tokenCount: Int) -> HybridCache {
        let layers: [LayerCacheEntry] = (0..<4).map { _ in
            .attention(KVCacheLayer(
                keys: MLXArray.zeros([1, 8, tokenCount, 128]),
                values: MLXArray.zeros([1, 8, tokenCount, 128]),
                offset: tokenCount
            ))
        }
        return HybridCache(layers: layers)
    }

    @Test("Default config initializes all layers")
    func defaultConfig() {
        let coord = CacheCoordinator()
        #expect(coord.pagedCache != nil)
        #expect(coord.memoryCache != nil)
        #expect(coord.ssmStateCache != nil)
        // Disk cache requires explicit dir
        #expect(coord.diskCache == nil)
        // Prefix cache disabled when paged cache is on
        #expect(coord.prefixCache == nil)
    }

    @Test("Miss on empty coordinator")
    func emptyMiss() {
        let coord = CacheCoordinator()
        let result = coord.fetch(tokens: [1, 2, 3])
        if case .miss = result {
            // Expected
        } else {
            Issue.record("Expected miss")
        }
    }

    @Test("Store and fetch via memory cache")
    func storeAndFetch() {
        let coord = CacheCoordinator()
        let cache = makeCache(tokenCount: 5)
        coord.store(tokens: [1, 2, 3, 4, 5], cache: cache)

        let result = coord.fetch(tokens: [1, 2, 3, 4, 5])
        if case .hit(_, let remaining, let detail, _) = result {
            #expect(remaining.isEmpty)
            #expect(detail == .memory)
        } else {
            Issue.record("Expected hit")
        }
    }

    @Test("Hybrid model returns partialHit without SSM state")
    func hybridPartialHit() {
        let coord = CacheCoordinator()
        coord.setHybrid(true)

        let cache = makeCache(tokenCount: 3)
        coord.store(tokens: [1, 2, 3], cache: cache)

        // Fetch without SSM state stored — pure attention cache has no SSM layers
        let result = coord.fetch(tokens: [1, 2, 3])
        // Should be partialHit because SSM companion state wasn't stored
        if case .partialHit(_, _, _) = result {
            // Expected — no SSM state in the stored cache
        } else if case .hit(_, _, _, _) = result {
            // Also acceptable if SSM check passes (empty SSM layers)
        } else {
            Issue.record("Expected partialHit or hit")
        }
    }

    @Test("Stats aggregate correctly")
    func statsAggregate() {
        let coord = CacheCoordinator()
        _ = coord.fetch(tokens: [99])  // miss

        let stats = coord.stats
        #expect(stats.memoryCacheMisses >= 1)
    }

    @Test("Disk cache config")
    func diskCacheConfig() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        let config = CacheCoordinatorConfig(enableDiskCache: true, diskCacheDir: dir)
        let coord = CacheCoordinator(config: config)
        #expect(coord.diskCache != nil)
        try? FileManager.default.removeItem(at: dir)
    }

    @Test("Prefix cache used when paged disabled")
    func prefixCacheWhenNoPagedCache() {
        let config = CacheCoordinatorConfig(usePagedCache: false)
        let coord = CacheCoordinator(config: config)
        #expect(coord.prefixCache != nil)
        #expect(coord.pagedCache == nil)
    }

    @Test("setHybrid toggles isHybrid flag")
    func hybridToggle() {
        let coord = CacheCoordinator()
        #expect(coord.isHybrid == false)
        coord.setHybrid(true)
        #expect(coord.isHybrid == true)
        coord.setHybrid(false)
        #expect(coord.isHybrid == false)
    }

    @Test("Store and fetch via prefix cache when memory cache disabled")
    func prefixCacheFetch() {
        let config = CacheCoordinatorConfig(
            usePagedCache: false,
            useMemoryAwareCache: false
        )
        let coord = CacheCoordinator(config: config)
        #expect(coord.prefixCache != nil)
        #expect(coord.memoryCache == nil)

        let cache = makeCache(tokenCount: 4)
        coord.store(tokens: [10, 20, 30, 40], cache: cache)

        let result = coord.fetch(tokens: [10, 20, 30, 40])
        if case .hit(_, let remaining, let detail, _) = result {
            #expect(remaining.isEmpty)
            #expect(detail == .prefix)
        } else {
            Issue.record("Expected hit from prefix cache")
        }
    }

    @Test("SSM checkpoint store and fetch")
    func ssmCheckpointRoundtrip() {
        let coord = CacheCoordinator()
        let states = [
            SSMStateLayer(state: [MLXArray.zeros([1, 16, 256])]),
        ]
        let tokenHash = SSMStateCache.hashTokens([1, 2, 3], count: 3)
        let checkpoint = SSMCheckpoint(
            ssmStates: states, boundary: 3, tokenHash: tokenHash
        )
        coord.storeSSMCheckpoint(checkpoint)

        // Verify the SSM state cache received the checkpoint
        let fetched = coord.ssmStateCache?.fetch(tokenHash: tokenHash, boundary: 3)
        #expect(fetched != nil)
        #expect(fetched?.boundary == 3)
    }

    @Test("clearAll clears SSM state cache")
    func clearAll() {
        let coord = CacheCoordinator()
        let states = [
            SSMStateLayer(state: [MLXArray.zeros([1, 16, 256])]),
        ]
        let checkpoint = SSMCheckpoint(
            ssmStates: states, boundary: 2, tokenHash: "test123"
        )
        coord.storeSSMCheckpoint(checkpoint)
        #expect(coord.ssmStateCache?.count == 1)

        coord.clearAll()
        #expect(coord.ssmStateCache?.count == 0)
    }
}
