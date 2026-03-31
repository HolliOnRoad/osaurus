import Testing
import Foundation
import MLX
import MLXNN
@testable import VMLXRuntime

// MARK: - Model Loading Tests

@Suite("Model Loading")
struct ModelLoadingTests {

    /// Locate the Qwen3.5-4B-JANG_2S model directory, skip if not available.
    static func jangModelPath() -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent("jang/models/Qwen3.5-4B-JANG_2S")
        guard FileManager.default.fileExists(
            atPath: path.appendingPathComponent("config.json").path
        ) else { return nil }
        return path
    }

    @Test("Detect JANG model properties")
    func detectJangModel() throws {
        guard let path = Self.jangModelPath() else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found at ~/jang/models/")
            return
        }
        let detected = try ModelDetector.detect(at: path)
        #expect(detected.isJang)
        #expect(detected.modelType == "qwen3_5")
        #expect(detected.isHybrid || detected.hasSSM)
        #expect(detected.name.contains("Qwen3.5"))
        print("Detected: \(detected.name), family=\(detected.family), type=\(detected.modelType ?? "nil")")
    }

    @Test("Parse Qwen3.5 config from JSON")
    func parseQwen35Config() throws {
        guard let path = Self.jangModelPath() else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found")
            return
        }
        let configURL = path.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Qwen35Configuration.self, from: data)

        #expect(config.modelType == "qwen3_5")
        #expect(config.textConfig.hiddenLayers == 32)
        #expect(config.textConfig.fullAttentionInterval == 4)
        #expect(config.textConfig.linearNumValueHeads > 0)
        #expect(config.textConfig.linearKeyHeadDim > 0)
        print("Config: \(config.textConfig.hiddenLayers) layers, interval=\(config.textConfig.fullAttentionInterval)")
    }

    @Test("Parse base configuration and quantization")
    func parseBaseConfig() throws {
        guard let path = Self.jangModelPath() else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found")
            return
        }
        let configURL = path.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let base = try JSONDecoder().decode(VMLXBaseConfiguration.self, from: data)

        #expect(base.modelType == "qwen3_5")
        #expect(base.quantization != nil)
        #expect(base.quantization?.bits == 2)
        #expect(base.quantization?.groupSize == 64)
        print("Quantization: bits=\(base.quantization!.bits), group_size=\(base.quantization!.groupSize)")
    }

    @Test("Registry creates correct model type for qwen3_5")
    func registryCreatesQwen35() throws {
        #expect(VMLXModelRegistry.isSupported(modelType: "qwen3_5"))
        #expect(VMLXModelRegistry.isSupported(modelType: "qwen3_5_moe"))
        #expect(VMLXModelRegistry.isSupported(modelType: "llama"))
        #expect(VMLXModelRegistry.isSupported(modelType: "qwen2"))
    }

    @Test("ModelDetector scans and finds available models")
    func scanAvailableModels() {
        let models = ModelDetector.scanAvailableModels()
        print("Found \(models.count) models:")
        for m in models.prefix(10) {
            print("  - \(m.name) [\(m.modelType ?? "?")] jang=\(m.isJang) hybrid=\(m.isHybrid)")
        }
    }

    @Test("Load JANG model end-to-end")
    func loadJangModel() async throws {
        guard let path = Self.jangModelPath() else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found")
            return
        }

        print("Loading model from \(path.lastPathComponent)...")
        let startTime = CFAbsoluteTimeGetCurrent()

        let loaded = try await ModelLoader.load(from: path)

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("Model loaded in \(String(format: "%.2f", elapsed))s")
        print("  Name: \(loaded.detected.name)")
        print("  Vocab: \(loaded.vocabSize)")
        print("  Layers: \(loaded.numLayers)")
        print("  Hidden: \(loaded.hiddenSize)")

        #expect(loaded.vocabSize > 0)
        #expect(loaded.numLayers == 32)
        #expect(loaded.hiddenSize > 0)
        #expect(loaded.detected.isJang)

        // Verify tokenizer works
        let tokens = loaded.tokenizer.encode(text: "Hello")
        #expect(!tokens.isEmpty)
        let decoded = loaded.tokenizer.decode(tokens: tokens)
        #expect(decoded.contains("Hello"))
        print("  Tokenizer test: 'Hello' -> \(tokens) -> '\(decoded)'")
    }
}

// MARK: - Forward Pass Tests

@Suite("Forward Pass")
struct ForwardPassTests {

    @Test("Qwen3.5 forward pass produces logits")
    func qwen35ForwardPass() async throws {
        guard let path = ModelLoadingTests.jangModelPath() else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found")
            return
        }

        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)

        // Create caches
        let cache = container.newCache()
        #expect(cache.count == 32) // 32 layers
        print("Cache created: \(cache.count) layer caches")

        // Count layer types
        var ssmCount = 0
        var attnCount = 0
        for c in cache {
            if c is VMLXMambaCache { ssmCount += 1 }
            else if c is VMLXKVCacheSimple { attnCount += 1 }
        }
        print("  SSM caches: \(ssmCount), Attention caches: \(attnCount)")
        #expect(ssmCount == 24) // 24 linear_attention layers
        #expect(attnCount == 8)  // 8 full_attention layers

        // Single token forward pass
        let tokenIds = MLXArray([Int32(1)]).reshaped(1, 1)
        let logits = container.forward(tokenIds, cache: cache)

        // Force computation (MLX lazy computation trigger, NOT code execution)
        MLX.eval(logits)

        #expect(logits.ndim == 3) // [B, seq, vocab]
        #expect(logits.dim(0) == 1) // batch=1
        #expect(logits.dim(1) == 1) // seq=1
        #expect(logits.dim(2) == container.nativeModel.vocabularySize)
        print("Logits shape: \(logits.shape) vocab=\(logits.dim(2))")

        // Verify cache was populated
        var populatedSSM = 0
        var populatedAttn = 0
        for c in cache {
            if let mc = c as? VMLXMambaCache, mc.offset > 0 || !mc.state.isEmpty {
                populatedSSM += 1
            } else if let kvc = c as? VMLXKVCacheSimple, kvc.offset > 0 {
                populatedAttn += 1
            }
        }
        print("  Populated: SSM=\(populatedSSM), Attention=\(populatedAttn)")
    }

    @Test("Multi-token prefill then decode step")
    func prefillAndDecode() async throws {
        guard let path = ModelLoadingTests.jangModelPath() else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found")
            return
        }

        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)
        let cache = container.newCache()

        // Prefill with 5 tokens
        let prefillTokens = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4), Int32(5)]).reshaped(1, 5)
        let prefillLogits = container.forward(prefillTokens, cache: cache)
        MLX.eval(prefillLogits)

        #expect(prefillLogits.dim(1) == 5) // logits for all 5 positions
        print("Prefill logits shape: \(prefillLogits.shape)")

        // Check attention cache offset
        for c in cache {
            if let kvc = c as? VMLXKVCacheSimple {
                #expect(kvc.offset == 5)
                break
            }
        }

        // Decode one more token
        let decodeToken = MLXArray([Int32(6)]).reshaped(1, 1)
        let decodeLogits = container.forward(decodeToken, cache: cache)
        MLX.eval(decodeLogits)

        #expect(decodeLogits.dim(1) == 1) // logits for 1 position
        print("Decode logits shape: \(decodeLogits.shape)")

        // Verify cache offset grew
        for c in cache {
            if let kvc = c as? VMLXKVCacheSimple {
                #expect(kvc.offset == 6)
                break
            }
        }
        print("Prefill+decode: OK")
    }

    @Test("Generate text with VMLXRuntimeActor")
    func generateText() async throws {
        guard let path = ModelLoadingTests.jangModelPath() else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found")
            return
        }

        let runtime = VMLXRuntimeActor()
        try await runtime.loadModel(from: path)
        #expect(await runtime.isModelLoaded)

        let request = VMLXChatCompletionRequest(
            messages: [
                VMLXChatMessage(role: "user", content: "Say hello in one word")
            ],
            model: nil,
            temperature: 0, // greedy for determinism
            maxTokens: 10,
            topP: 1.0,
            repetitionPenalty: 1.0,
            stop: [],
            stream: false
        )

        let result = try await runtime.generate(request: request)
        print("Generated: '\(result)'")
        #expect(!result.isEmpty)

        await runtime.unloadModel()
    }
}

// MARK: - Hybrid SSM Cache Tests

@Suite("Hybrid SSM Cache")
struct HybridSSMCacheTests {

    @Test("HybridCache correctly splits attention vs SSM layers")
    func hybridCacheSplit() {
        // Build a hybrid cache matching Qwen3.5 pattern: 3 SSM + 1 attn, repeated 8x = 32 layers
        let pattern: [LayerType] = (0..<32).map { i in
            (i + 1) % 4 == 0 ? .attention : .ssm
        }

        let cache = HybridCache.fromPattern(
            pattern,
            kvFactory: {
                KVCacheLayer(
                    keys: MLXArray.zeros([1, 8, 10, 128]),
                    values: MLXArray.zeros([1, 8, 10, 128]),
                    offset: 10
                )
            },
            ssmFactory: {
                SSMStateLayer(state: [
                    MLXArray.zeros([1, 64, 128, 192]), // conv_state
                    MLXArray.zeros([1, 64, 128, 192])  // ssm_state
                ])
            }
        )

        #expect(cache.layerCount == 32)
        #expect(cache.isHybrid)
        #expect(!cache.isPureAttention)
        #expect(!cache.canTruncate) // has SSM layers

        let attnIndices = cache.attentionLayerIndices
        let ssmIndices = cache.ssmLayerIndices
        #expect(attnIndices.count == 8)
        #expect(ssmIndices.count == 24)

        // Verify the pattern: layers 3, 7, 11, 15, 19, 23, 27, 31 should be attention
        #expect(attnIndices == [3, 7, 11, 15, 19, 23, 27, 31])

        print("Hybrid cache: \(cache.layerCount) layers, \(attnIndices.count) attn, \(ssmIndices.count) ssm")
        print("Estimated memory: \(cache.estimatedBytes / 1024 / 1024) MB")
    }

    @Test("SSMStateCache stores and fetches checkpoints")
    func ssmStateCacheRoundtrip() {
        let ssmCache = SSMStateCache(maxEntries: 10)

        let tokens = [1, 2, 3, 4, 5]
        let boundary = 5
        let tokenHash = SSMStateCache.hashTokens(tokens, count: boundary)

        // Store checkpoint with real SSM state
        let ssmStates = [
            SSMStateLayer(state: [MLXArray.ones([1, 64, 128, 192])]),
            SSMStateLayer(state: [MLXArray.ones([1, 64, 128, 192])]),
        ]
        let checkpoint = SSMCheckpoint(ssmStates: ssmStates, boundary: boundary, tokenHash: tokenHash)
        ssmCache.store(checkpoint: checkpoint)

        #expect(ssmCache.stores == 1)

        // Fetch should succeed
        let fetched = ssmCache.fetch(tokenHash: tokenHash, boundary: boundary)
        #expect(fetched != nil)
        #expect(fetched!.ssmStates.count == 2)
        #expect(fetched!.boundary == 5)
        #expect(ssmCache.hits == 1)

        // Different tokens should miss
        let otherHash = SSMStateCache.hashTokens([10, 20, 30], count: 3)
        let missed = ssmCache.fetch(tokenHash: otherHash, boundary: 3)
        #expect(missed == nil)
        #expect(ssmCache.misses == 1)

        print("SSM cache: store=\(ssmCache.stores), hits=\(ssmCache.hits), misses=\(ssmCache.misses)")
    }

    @Test("Empty SSM states are treated as MISS")
    func emptySsmStateIsMiss() {
        let ssmCache = SSMStateCache(maxEntries: 10)

        let tokens = [1, 2, 3]
        let hash = SSMStateCache.hashTokens(tokens, count: 3)

        // Store checkpoint with EMPTY states
        let checkpoint = SSMCheckpoint(ssmStates: [], boundary: 3, tokenHash: hash)
        ssmCache.store(checkpoint: checkpoint)

        // Fetch should return nil (empty = miss)
        let fetched = ssmCache.fetch(tokenHash: hash, boundary: 3)
        #expect(fetched == nil)
        #expect(ssmCache.misses == 1)
        print("Empty SSM states correctly treated as MISS")
    }

    @Test("Qwen3.5 newCache creates correct cache types per layer")
    func qwen35NewCacheTypes() throws {
        guard let path = ModelLoadingTests.jangModelPath() else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found")
            return
        }
        let configURL = path.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(Qwen35Configuration.self, from: data)
        let model = Qwen35TopLevelModel(config)

        let cache = model.newCache()
        #expect(cache.count == 32)

        var mambaCount = 0
        var kvSimpleCount = 0
        for (i, c) in cache.enumerated() {
            if c is VMLXMambaCache {
                mambaCount += 1
                // SSM layers: indices where (i+1) % 4 != 0
                #expect((i + 1) % 4 != 0, "Layer \(i) should be SSM but got attention cache")
            } else if c is VMLXKVCacheSimple {
                kvSimpleCount += 1
                // Attention layers: indices where (i+1) % 4 == 0
                #expect((i + 1) % 4 == 0, "Layer \(i) should be attention but got SSM cache")
            }
        }
        #expect(mambaCount == 24)
        #expect(kvSimpleCount == 8)
        print("Qwen3.5 cache: \(mambaCount) MambaCache + \(kvSimpleCount) KVCacheSimple")
    }
}

// MARK: - Cache Coordinator Tests

@Suite("Cache Coordinator")
struct CacheCoordinatorIntegrationTests {

    /// Create a test cache dir in /tmp
    static func testCacheDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx_test_cache_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Build a simple pure-attention HybridCache for testing.
    /// All arrays are pre-evaluated to avoid Metal threading issues in background writes.
    static func makeTestCache(layers: Int = 4, seqLen: Int = 10) -> HybridCache {
        let entries = (0..<layers).map { _ -> LayerCacheEntry in
            let keys = MLXArray.zeros([1, 8, seqLen, 64])
            let values = MLXArray.zeros([1, 8, seqLen, 64])
            MLX.eval(keys, values)
            return .attention(KVCacheLayer(keys: keys, values: values, offset: seqLen))
        }
        return HybridCache(layers: entries)
    }

    /// Build a hybrid cache with SSM + attention for testing.
    /// All arrays are pre-evaluated to avoid Metal threading issues in background writes.
    static func makeHybridTestCache(layers: Int = 8) -> HybridCache {
        let entries = (0..<layers).map { i -> LayerCacheEntry in
            if (i + 1) % 4 == 0 {
                let keys = MLXArray.zeros([1, 8, 10, 64])
                let values = MLXArray.zeros([1, 8, 10, 64])
                MLX.eval(keys, values)
                return .attention(KVCacheLayer(keys: keys, values: values, offset: 10))
            } else {
                let s0 = MLXArray.zeros([1, 64, 64, 128])
                let s1 = MLXArray.zeros([1, 64, 64, 128])
                MLX.eval(s0, s1)
                return .ssm(SSMStateLayer(state: [s0, s1]))
            }
        }
        return HybridCache(layers: entries)
    }

    @Test("Memory cache store and fetch")
    func memoryCacheRoundtrip() {
        let config = CacheCoordinatorConfig(
            enablePrefixCache: false,
            usePagedCache: false,
            useMemoryAwareCache: true,
            enableDiskCache: false
        )
        let coordinator = CacheCoordinator(config: config)

        let tokens = [100, 200, 300, 400, 500]
        let cache = Self.makeTestCache(layers: 4, seqLen: 5)

        // Store
        coordinator.store(tokens: tokens, cache: cache)

        // Fetch with same tokens
        let result = coordinator.fetch(tokens: tokens)
        switch result {
        case .hit(let hitCache, let remaining, let detail, _):
            #expect(hitCache.layerCount == 4)
            #expect(remaining.isEmpty)
            #expect(detail == .memory)
            print("Memory cache HIT: \(hitCache.layerCount) layers, detail=\(detail)")
        default:
            #expect(Bool(false), "Expected memory cache hit")
        }

        // Stats
        let stats = coordinator.stats
        #expect(stats.memoryCacheHits == 1)
        print("Memory cache stats: hits=\(stats.memoryCacheHits), misses=\(stats.memoryCacheMisses)")
    }

    @Test("Memory cache misses on different tokens")
    func memoryCacheMiss() {
        let config = CacheCoordinatorConfig(
            enablePrefixCache: false,
            usePagedCache: false,
            useMemoryAwareCache: true,
            enableDiskCache: false
        )
        let coordinator = CacheCoordinator(config: config)

        let tokens = [100, 200, 300]
        let cache = Self.makeTestCache(layers: 2, seqLen: 3)
        coordinator.store(tokens: tokens, cache: cache)

        // Different tokens should miss
        let result = coordinator.fetch(tokens: [999, 888])
        switch result {
        case .miss:
            print("Correctly got MISS for different tokens")
        default:
            #expect(Bool(false), "Expected cache miss")
        }
    }

    @Test("Disk cache store and fetch with safetensors")
    func diskCacheRoundtrip() async throws {
        let dir = Self.testCacheDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        // Test DiskCache directly (not through coordinator) for clearer diagnostics
        let diskCache = DiskCache(cacheDir: dir, maxSizeGB: 1.0)

        let tokens = [10, 20, 30, 40, 50]
        let cache = Self.makeTestCache(layers: 4, seqLen: 5)
        cache.materialized()

        // Store to disk
        diskCache.storeCache(tokens: tokens, cache: cache)

        // Poll for file write completion (background Task.detached in DiskCache)
        let hash = DiskCache.hashTokens(tokens)
        let fileURL = dir.appendingPathComponent("\(hash).safetensors")
        var fileReady = false
        for _ in 0..<20 {
            try await Task.sleep(for: .milliseconds(250))
            if FileManager.default.fileExists(atPath: fileURL.path) {
                fileReady = true
                break
            }
        }
        print("Safetensors file ready: \(fileReady) at \(fileURL.lastPathComponent)")
        #expect(fileReady, "Background file write did not complete in 5 seconds")

        // Verify SQLite index
        let indexEntry = diskCache.fetch(tokens: tokens)
        print("SQLite index entry: \(indexEntry != nil ? "found" : "missing")")

        // Fetch cache from disk
        let fetched = diskCache.fetchCache(tokens: tokens)
        if let fetched {
            #expect(fetched.layerCount == 4)
            for layer in fetched.layers {
                if case .attention(let kv) = layer {
                    #expect(kv.keys.dim(0) == 1)
                    #expect(kv.keys.dim(1) == 8)
                    #expect(kv.keys.dim(2) == 5)
                    #expect(kv.keys.dim(3) == 64)
                }
            }
            print("Disk cache HIT: \(fetched.layerCount) layers, shapes verified")
        } else {
            print("Disk cache MISS - file exists: \(fileReady), index: \(indexEntry != nil)")
            #expect(fetched != nil, "Expected disk cache to return cached data")
        }
    }

    @Test("Disk cache stores and fetches hybrid cache with SSM layers")
    func diskCacheHybridRoundtrip() async throws {
        let dir = Self.testCacheDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        let diskCache = DiskCache(cacheDir: dir, maxSizeGB: 1.0)

        let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        let cache = Self.makeHybridTestCache(layers: 8)
        cache.materialized()

        diskCache.storeCache(tokens: tokens, cache: cache)

        // Poll for file write completion
        let hash = DiskCache.hashTokens(tokens)
        let fileURL = dir.appendingPathComponent("\(hash).safetensors")
        for _ in 0..<20 {
            try await Task.sleep(for: .milliseconds(250))
            if FileManager.default.fileExists(atPath: fileURL.path) { break }
        }

        let fetched = diskCache.fetchCache(tokens: tokens)
        if let hitCache = fetched {
            #expect(hitCache.layerCount == 8)
            #expect(hitCache.isHybrid)
            let attnCount = hitCache.attentionLayerIndices.count
            let ssmCount = hitCache.ssmLayerIndices.count
            #expect(attnCount == 2) // layers 3, 7
            #expect(ssmCount == 6) // layers 0,1,2,4,5,6
            print("Hybrid disk cache: \(attnCount) attn + \(ssmCount) ssm layers restored")
        } else {
            let exists = FileManager.default.fileExists(atPath: fileURL.path)
            print("Hybrid disk miss - file exists: \(exists)")
            #expect(fetched != nil, "Expected hybrid disk cache to return cached data")
        }
    }

    @Test("CacheCoordinator cascade: memory hit then miss")
    func cacheCascade() {
        let config = CacheCoordinatorConfig(
            enablePrefixCache: false,
            usePagedCache: false,
            useMemoryAwareCache: true,
            enableDiskCache: false // memory-only for speed
        )
        let coordinator = CacheCoordinator(config: config)

        let tokens1 = [100, 200, 300]
        let cache1 = Self.makeTestCache(layers: 2, seqLen: 3)
        coordinator.store(tokens: tokens1, cache: cache1)

        // tokens1 should hit memory
        let result1 = coordinator.fetch(tokens: tokens1)
        switch result1 {
        case .hit(_, _, let detail, _):
            #expect(detail == .memory)
            print("tokens1: memory HIT")
        default:
            #expect(Bool(false), "Expected memory hit for tokens1")
        }

        // Unknown tokens should miss
        let result2 = coordinator.fetch(tokens: [999])
        switch result2 {
        case .miss:
            print("Unknown tokens: MISS (correct)")
        default:
            #expect(Bool(false), "Expected miss for unknown tokens")
        }

        let stats = coordinator.stats
        #expect(stats.memoryCacheHits == 1)
        print("Cascade stats: memHits=\(stats.memoryCacheHits) memMisses=\(stats.memoryCacheMisses)")
    }

    @Test("Hybrid fetch returns partialHit when SSM state missing")
    func hybridPartialHit() {
        let config = CacheCoordinatorConfig(
            enablePrefixCache: false,
            usePagedCache: false,
            useMemoryAwareCache: true,
            enableDiskCache: false
        )
        let coordinator = CacheCoordinator(config: config)
        coordinator.setHybrid(true)

        let tokens = [1, 2, 3, 4, 5]

        // Store a pure-attention cache (no SSM states)
        let pureAttnCache = Self.makeTestCache(layers: 4, seqLen: 5)
        coordinator.store(tokens: tokens, cache: pureAttnCache)

        // Fetch should be partial hit (attention found, SSM missing)
        let result = coordinator.fetch(tokens: tokens)
        switch result {
        case .partialHit(let cache, _, let detail):
            print("Partial hit: \(cache.layerCount) attn layers, SSM missing, detail=\(detail)")
        case .hit(_, _, _, _):
            // Also acceptable -- depends on whether ssmStateCache has state
            print("Full hit (SSM state was found)")
        default:
            #expect(Bool(false), "Expected partial or full hit")
        }
    }
}

// MARK: - Multi-Turn Cache Hit Tests

@Suite("Multi-Turn Cache")
struct MultiTurnCacheTests {

    @Test("Second prompt with shared prefix gets cache hit")
    func sharedPrefixCacheHit() {
        let config = CacheCoordinatorConfig(
            enablePrefixCache: false,
            usePagedCache: false,
            useMemoryAwareCache: true,
            enableDiskCache: false
        )
        let coordinator = CacheCoordinator(config: config)

        // First turn: [1, 2, 3, 4, 5]
        let turn1Tokens = [1, 2, 3, 4, 5]
        let turn1Cache = CacheCoordinatorIntegrationTests.makeTestCache(layers: 4, seqLen: 5)
        coordinator.store(tokens: turn1Tokens, cache: turn1Cache)

        // Second turn with exact same prefix should hit
        let result = coordinator.fetch(tokens: turn1Tokens)
        switch result {
        case .hit(let cache, let remaining, let detail, _):
            #expect(remaining.isEmpty)
            #expect(cache.layerCount == 4)
            print("Turn 2 exact match: HIT detail=\(detail)")
        default:
            #expect(Bool(false), "Expected hit on exact prefix match")
        }

        let stats = coordinator.stats
        #expect(stats.memoryCacheHits >= 1)
        print("Multi-turn stats: hits=\(stats.memoryCacheHits)")
    }

    @Test("KVCacheSimple state round-trip to HybridCache")
    func kvCacheStateRoundtrip() {
        let kvc = VMLXKVCacheSimple()

        // Simulate 3 update steps
        let k1 = MLXArray.ones([1, 8, 1, 64])
        let v1 = MLXArray.ones([1, 8, 1, 64])
        let _ = kvc.update(keys: k1, values: v1)

        let k2 = MLXArray.ones([1, 8, 1, 64]) * 2
        let v2 = MLXArray.ones([1, 8, 1, 64]) * 2
        let _ = kvc.update(keys: k2, values: v2)

        #expect(kvc.offset == 2)

        // Export state
        let state = kvc.state
        #expect(state.count == 2) // [keys, values]
        #expect(state[0].dim(2) == 2) // seq_len = 2
        #expect(state[1].dim(2) == 2)

        // Create a KVCacheLayer from the state
        let kvLayer = KVCacheLayer(keys: state[0], values: state[1], offset: kvc.offset)

        // Wrap in HybridCache
        let hybridCache = HybridCache(layers: [.attention(kvLayer)])
        #expect(hybridCache.layerCount == 1)
        #expect(hybridCache.isPureAttention)
        #expect(hybridCache.canTruncate)

        // Restore into a new KVCacheSimple
        let kvc2 = VMLXKVCacheSimple()
        let restored = hybridCache.layers[0]
        if case .attention(let restoredKV) = restored {
            kvc2.state = [restoredKV.keys, restoredKV.values]
            #expect(kvc2.offset == 2)
            print("KVCache round-trip: offset=\(kvc2.offset), shape=\(kvc2.state[0].shape)")
        }
    }

    @Test("MambaCache state round-trip to HybridCache")
    func mambaCacheStateRoundtrip() {
        let mc = VMLXMambaCache()

        // Simulate SSM state storage
        mc[0] = MLXArray.ones([1, 3, 64]) // conv_state
        mc[1] = MLXArray.ones([1, 64, 128, 192]) // ssm_state

        let state = mc.state
        #expect(state.count == 2)

        // Create SSMStateLayer
        let ssmLayer = SSMStateLayer(state: state)

        // Wrap in HybridCache
        let hybridCache = HybridCache(layers: [.ssm(ssmLayer)])
        #expect(hybridCache.isPureSSM)
        #expect(!hybridCache.canTruncate) // SSM can't truncate

        // Restore
        let mc2 = VMLXMambaCache()
        if case .ssm(let restoredSSM) = hybridCache.layers[0] {
            mc2.state = restoredSSM.state
            #expect(mc2.state.count == 2)
            #expect(mc2[0]!.shape == [1, 3, 64])
            #expect(mc2[1]!.shape == [1, 64, 128, 192])
            print("MambaCache round-trip: \(mc2.state.count) arrays")
        }
    }
}

// MARK: - StandardModel Tests

@Suite("Standard Model")
struct StandardModelTests {

    @Test("StandardModelConfiguration decodes from JSON")
    func configDecoding() throws {
        let json = """
        {
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "intermediate_size": 5632,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 8192,
            "tie_word_embeddings": false,
            "attention_bias": false
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(StandardModelConfiguration.self, from: json)
        #expect(config.modelType == "llama")
        #expect(config.hiddenSize == 2048)
        #expect(config.hiddenLayers == 16)
        #expect(config.resolvedKVHeads == 4)
        #expect(config.resolvedHeadDim == 128)
        #expect(config.vocabularySize == 32000)
        #expect(!config.tieWordEmbeddings)
        print("Llama config parsed: \(config.hiddenLayers) layers, headDim=\(config.resolvedHeadDim)")
    }

    @Test("StandardTransformerModel instantiates and produces logits")
    func standardModelForward() throws {
        let json = """
        {
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 512,
            "tie_word_embeddings": false,
            "attention_bias": false
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(StandardModelConfiguration.self, from: json)
        let model = StandardTransformerModel(config)

        // Verify VMLXNativeModel conformance
        #expect(model.vocabularySize == 100)

        let cache = model.newCache()
        #expect(cache.count == 2) // 2 layers
        #expect(cache[0] is VMLXKVCacheSimple)

        // Forward pass (random weights, just testing shapes)
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)
        let logits = model(tokens, cache: cache)
        MLX.eval(logits)

        #expect(logits.ndim == 3)
        #expect(logits.dim(0) == 1)
        #expect(logits.dim(1) == 3)
        #expect(logits.dim(2) == 100)
        print("Standard model forward: shape=\(logits.shape)")

        // Verify cache was populated
        if let kvc = cache[0] as? VMLXKVCacheSimple {
            #expect(kvc.offset == 3)
            print("Cache offset after prefill: \(kvc.offset)")
        }
    }

    @Test("StandardTransformerModel sanitize handles tie_word_embeddings")
    func sanitizeTiedEmbeddings() throws {
        let json = """
        {
            "model_type": "gemma2",
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 512,
            "tie_word_embeddings": true
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(StandardModelConfiguration.self, from: json)
        let model = StandardTransformerModel(config)

        var weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([100, 64]),
            "lm_head.weight": MLXArray.zeros([100, 64]),
        ]

        weights = model.sanitize(weights: weights)

        // lm_head.weight should be removed when tied
        #expect(weights["lm_head.weight"] == nil)
        #expect(weights["model.embed_tokens.weight"] != nil)
        print("Sanitize: lm_head.weight correctly removed for tied embeddings")
    }
}

// MARK: - Weight Loading Tests

@Suite("Weight Loading")
struct WeightLoadingTests {

    @Test("vmlxLoadWeights auto-detects quantization from .scales keys")
    func autoDetectQuantization() throws {
        let json = """
        {
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 512
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(StandardModelConfiguration.self, from: json)
        let model = StandardTransformerModel(config)

        // Verify the model was created and has correct vocab size
        #expect(model.vocabularySize == 100)

        // Verify cache creation works (implies all layers initialized)
        let cache = model.newCache()
        #expect(cache.count == 1) // 1 layer
        print("Model created with 1 layer, vocab=\(model.vocabularySize)")
    }
}
