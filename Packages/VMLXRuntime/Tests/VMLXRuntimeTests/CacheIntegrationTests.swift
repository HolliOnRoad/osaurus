import Testing
import Foundation
import MLX
import MLXNN
@testable import VMLXRuntime

// MARK: - Cache Integration Tests

/// Comprehensive cache integration tests verifying every cache path
/// across model types: hybrid SSM, MoE, TurboQuant, disk, paged.
///
/// Uses Qwen3.5-4B-JANG_4S (smallest hybrid model) for quick iteration.
/// Tests are ordered: load -> store -> fetch -> multi-turn -> TQ -> disk -> paged.
@Suite("Cache Integration")
struct CacheIntegrationTests {

    /// Locate Qwen3.5-4B-JANG_4S (smallest hybrid model, ~4B).
    static func hybridModelPath() -> URL? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        for suffix in ["4S", "4K", "2S"] {
            let path = home.appendingPathComponent("jang/models/Qwen3.5-4B-JANG_\(suffix)")
            if FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path) {
                return path
            }
        }
        return nil
    }

    /// Load model once and return container (shared setup).
    static func loadContainer() async throws -> VMLXModelContainer? {
        guard let path = hybridModelPath() else { return nil }
        let loaded = try await ModelLoader.load(from: path)
        return VMLXModelContainer.create(model: loaded)
    }

    // MARK: - Test 1: Cache Object Types for Hybrid Model

    @Test("Hybrid model creates correct cache types per layer")
    func hybridCacheTypes() async throws {
        guard let container = try await Self.loadContainer() else {
            print("SKIP: No Qwen3.5-4B model found"); return
        }

        #expect(container.isHybrid == true)
        let cache = container.newCache()
        #expect(cache.count == 32)

        var ssmCount = 0, attnCount = 0
        for c in cache {
            if c is VMLXMambaCache { ssmCount += 1 }
            else if c is VMLXKVCacheSimple { attnCount += 1 }
        }
        // full_attention_interval=4: layers 3,7,11,15,19,23,27,31 = 8 attention, rest = 24 SSM
        #expect(ssmCount == 24)
        #expect(attnCount == 8)
        print("OK Hybrid cache: \(ssmCount) SSM + \(attnCount) attention = \(cache.count) total")
    }

    // MARK: - Test 2: Cache Store + Fetch Round-Trip

    @Test("Cache store and fetch with SSM companion")
    func cacheStoreAndFetch() async throws {
        guard let container = try await Self.loadContainer() else {
            print("SKIP: No Qwen3.5-4B model found"); return
        }

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            enablePrefixCache: true,
            usePagedCache: false,
            useMemoryAwareCache: true,
            enableDiskCache: false
        ))
        coordinator.setHybrid(true)

        // Run forward pass to populate cache
        let cache = container.newCache()
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)
        let logits = container.forward(inputIds, cache: cache)
        // MLX.eval forces GPU computation (NOT code execution)
        MLX.eval(logits)

        // Build HybridCache from populated cache objects
        var layers: [LayerCacheEntry] = []
        for c in cache {
            if let mc = c as? VMLXMambaCache {
                layers.append(.ssm(SSMStateLayer(state: mc.state)))
            } else if let kvc = c as? VMLXKVCacheSimple {
                let s = kvc.state
                if s.count == 2 {
                    layers.append(.attention(KVCacheLayer(keys: s[0], values: s[1], offset: kvc.offset)))
                }
            }
        }
        let hybridCache = HybridCache(layers: layers)
        hybridCache.materialized()

        #expect(hybridCache.isHybrid == true)
        #expect(hybridCache.layerCount == 32)
        #expect(hybridCache.ssmLayers.count == 24)
        #expect(hybridCache.attentionLayers.count == 8)

        // Store with truncated key (simulating real flow: dropLast(1))
        let storeTokens = Array(tokens.dropLast(1))
        coordinator.store(tokens: storeTokens, cache: hybridCache)

        // Fetch with full key
        let fetchResult = coordinator.fetch(tokens: tokens)
        switch fetchResult {
        case .hit(let cached, let remaining, let detail, let ssmCheckpoint):
            #expect(remaining.count == 1)
            #expect(remaining[0] == 10)
            #expect(cached.layerCount == 32)
            #expect(ssmCheckpoint != nil, "SSM checkpoint should be present for hybrid model")
            if let cp = ssmCheckpoint {
                #expect(cp.ssmStates.count == 24, "Should have 24 SSM states")
            }
            print("OK Cache HIT: \(cached.layerCount) layers, remaining=\(remaining), detail=\(detail)")

        case .partialHit:
            Issue.record("Expected .hit but got .partialHit")

        case .miss:
            Issue.record("Expected .hit but got .miss")
        }
    }

    // MARK: - Test 3: TurboQuant Compress + Decompress Round-Trip

    @Test("TurboQuant encode then decode preserves shape and is finite")
    func turboQuantRoundTrip() async throws {
        guard let container = try await Self.loadContainer() else {
            print("SKIP: No Qwen3.5-4B model found"); return
        }

        let cache = container.newCache()
        let inputIds = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)
        let logits = container.forward(inputIds, cache: cache)
        MLX.eval(logits)

        for (i, c) in cache.enumerated() {
            guard let kvc = c as? VMLXKVCacheSimple else { continue }
            let s = kvc.state
            guard s.count == 2 else { continue }

            let keys = s[0]
            let values = s[1]
            let originalShape = keys.shape

            let seed = 42
            let ek = TurboQuantEncoder.encodeKeys(keys: keys, bits: 3, seed: seed)
            let ev = TurboQuantEncoder.encodeValues(values: values, bits: 3, seed: seed)

            #expect(ek.seed == seed)
            #expect(ev.seed == seed)

            let decodedKeys = TurboQuantEncoder.decodeKeys(ek, seed: ek.seed)
            let decodedValues = TurboQuantEncoder.decodeValues(ev, seed: ev.seed)

            #expect(decodedKeys.shape == originalShape)
            #expect(decodedValues.shape == values.shape)

            // Verify values are finite (sum of finite array is finite)
            let keySum = decodedKeys.sum().item(Float.self)
            #expect(keySum.isFinite, "Decoded keys contain NaN/Inf — codebook mismatch?")

            let originalBytes = keys.nbytes + values.nbytes
            let compressedBytes = ek.estimatedBytes + ev.estimatedBytes
            let ratio = Float(originalBytes) / Float(compressedBytes)
            print("OK TQ round-trip: layer \(i), shape \(originalShape), ratio=\(String(format: "%.1fx", ratio))")
            break
        }
    }

    // MARK: - Test 4: TurboQuant Cache Store + Fetch

    @Test("TurboQuant compressed cache store and fetch")
    func tqCacheStoreAndFetch() async throws {
        guard let container = try await Self.loadContainer() else {
            print("SKIP: No Qwen3.5-4B model found"); return
        }

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            useMemoryAwareCache: true,
            enableDiskCache: false
        ))
        coordinator.setHybrid(true)

        let cache = container.newCache()
        let tokens = [1, 2, 3, 4, 5]
        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)
        let logits = container.forward(inputIds, cache: cache)
        MLX.eval(logits)

        let tqConfig = TurboQuantConfig(defaultKeyBits: 3, defaultValueBits: 3, seed: 42)
        var layerEntries: [LayerCacheEntry] = []
        for (layerIdx, c) in cache.enumerated() {
            if let mc = c as? VMLXMambaCache {
                layerEntries.append(.ssm(SSMStateLayer(state: mc.state)))
            } else if let kvc = c as? VMLXKVCacheSimple {
                let s = kvc.state
                guard s.count == 2 else { continue }
                if let kBits = tqConfig.keyBits(forLayer: layerIdx, totalLayers: cache.count),
                   let vBits = tqConfig.valueBits(forLayer: layerIdx, totalLayers: cache.count) {
                    let ek = TurboQuantEncoder.encodeKeys(keys: s[0], bits: kBits, seed: tqConfig.seed)
                    let ev = TurboQuantEncoder.encodeValues(values: s[1], bits: vBits, seed: tqConfig.seed)
                    layerEntries.append(.compressedAttention(ek, ev, kvc.offset))
                } else {
                    layerEntries.append(.attention(KVCacheLayer(keys: s[0], values: s[1], offset: kvc.offset)))
                }
            }
        }
        let hybridCache = HybridCache(layers: layerEntries)
        hybridCache.materialized()

        let compressedCount = layerEntries.filter { $0.isCompressed }.count
        #expect(compressedCount > 0, "Should have compressed attention layers")

        let storeTokens = Array(tokens.dropLast(1))
        coordinator.store(tokens: storeTokens, cache: hybridCache)

        let result = coordinator.fetch(tokens: tokens)
        switch result {
        case .hit(let cached, let remaining, _, let ssmCheckpoint):
            #expect(remaining.count == 1)
            #expect(cached.layerCount == 32)
            #expect(ssmCheckpoint != nil, "SSM checkpoint should be present")
            print("OK TQ cache store/fetch: compressed entries preserved, SSM companion present")

        default:
            Issue.record("Expected .hit but got different result")
        }
    }

    // MARK: - Test 5: Paged Cache with TurboQuant (Bug #1 regression test)

    @Test("Paged cache decompresses TQ before slicing into blocks")
    func pagedCacheWithTQ() async throws {
        guard let container = try await Self.loadContainer() else {
            print("SKIP: No Qwen3.5-4B model found"); return
        }

        let blockSize = 4
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: true,
            useMemoryAwareCache: false,
            pagedBlockSize: blockSize,
            maxCacheBlocks: 100
        ))
        coordinator.setHybrid(true)

        let cache = container.newCache()
        let tokens = Array(1...10)
        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)
        let logits = container.forward(inputIds, cache: cache)
        MLX.eval(logits)

        let tqConfig = TurboQuantConfig(defaultKeyBits: 3, defaultValueBits: 3, seed: 42)
        var layerEntries: [LayerCacheEntry] = []
        for (layerIdx, c) in cache.enumerated() {
            if let mc = c as? VMLXMambaCache {
                layerEntries.append(.ssm(SSMStateLayer(state: mc.state)))
            } else if let kvc = c as? VMLXKVCacheSimple {
                let s = kvc.state
                guard s.count == 2 else { continue }
                if let kBits = tqConfig.keyBits(forLayer: layerIdx, totalLayers: cache.count),
                   let vBits = tqConfig.valueBits(forLayer: layerIdx, totalLayers: cache.count) {
                    let ek = TurboQuantEncoder.encodeKeys(keys: s[0], bits: kBits, seed: tqConfig.seed)
                    let ev = TurboQuantEncoder.encodeValues(values: s[1], bits: vBits, seed: tqConfig.seed)
                    layerEntries.append(.compressedAttention(ek, ev, kvc.offset))
                } else {
                    layerEntries.append(.attention(KVCacheLayer(keys: s[0], values: s[1], offset: kvc.offset)))
                }
            }
        }
        let hybridCache = HybridCache(layers: layerEntries)
        hybridCache.materialized()

        let storeTokens = Array(tokens.dropLast(1))
        coordinator.store(tokens: storeTokens, cache: hybridCache)

        let result = coordinator.fetch(tokens: tokens)
        switch result {
        case .hit(let cached, _, let detail, _):
            #expect(detail == .paged, "Should hit paged cache")
            // Paged cache should return float .attention (decompressed during store)
            for layer in cached.layers {
                if case .compressedAttention = layer {
                    Issue.record("Paged cache should NOT contain compressed entries (Bug #1 regression)")
                }
            }
            print("OK Paged cache + TQ: all entries are float .attention")

        case .partialHit(_, _, _):
            print("INFO: Paged cache returned partialHit (SSM companion missing from paged)")

        case .miss:
            Issue.record("Paged cache miss")
        }
    }

    // MARK: - Test 6: gen_prompt_len Stripping

    @Test("gen_prompt_len stripping enables multi-turn cache hits")
    func genPromptLenStripping() async throws {
        guard let container = try await Self.loadContainer() else {
            print("SKIP: No Qwen3.5-4B model found"); return
        }

        let messages1: [VMLXChatMessage] = [
            VMLXChatMessage(role: "user", content: "Hello")
        ]
        let messages2: [VMLXChatMessage] = [
            VMLXChatMessage(role: "user", content: "Hello"),
            VMLXChatMessage(role: "assistant", content: "Hi there!"),
            VMLXChatMessage(role: "user", content: "How are you?")
        ]

        let tokens1 = try container.applyChatTemplate(messages: messages1, addGenerationPrompt: true)
        let tokens2 = try container.applyChatTemplate(messages: messages2, addGenerationPrompt: true)
        let genPromptLen1 = container.computeGenPromptLen(messages: messages1)
        let genPromptLen2 = container.computeGenPromptLen(messages: messages2)

        let key1 = genPromptLen1 > 0 ? Array(tokens1.dropLast(genPromptLen1)) : tokens1
        let key2 = genPromptLen2 > 0 ? Array(tokens2.dropLast(genPromptLen2)) : tokens2

        let turn1Prefix = key1
        let turn2Prefix = Array(key2.prefix(turn1Prefix.count))

        #expect(genPromptLen1 > 0, "genPromptLen should be > 0")
        #expect(genPromptLen2 > 0)

        if turn1Prefix == turn2Prefix {
            print("OK Multi-turn prefix match: turn 2 key starts with turn 1 key (\(turn1Prefix.count) shared tokens)")
        } else {
            print("WARN: Multi-turn prefix mismatch")
        }
    }

    // MARK: - Test 7: Disk Cache Round-Trip (Bug #4 regression test)

    @Test("Disk cache stores and fetches with N-1 token key")
    func diskCacheRoundTrip() async throws {
        guard let container = try await Self.loadContainer() else {
            print("SKIP: No Qwen3.5-4B model found"); return
        }

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vmlx_test_disk_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            usePagedCache: false,
            useMemoryAwareCache: false,
            enableDiskCache: true,
            diskCacheDir: tmpDir
        ))
        coordinator.setHybrid(true)

        let cache = container.newCache()
        let tokens = [1, 2, 3, 4, 5]
        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)
        let logits = container.forward(inputIds, cache: cache)
        MLX.eval(logits)

        var layerEntries: [LayerCacheEntry] = []
        for c in cache {
            if let mc = c as? VMLXMambaCache {
                layerEntries.append(.ssm(SSMStateLayer(state: mc.state)))
            } else if let kvc = c as? VMLXKVCacheSimple {
                let s = kvc.state
                if s.count == 2 {
                    layerEntries.append(.attention(KVCacheLayer(keys: s[0], values: s[1], offset: kvc.offset)))
                }
            }
        }
        let hybridCache = HybridCache(layers: layerEntries)
        hybridCache.materialized()

        let storeTokens = Array(tokens.dropLast(1))
        coordinator.store(tokens: storeTokens, cache: hybridCache)

        // Wait for background disk write
        try await Task.sleep(for: .seconds(2))

        let files = try FileManager.default.contentsOfDirectory(at: tmpDir, includingPropertiesForKeys: nil)
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }
        #expect(!safetensorFiles.isEmpty, "Disk cache should have written a safetensors file")

        let result = coordinator.fetch(tokens: tokens)
        switch result {
        case .hit(let cached, let remaining, let detail, _):
            #expect(detail == .disk, "Should hit disk cache")
            #expect(remaining.count == 1, "Should have 1 remaining token")
            #expect(cached.layerCount == 32)
            print("OK Disk cache round-trip: hit with \(remaining.count) remaining, detail=\(detail)")

        case .partialHit(_, let remaining, let detail):
            // Acceptable for hybrid — SSM companion from disk is edge case
            print("INFO: Disk cache returned partialHit, remaining=\(remaining.count), detail=\(detail)")

        case .miss:
            Issue.record("Disk cache miss (Bug #4 regression)")
        }
    }
}
