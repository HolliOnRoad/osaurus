import Testing
import Foundation
import MLX
import MLXNN
@testable import VMLXRuntime

// MARK: - Real Model Paths

private enum TestModels {
    static let home = FileManager.default.homeDirectoryForCurrentUser

    /// JANG: Qwen3.5-4B-JANG_2S (hybrid SSM+attention, 2-bit JANG quantized)
    static var jangQwen35: URL? {
        let path = home.appendingPathComponent("jang/models/Qwen3.5-4B-JANG_2S")
        return hasConfig(path) ? path : nil
    }

    /// MLX: Llama-3.2-1B-Instruct-4bit (standard transformer, 4-bit MLX quantized)
    static var mlxLlama: URL? {
        let path = home.appendingPathComponent(
            ".cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/08231374eeacb049a0eade7922910865b8fce912")
        return hasConfig(path) ? path : nil
    }

    /// MLX: Qwen2.5-0.5B-Instruct-4bit (standard transformer, 4-bit MLX quantized)
    static var mlxQwen25: URL? {
        let path = home.appendingPathComponent(
            ".cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3")
        return hasConfig(path) ? path : nil
    }

    private static func hasConfig(_ path: URL) -> Bool {
        FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path)
    }
}

// MARK: - Llama 3.2 1B (Standard MLX Model)

@Suite("Llama 3.2 1B MLX")
struct LlamaMLXTests {

    @Test("Load Llama 3.2 1B model")
    func loadModel() async throws {
        guard let path = TestModels.mlxLlama else {
            print("SKIP: Llama-3.2-1B-Instruct-4bit not found in HF cache")
            return
        }

        print("Loading Llama 3.2 1B from \(path.lastPathComponent)...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let loaded = try await ModelLoader.load(from: path)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print("  Loaded in \(String(format: "%.2f", elapsed))s")

        #expect(loaded.vocabSize == 128256)
        #expect(loaded.numLayers == 16)
        #expect(loaded.hiddenSize == 2048)
        #expect(!loaded.detected.isJang)

        // Verify it created a StandardTransformerModel
        #expect(loaded.nativeModel is StandardTransformerModel)
        print("  Model type: StandardTransformerModel")
        print("  Vocab: \(loaded.vocabSize), Layers: \(loaded.numLayers), Hidden: \(loaded.hiddenSize)")
    }

    @Test("Llama forward pass and cache")
    func forwardPass() async throws {
        guard let path = TestModels.mlxLlama else {
            print("SKIP: Llama-3.2-1B-Instruct-4bit not found")
            return
        }

        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)

        // All caches should be KVCacheSimple (no SSM)
        let cache = container.newCache()
        #expect(cache.count == 16)
        #expect(cache.allSatisfy { $0 is VMLXKVCacheSimple })
        print("Cache: \(cache.count) VMLXKVCacheSimple layers (pure attention)")

        // Prefill 3 tokens
        let tokens = MLXArray([Int32(128000), Int32(9906), Int32(0)]).reshaped(1, 3)
        let t0 = CFAbsoluteTimeGetCurrent()
        let logits = container.forward(tokens, cache: cache)
        // Force MLX lazy computation to materialize tensors on GPU (not code execution)
        MLX.eval(logits)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        #expect(logits.shape == [1, 3, 128256])
        print("Prefill 3 tokens: \(logits.shape) in \(String(format: "%.3f", elapsed))s")

        // Cache should have offset 3
        let kvc = cache[0] as! VMLXKVCacheSimple
        #expect(kvc.offset == 3)

        // Decode 1 token
        let nextToken = logits[0, -1].argMax().item(Int.self)
        let decodeInput = MLXArray([Int32(nextToken)]).reshaped(1, 1)
        let t1 = CFAbsoluteTimeGetCurrent()
        let decodeLogits = container.forward(decodeInput, cache: cache)
        MLX.eval(decodeLogits)
        let decodeElapsed = CFAbsoluteTimeGetCurrent() - t1

        #expect(decodeLogits.shape == [1, 1, 128256])
        #expect(kvc.offset == 4)
        print("Decode 1 token: \(String(format: "%.3f", decodeElapsed))s, cache offset=\(kvc.offset)")
    }

    @Test("Llama generate text")
    func generateText() async throws {
        guard let path = TestModels.mlxLlama else {
            print("SKIP: Llama-3.2-1B-Instruct-4bit not found")
            return
        }

        let runtime = VMLXRuntimeActor()
        try await runtime.loadModel(from: path)

        let request = VMLXChatCompletionRequest(
            messages: [
                VMLXChatMessage(role: "user", content: "What is 2+2? Answer with just the number.")
            ],
            model: nil,
            temperature: 0,
            maxTokens: 20,
            topP: 1.0,
            repetitionPenalty: 1.0,
            stop: [],
            stream: false
        )

        let t0 = CFAbsoluteTimeGetCurrent()
        let result = try await runtime.generate(request: request)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        print("Generated (\(String(format: "%.2f", elapsed))s): '\(result)'")
        #expect(!result.isEmpty)

        await runtime.unloadModel()
    }
}

// MARK: - Qwen 2.5 0.5B (Standard MLX Model)

@Suite("Qwen 2.5 0.5B MLX")
struct QwenMLXTests {

    @Test("Load Qwen 2.5 0.5B model")
    func loadModel() async throws {
        guard let path = TestModels.mlxQwen25 else {
            print("SKIP: Qwen2.5-0.5B-Instruct-4bit not found in HF cache")
            return
        }

        print("Loading Qwen 2.5 0.5B from \(path.lastPathComponent)...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let loaded = try await ModelLoader.load(from: path)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print("  Loaded in \(String(format: "%.2f", elapsed))s")

        #expect(loaded.vocabSize == 151936)
        #expect(loaded.numLayers == 24)
        #expect(loaded.hiddenSize == 896)
        #expect(!loaded.detected.isJang)
        #expect(loaded.nativeModel is StandardTransformerModel)
        print("  Vocab: \(loaded.vocabSize), Layers: \(loaded.numLayers), Hidden: \(loaded.hiddenSize)")
    }

    @Test("Qwen 2.5 forward pass")
    func forwardPass() async throws {
        guard let path = TestModels.mlxQwen25 else {
            print("SKIP: Qwen2.5-0.5B-Instruct-4bit not found")
            return
        }

        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)
        let cache = container.newCache()

        #expect(cache.count == 24)
        #expect(cache.allSatisfy { $0 is VMLXKVCacheSimple })

        // Prefill
        let tokens = MLXArray([Int32(1), Int32(100), Int32(200)]).reshaped(1, 3)
        let t0 = CFAbsoluteTimeGetCurrent()
        let logits = container.forward(tokens, cache: cache)
        MLX.eval(logits)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        #expect(logits.shape == [1, 3, 151936])
        print("Qwen2.5 prefill 3 tokens: \(logits.shape) in \(String(format: "%.3f", elapsed))s")
    }

    @Test("Qwen 2.5 generate text")
    func generateText() async throws {
        guard let path = TestModels.mlxQwen25 else {
            print("SKIP: Qwen2.5-0.5B-Instruct-4bit not found")
            return
        }

        let runtime = VMLXRuntimeActor()
        try await runtime.loadModel(from: path)

        let request = VMLXChatCompletionRequest(
            messages: [
                VMLXChatMessage(role: "user", content: "Say hello")
            ],
            model: nil,
            temperature: 0,
            maxTokens: 15,
            topP: 1.0,
            repetitionPenalty: 1.0,
            stop: [],
            stream: false
        )

        let t0 = CFAbsoluteTimeGetCurrent()
        let result = try await runtime.generate(request: request)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        print("Qwen2.5 generated (\(String(format: "%.2f", elapsed))s): '\(result)'")
        #expect(!result.isEmpty)

        await runtime.unloadModel()
    }
}

// MARK: - Qwen 3.5 4B JANG (Hybrid SSM Model)

@Suite("Qwen 3.5 4B JANG")
struct QwenJANGTests {

    @Test("Load Qwen3.5-4B-JANG_2S model")
    func loadModel() async throws {
        guard let path = TestModels.jangQwen35 else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found at ~/jang/models/")
            return
        }

        print("Loading Qwen3.5-4B-JANG_2S...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let loaded = try await ModelLoader.load(from: path)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print("  Loaded in \(String(format: "%.2f", elapsed))s")

        #expect(loaded.numLayers == 32)
        #expect(loaded.detected.isJang)
        #expect(loaded.nativeModel is Qwen35TopLevelModel)
        print("  Model type: Qwen35TopLevelModel (hybrid SSM)")
        print("  Vocab: \(loaded.vocabSize), Layers: \(loaded.numLayers), Hidden: \(loaded.hiddenSize)")
    }

    @Test("Qwen3.5 JANG hybrid cache structure")
    func hybridCache() async throws {
        guard let path = TestModels.jangQwen35 else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found")
            return
        }

        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)
        let cache = container.newCache()

        #expect(cache.count == 32)

        var ssmCount = 0
        var attnCount = 0
        for c in cache {
            if c is VMLXMambaCache { ssmCount += 1 }
            else if c is VMLXKVCacheSimple { attnCount += 1 }
        }

        #expect(ssmCount == 24)
        #expect(attnCount == 8)
        print("Hybrid cache: \(ssmCount) SSM (GatedDeltaNet) + \(attnCount) attention (GQA)")

        #expect(container.isHybrid || container.layerPattern != nil)
        print("Container isHybrid: \(container.isHybrid), layerPattern: \(container.layerPattern?.count ?? 0) layers")
    }

    @Test("Qwen3.5 JANG forward pass with hybrid caching")
    func forwardPass() async throws {
        guard let path = TestModels.jangQwen35 else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found")
            return
        }

        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)
        let cache = container.newCache()

        // Prefill 3 tokens
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)
        let t0 = CFAbsoluteTimeGetCurrent()
        let logits = container.forward(tokens, cache: cache)
        MLX.eval(logits)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        #expect(logits.ndim == 3)
        #expect(logits.dim(0) == 1)
        #expect(logits.dim(1) == 3)
        print("Qwen3.5 JANG prefill 3 tokens: \(logits.shape) in \(String(format: "%.3f", elapsed))s")
        print("  Vocab from logits: \(logits.dim(2))")

        // Verify SSM caches populated
        var ssmPopulated = 0
        var attnPopulated = 0
        for c in cache {
            if let mc = c as? VMLXMambaCache, !mc.state.isEmpty { ssmPopulated += 1 }
            else if let kvc = c as? VMLXKVCacheSimple, kvc.offset > 0 { attnPopulated += 1 }
        }
        print("  After prefill: \(ssmPopulated) SSM populated, \(attnPopulated) attention populated")

        // Decode 1 token
        let nextToken = logits[0, -1].argMax().item(Int.self)
        let decodeInput = MLXArray([Int32(nextToken)]).reshaped(1, 1)
        let t1 = CFAbsoluteTimeGetCurrent()
        let decodeLogits = container.forward(decodeInput, cache: cache)
        MLX.eval(decodeLogits)
        let decodeElapsed = CFAbsoluteTimeGetCurrent() - t1

        #expect(decodeLogits.dim(1) == 1)
        print("  Decode step: \(String(format: "%.3f", decodeElapsed))s")

        for c in cache {
            if let kvc = c as? VMLXKVCacheSimple {
                #expect(kvc.offset == 4)
                print("  Attention cache offset: \(kvc.offset)")
                break
            }
        }
    }

    @Test("Qwen3.5 JANG generate text")
    func generateText() async throws {
        guard let path = TestModels.jangQwen35 else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found")
            return
        }

        let runtime = VMLXRuntimeActor()
        try await runtime.loadModel(from: path)

        let request = VMLXChatCompletionRequest(
            messages: [
                VMLXChatMessage(role: "user", content: "Hello")
            ],
            model: nil,
            temperature: 0,
            maxTokens: 15,
            topP: 1.0,
            repetitionPenalty: 1.0,
            stop: [],
            stream: false
        )

        let t0 = CFAbsoluteTimeGetCurrent()
        let result = try await runtime.generate(request: request)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        print("Qwen3.5 JANG generated (\(String(format: "%.2f", elapsed))s): '\(result)'")
        #expect(!result.isEmpty)

        await runtime.unloadModel()
    }

    @Test("Qwen3.5 JANG SSM state stored in cache coordinator")
    func ssmCacheStore() async throws {
        guard let path = TestModels.jangQwen35 else {
            print("SKIP: Qwen3.5-4B-JANG_2S not found")
            return
        }

        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)
        let cache = container.newCache()

        // Run forward pass to populate caches
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)
        let logits = container.forward(tokens, cache: cache)
        MLX.eval(logits)

        // Extract SSM and attention state into HybridCache
        var layers: [LayerCacheEntry] = []
        for c in cache {
            if let mc = c as? VMLXMambaCache {
                layers.append(.ssm(SSMStateLayer(state: mc.state)))
            } else if let kvc = c as? VMLXKVCacheSimple {
                let state = kvc.state
                if state.count == 2 {
                    layers.append(.attention(KVCacheLayer(
                        keys: state[0], values: state[1], offset: kvc.offset)))
                } else {
                    layers.append(.attention(KVCacheLayer(
                        keys: MLXArray(), values: MLXArray(), offset: 0)))
                }
            }
        }

        let hybridCache = HybridCache(layers: layers)
        #expect(hybridCache.layerCount == 32)
        print("Extracted HybridCache: \(hybridCache.layerCount) layers")
        print("  Attention: \(hybridCache.attentionLayerIndices.count)")
        print("  SSM: \(hybridCache.ssmLayerIndices.count)")
        print("  Estimated memory: \(hybridCache.estimatedBytes / 1024 / 1024) MB")

        // Store in CacheCoordinator
        let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
            enablePrefixCache: false,
            usePagedCache: false,
            useMemoryAwareCache: true,
            enableDiskCache: false
        ))
        coordinator.setHybrid(true)

        hybridCache.materialized()
        coordinator.store(tokens: [1, 2, 3], cache: hybridCache)

        let result = coordinator.fetch(tokens: [1, 2, 3])
        switch result {
        case .hit(let cached, let remaining, let detail, _):
            #expect(remaining.isEmpty)
            #expect(cached.layerCount == 32)
            print("Cache coordinator HIT: \(cached.layerCount) layers, detail=\(detail)")
        case .partialHit(let cached, _, let detail):
            print("Cache coordinator PARTIAL HIT: \(cached.layerCount) layers, detail=\(detail)")
        default:
            #expect(Bool(false), "Expected cache hit or partial hit")
        }
    }
}
