# VMLXRuntime: Native Swift Inference Engine for Osaurus

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Osaurus's entire inference backend (`mlx-swift-lm`, `MLXService`, `ModelRuntime`, `MLXGenerationEngine`, `KVCacheStore`, `StreamAccumulator`) with a from-scratch Swift inference engine porting all core VMLX features natively.

**Architecture:** A new Swift package `VMLXRuntime` sits between `mlx-swift` (tensor ops only) and Osaurus's app layer. It owns the full pipeline: model loading (JANG-aware), tokenization, cache management (5-layer stack with hybrid SSM awareness baked in), continuous batching scheduler, TurboQuant 3-bit KV compression, vision-language preprocessing, generation loop, tool/reasoning parsing, and streaming output. It conforms to Osaurus's `ModelService`/`ToolCapableService` protocols so the existing UI, agents, plugins, and server layer keep working during migration.

**Tech Stack:** Swift 6.2, mlx-swift (tensor ops), swift-transformers (tokenizers/Hub), SQLite (disk cache index), safetensors (cache serialization), Metal (via MLX)

**Reference Implementations:**
- VMLX Python engine: `/Users/eric/mlx/vllm-mlx/vmlx_engine/`
- ExploitBot Swift+VMLX: `/Users/eric/exploitbot/`
- Osaurus current backend: `/Users/eric/osa-jang/Packages/OsaurusCore/`

---

## Package Structure

```
Packages/VMLXRuntime/
├── Package.swift
├── Sources/VMLXRuntime/
│   ├── Core/
│   │   ├── Types.swift                    # Request, SamplingParams, RequestOutput, RequestStatus
│   │   ├── LayerCache.swift               # Unified cache enum: .attention(KV) | .ssm(State)
│   │   ├── HybridCache.swift              # [LayerCacheEntry] wrapper with hybrid-aware ops
│   │   ├── SSMCheckpoint.swift            # Mid-prefill SSM checkpoint for thinking models
│   │   ├── ModelContainer.swift           # Model + tokenizer + config wrapper
│   │   └── ModelConfig.swift              # Per-family model configs (65+ families)
│   │
│   ├── Cache/
│   │   ├── CacheBlock.swift               # Block with ref counting, doubly-linked list
│   │   ├── FreeBlockQueue.swift           # O(1) LRU free list
│   │   ├── BlockHashMap.swift             # SHA-256 hash chain -> block lookup
│   │   ├── BlockTable.swift               # Request -> block ID mapping
│   │   ├── PagedCacheManager.swift        # Block allocation, COW, eviction
│   │   ├── PrefixCache.swift              # Trie-based token prefix matching
│   │   ├── BlockAwarePrefixCache.swift    # Block-level prefix cache (hybrid-safe)
│   │   ├── MemoryCache.swift              # RAM-aware LRU with pressure adaptation
│   │   ├── DiskCache.swift                # L2 SSD persistence, SQLite index
│   │   ├── TQDiskStore.swift              # TurboQuant-native 26x compressed serialization
│   │   ├── BlockDiskStore.swift           # Block-level L2 persistence
│   │   ├── SSMStateCache.swift            # Hybrid SSM companion LRU (max 50)
│   │   ├── SSMReDeriver.swift             # Async re-derivation of SSM state from tokens
│   │   └── CacheCoordinator.swift         # Orchestrates all cache layers, unified fetch/store
│   │
│   ├── Quantization/
│   │   ├── TurboQuant.swift               # 3-bit KV compression: encode/decode/recompress
│   │   ├── TurboQuantConfig.swift         # Per-layer bit widths, critical layers
│   │   ├── TurboQuantKVCache.swift        # Cache subclass: compressed keys/values in GPU
│   │   ├── EncodedKeys.swift              # Packed indices, QJL, norms
│   │   ├── EncodedValues.swift            # Packed indices, norms
│   │   └── JangLoader.swift               # JANG v2 model loading, TQ patching, hybrid detect
│   │
│   ├── Scheduler/
│   │   ├── SchedulerConfig.swift          # All knobs: batch sizes, cache flags, KV quant, etc.
│   │   ├── Scheduler.swift                # Continuous batching: waiting->running, prefill+decode
│   │   ├── RequestQueue.swift             # FCFS with priority support
│   │   ├── BatchBuilder.swift             # Constructs batched input tensors from request pool
│   │   └── MLLMScheduler.swift            # Vision-aware scheduler (image/video preprocessing)
│   │
│   ├── Generation/
│   │   ├── GenerationEngine.swift         # Prefill + decode loop, cache reuse, two-phase
│   │   ├── Sampler.swift                  # Temperature, top-p, top-k, min-p, repetition
│   │   ├── StreamAccumulator.swift        # Token->text, stop detection, tool extraction
│   │   └── StopSequenceDetector.swift     # Sliding window stop sequence matching
│   │
│   ├── Vision/
│   │   ├── VisionProcessor.swift          # Image/video preprocessing, frame extraction
│   │   ├── VisionEmbeddingCache.swift     # Cached image embeddings
│   │   └── VLMModelWrapper.swift          # Vision encoder + LLM bridge
│   │
│   ├── Parsers/
│   │   ├── ToolCallParser.swift           # Protocol + auto-detect from model config
│   │   ├── ReasoningParser.swift          # Protocol + auto-detect (think blocks)
│   │   ├── ToolParsers/                   # 14 implementations
│   │   │   ├── QwenToolParser.swift
│   │   │   ├── LlamaToolParser.swift
│   │   │   ├── MistralToolParser.swift
│   │   │   ├── DeepSeekToolParser.swift
│   │   │   ├── HermesToolParser.swift
│   │   │   ├── FunctionaryToolParser.swift
│   │   │   ├── GraniteToolParser.swift
│   │   │   ├── GLMToolParser.swift
│   │   │   ├── MiniMaxToolParser.swift
│   │   │   ├── NemotronToolParser.swift
│   │   │   ├── XLAMToolParser.swift
│   │   │   ├── MoonshotToolParser.swift
│   │   │   ├── StepFunToolParser.swift
│   │   │   └── GenericToolParser.swift    # JSON fallback
│   │   └── ReasoningParsers/
│   │       ├── Qwen3ReasoningParser.swift
│   │       ├── DeepSeekR1Parser.swift
│   │       ├── GPTOSSParser.swift
│   │       └── MistralReasoningParser.swift
│   │
│   └── Integration/
│       ├── VMLXService.swift              # Conforms to ToolCapableService (drop-in for MLXService)
│       ├── VMLXRuntime.swift              # Actor singleton (replaces ModelRuntime)
│       └── ChatMessageMapper.swift        # ChatMessage -> internal format
│
├── Tests/VMLXRuntimeTests/
│   ├── Cache/
│   │   ├── CacheBlockTests.swift
│   │   ├── FreeBlockQueueTests.swift
│   │   ├── PagedCacheManagerTests.swift
│   │   ├── PrefixCacheTests.swift
│   │   ├── MemoryCacheTests.swift
│   │   ├── DiskCacheTests.swift
│   │   ├── TQDiskStoreTests.swift
│   │   ├── SSMStateCacheTests.swift
│   │   └── CacheCoordinatorTests.swift
│   ├── Quantization/
│   │   ├── TurboQuantTests.swift
│   │   ├── TurboQuantKVCacheTests.swift
│   │   └── JangLoaderTests.swift
│   ├── Scheduler/
│   │   ├── SchedulerTests.swift
│   │   ├── RequestQueueTests.swift
│   │   └── BatchBuilderTests.swift
│   ├── Generation/
│   │   ├── SamplerTests.swift
│   │   ├── StreamAccumulatorTests.swift
│   │   └── StopSequenceDetectorTests.swift
│   ├── Parsers/
│   │   ├── ToolParserTests.swift
│   │   └── ReasoningParserTests.swift
│   └── Integration/
│       └── VMLXServiceTests.swift
```

---

## Phase 1: Foundation -- Core Types & Unified Cache Abstraction

The entire engine is built on the insight that hybrid SSM models mix attention layers (KV cache) with SSM layers (cumulative state). Every cache operation must handle both. We bake this in from line one.

### Task 1.1: Package Scaffold

**Files:**
- Create: `Packages/VMLXRuntime/Package.swift`
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/VMLXRuntime.swift` (module namespace)

**Step 1: Create Package.swift**

```swift
// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "VMLXRuntime",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "VMLXRuntime", targets: ["VMLXRuntime"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.29.1"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.1.6"),
    ],
    targets: [
        .target(
            name: "VMLXRuntime",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ]
        ),
        .testTarget(
            name: "VMLXRuntimeTests",
            dependencies: ["VMLXRuntime"]
        ),
    ]
)
```

**Step 2: Create module file**

```swift
// Sources/VMLXRuntime/VMLXRuntime.swift
public enum VMLXRuntimeVersion {
    public static let version = "0.1.0"
}
```

**Step 3: Wire into Osaurus workspace**

Add to `Packages/OsaurusCore/Package.swift` dependencies:
```swift
.package(path: "../VMLXRuntime"),
```

And to the OsaurusCore target:
```swift
.product(name: "VMLXRuntime", package: "VMLXRuntime"),
```

**Step 4: Verify build**

Run: `cd /Users/eric/osa-jang && xcodebuild -workspace osaurus.xcworkspace -scheme OsaurusCore -destination 'platform=macOS' build 2>&1 | tail -5`
Expected: BUILD SUCCEEDED

**Step 5: Commit**

```bash
git add Packages/VMLXRuntime/ Packages/OsaurusCore/Package.swift
git commit -m "feat: scaffold VMLXRuntime package with mlx-swift tensor deps"
```

---

### Task 1.2: Unified Layer Cache Abstraction

This is the most critical type in the entire engine. Every cache operation dispatches on this enum. Hybrid SSM is not an afterthought -- it's a first-class variant.

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Core/LayerCache.swift`
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Core/HybridCache.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Core/LayerCacheTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/VMLXRuntimeTests/Core/LayerCacheTests.swift
import Testing
import MLX
@testable import VMLXRuntime

@Suite("LayerCache")
struct LayerCacheTests {

    @Test("Attention entry stores and retrieves KV tensors")
    func attentionEntry() throws {
        let keys = MLXArray.zeros([1, 8, 64, 128])    // (batch, heads, tokens, dim)
        let values = MLXArray.zeros([1, 8, 64, 128])
        let entry = LayerCacheEntry.attention(KVCacheLayer(keys: keys, values: values, offset: 64))

        guard case .attention(let kv) = entry else {
            Issue.record("Expected attention entry")
            return
        }
        #expect(kv.offset == 64)
        #expect(kv.tokenCount == 64)
        #expect(kv.isAttention == true)
    }

    @Test("SSM entry stores cumulative state")
    func ssmEntry() throws {
        let state = [MLXArray.zeros([1, 16, 256])]  // (batch, d_state, d_inner)
        let entry = LayerCacheEntry.ssm(SSMStateLayer(state: state, isCumulative: true))

        guard case .ssm(let ssm) = entry else {
            Issue.record("Expected SSM entry")
            return
        }
        #expect(ssm.isCumulative == true)
        #expect(ssm.canTruncate == false)
    }

    @Test("HybridCache tracks layer types correctly")
    func hybridCache() throws {
        // Nemotron-H: 36 SSM + 12 attention, interleaved
        var layers: [LayerCacheEntry] = []
        for i in 0..<48 {
            if i % 4 == 3 {
                let kv = KVCacheLayer(
                    keys: MLXArray.zeros([1, 8, 10, 128]),
                    values: MLXArray.zeros([1, 8, 10, 128]),
                    offset: 10
                )
                layers.append(.attention(kv))
            } else {
                let ssm = SSMStateLayer(
                    state: [MLXArray.zeros([1, 16, 256])],
                    isCumulative: true
                )
                layers.append(.ssm(ssm))
            }
        }

        let cache = HybridCache(layers: layers)
        #expect(cache.layerCount == 48)
        #expect(cache.isHybrid == true)
        #expect(cache.attentionLayerIndices.count == 12)
        #expect(cache.ssmLayerIndices.count == 36)
        #expect(cache.canTruncate == false)  // Has SSM layers
    }

    @Test("Pure attention cache can truncate")
    func pureAttentionTruncate() throws {
        let layers: [LayerCacheEntry] = (0..<32).map { _ in
            .attention(KVCacheLayer(
                keys: MLXArray.zeros([1, 8, 100, 128]),
                values: MLXArray.zeros([1, 8, 100, 128]),
                offset: 100
            ))
        }

        let cache = HybridCache(layers: layers)
        #expect(cache.isHybrid == false)
        #expect(cache.canTruncate == true)

        let truncated = cache.truncated(to: 50)
        #expect(truncated != nil)
        #expect(truncated!.layers.count == 32)

        if case .attention(let kv) = truncated!.layers[0] {
            #expect(kv.offset == 50)
        }
    }

    @Test("Hybrid cache refuses truncation")
    func hybridRefusesTruncate() throws {
        let layers: [LayerCacheEntry] = [
            .attention(KVCacheLayer(keys: MLXArray.zeros([1, 8, 10, 128]), values: MLXArray.zeros([1, 8, 10, 128]), offset: 10)),
            .ssm(SSMStateLayer(state: [MLXArray.zeros([1, 16, 256])], isCumulative: true)),
        ]

        let cache = HybridCache(layers: layers)
        let truncated = cache.truncated(to: 5)
        #expect(truncated == nil)  // Cannot truncate with SSM layers
    }

    @Test("estimateMemoryBytes returns non-zero for populated cache")
    func memoryEstimate() throws {
        let kv = KVCacheLayer(
            keys: MLXArray.zeros([1, 8, 100, 128]),
            values: MLXArray.zeros([1, 8, 100, 128]),
            offset: 100
        )
        let entry = LayerCacheEntry.attention(kv)
        #expect(entry.estimatedBytes > 0)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/eric/osa-jang && swift test --package-path Packages/VMLXRuntime --filter LayerCacheTests 2>&1 | tail -10`
Expected: FAIL -- types don't exist yet

**Step 3: Implement LayerCache.swift**

```swift
// Sources/VMLXRuntime/Core/LayerCache.swift
import MLX

// MARK: - Attention Layer (KV Cache)

/// A single attention layer's key-value cache state.
/// Positional -- can be sliced/truncated to shorter token counts.
public struct KVCacheLayer: Sendable {
    public var keys: MLXArray      // (batch, n_kv_heads, tokens, head_dim)
    public var values: MLXArray    // same shape
    public var offset: Int         // current token position

    public var tokenCount: Int { offset }
    public var isAttention: Bool { true }

    public init(keys: MLXArray, values: MLXArray, offset: Int) {
        self.keys = keys
        self.values = values
        self.offset = offset
    }

    /// Slice to keep only the first `n` tokens.
    public func truncated(to n: Int) -> KVCacheLayer {
        precondition(n <= offset, "Cannot extend via truncation")
        return KVCacheLayer(
            keys: keys[.ellipsis, ..<n, 0...],
            values: values[.ellipsis, ..<n, 0...],
            offset: n
        )
    }

    public var estimatedBytes: Int {
        keys.nbytes + values.nbytes
    }
}

// MARK: - SSM Layer (Cumulative State)

/// A single SSM (Mamba/GatedDeltaNet) layer's cumulative state.
/// Path-dependent -- CANNOT be truncated. Includes all prior tokens' contributions.
public struct SSMStateLayer: Sendable {
    public var state: [MLXArray]       // per-component state arrays
    public var isCumulative: Bool      // always true for SSM

    public var canTruncate: Bool { false }

    public init(state: [MLXArray], isCumulative: Bool = true) {
        self.state = state
        self.isCumulative = isCumulative
    }

    public var estimatedBytes: Int {
        state.reduce(0) { $0 + $1.nbytes }
    }
}

// MARK: - Unified Layer Cache Entry

/// Every layer in a model produces exactly one of these.
/// Hybrid models (Nemotron-H, Jamba, Qwen3.5-A3B) mix both types.
public enum LayerCacheEntry: Sendable {
    case attention(KVCacheLayer)
    case ssm(SSMStateLayer)

    public var isAttention: Bool {
        if case .attention = self { return true }
        return false
    }

    public var isSSM: Bool {
        if case .ssm = self { return true }
        return false
    }

    public var canTruncate: Bool {
        switch self {
        case .attention: return true
        case .ssm(let s): return s.canTruncate
        }
    }

    public var estimatedBytes: Int {
        switch self {
        case .attention(let kv): return kv.estimatedBytes
        case .ssm(let ssm): return ssm.estimatedBytes
        }
    }

    /// Truncate if possible, returns nil for SSM layers.
    public func truncated(to tokenCount: Int) -> LayerCacheEntry? {
        switch self {
        case .attention(let kv):
            return .attention(kv.truncated(to: tokenCount))
        case .ssm:
            return nil  // Cumulative state cannot be un-done
        }
    }
}
```

**Step 4: Implement HybridCache.swift**

```swift
// Sources/VMLXRuntime/Core/HybridCache.swift
import MLX

/// A complete model cache -- one entry per layer.
/// Hybrid-aware from day one: truncation, serialization, eviction all
/// dispatch on the layer type rather than assuming uniform KV.
public struct HybridCache: Sendable {
    public var layers: [LayerCacheEntry]

    public init(layers: [LayerCacheEntry]) {
        self.layers = layers
    }

    // MARK: - Introspection

    public var layerCount: Int { layers.count }

    public var isHybrid: Bool {
        let hasAttention = layers.contains { $0.isAttention }
        let hasSSM = layers.contains { $0.isSSM }
        return hasAttention && hasSSM
    }

    public var isPureAttention: Bool {
        layers.allSatisfy { $0.isAttention }
    }

    public var isPureSSM: Bool {
        layers.allSatisfy { $0.isSSM }
    }

    /// Indices of attention (KV) layers -- used for TQ compression, paged blocks, etc.
    public var attentionLayerIndices: [Int] {
        layers.enumerated().compactMap { $0.element.isAttention ? $0.offset : nil }
    }

    /// Indices of SSM layers -- skipped during TQ, require companion state cache.
    public var ssmLayerIndices: [Int] {
        layers.enumerated().compactMap { $0.element.isSSM ? $0.offset : nil }
    }

    /// True only if ALL layers support truncation (no SSM).
    public var canTruncate: Bool {
        layers.allSatisfy { $0.canTruncate }
    }

    // MARK: - Operations

    /// Truncate all attention layers to `tokenCount`. Returns nil if any SSM layer present.
    /// This is the critical safety gate that prevents hybrid cache corruption.
    public func truncated(to tokenCount: Int) -> HybridCache? {
        guard canTruncate else { return nil }

        let truncatedLayers = layers.map { entry -> LayerCacheEntry in
            // Safe to force-unwrap: canTruncate already verified all are truncatable
            entry.truncated(to: tokenCount)!
        }
        return HybridCache(layers: truncatedLayers)
    }

    /// Total estimated memory across all layers.
    public var estimatedBytes: Int {
        layers.reduce(0) { $0 + $1.estimatedBytes }
    }

    /// Extract only the attention KV data (for paged block storage).
    /// Returns (layerIndex, kvLayer) pairs.
    public var attentionLayers: [(index: Int, kv: KVCacheLayer)] {
        layers.enumerated().compactMap { i, entry in
            if case .attention(let kv) = entry { return (i, kv) }
            return nil
        }
    }

    /// Extract only the SSM state data (for companion cache storage).
    /// Returns (layerIndex, ssmLayer) pairs.
    public var ssmLayers: [(index: Int, ssm: SSMStateLayer)] {
        layers.enumerated().compactMap { i, entry in
            if case .ssm(let ssm) = entry { return (i, ssm) }
            return nil
        }
    }

    /// Build a fresh hybrid cache from a model's layer pattern,
    /// inserting empty SSM state at SSM positions and empty KV at attention positions.
    /// Used when reconstructing from paged blocks (which only store KV).
    public static func fromPattern(
        _ pattern: [LayerType],
        kvFactory: () -> KVCacheLayer,
        ssmFactory: () -> SSMStateLayer
    ) -> HybridCache {
        let layers = pattern.map { type -> LayerCacheEntry in
            switch type {
            case .attention: return .attention(kvFactory())
            case .ssm: return .ssm(ssmFactory())
            case .expert: return .attention(kvFactory()) // MoE experts use attention
            }
        }
        return HybridCache(layers: layers)
    }

    /// Materialize all lazy MLXArrays to concrete GPU buffers.
    /// MUST be called after generation before storing in cache.
    /// Without this, next turn replays the entire computation graph.
    ///
    /// NOTE: MLXArray.eval() forces lazy computation to execute on the GPU.
    /// This is critical for cache correctness -- without it, reading the cache
    /// later would re-execute the full computation graph (effectively a full
    /// re-prefill), destroying any caching benefit.
    public func materialized() -> HybridCache {
        var result = self
        for i in 0..<result.layers.count {
            switch result.layers[i] {
            case .attention(var kv):
                kv.keys.eval()
                kv.values.eval()
                result.layers[i] = .attention(kv)
            case .ssm(var ssm):
                ssm.state.forEach { $0.eval() }
                result.layers[i] = .ssm(ssm)
            }
        }
        return result
    }
}

// MARK: - Layer Type Pattern

/// Describes a model's layer composition pattern.
/// Detected from model architecture or jang_config.json hybrid_override_pattern.
public enum LayerType: Sendable, Equatable {
    case attention  // Standard attention -- KV cache
    case ssm        // Mamba/GatedDeltaNet -- cumulative state
    case expert     // MoE expert layer -- typically no cache (or shares attention)
}

/// Parse a hybrid override pattern string like "MMM*MMM*..." into LayerType array.
/// "*" = attention, "M" = SSM, "E" = expert (no cache).
public func parseHybridPattern(_ pattern: String) -> [LayerType] {
    pattern.map { char -> LayerType in
        switch char {
        case "*": return .attention
        case "M": return .ssm
        case "E": return .expert
        default: return .attention
        }
    }
}
```

**Step 5: Run tests**

Run: `cd /Users/eric/osa-jang && swift test --package-path Packages/VMLXRuntime --filter LayerCacheTests 2>&1 | tail -10`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add Packages/VMLXRuntime/Sources/VMLXRuntime/Core/
git add Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Core/
git commit -m "feat(vmlx): unified LayerCacheEntry with hybrid SSM first-class support"
```

---

### Task 1.3: Core Request Types

Port VMLX's Request, SamplingParams, and RequestOutput to Swift.

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Core/Types.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Core/TypesTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/VMLXRuntimeTests/Core/TypesTests.swift
import Testing
import MLX
@testable import VMLXRuntime

@Suite("Core Types")
struct TypesTests {

    @Test("Request defaults")
    func requestDefaults() {
        let req = InferenceRequest(
            requestId: "test-1",
            promptTokenIds: [1, 2, 3, 4, 5]
        )
        #expect(req.status == .waiting)
        #expect(req.numPromptTokens == 5)
        #expect(req.outputTokenIds.isEmpty)
        #expect(req.isFinished == false)
        #expect(req.samplingParams.temperature == 0.7)
    }

    @Test("Request lifecycle: waiting -> running -> finished")
    func requestLifecycle() {
        var req = InferenceRequest(
            requestId: "test-2",
            promptTokenIds: [1, 2, 3]
        )
        #expect(req.status == .waiting)

        req.status = .running
        req.appendOutputToken(100)
        req.appendOutputToken(101)
        #expect(req.numOutputTokens == 2)
        #expect(req.outputTokenIds == [100, 101])

        req.finish(reason: .stop)
        #expect(req.isFinished)
        #expect(req.finishReason == .stop)
    }

    @Test("SamplingParams validation")
    func samplingParams() {
        let params = SamplingParams(
            maxTokens: 1024,
            temperature: 0.0,
            topP: 0.95,
            topK: 50,
            minP: 0.05,
            repetitionPenalty: 1.1
        )
        #expect(params.isGreedy)  // temperature == 0
        #expect(params.maxTokens == 1024)
    }

    @Test("RequestOutput captures deltas")
    func requestOutput() {
        let output = RequestOutput(
            requestId: "test-3",
            newTokenIds: [100, 101, 102],
            newText: "hello"
        )
        #expect(output.newTokenIds.count == 3)
        #expect(output.finishReason == nil)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/eric/osa-jang && swift test --package-path Packages/VMLXRuntime --filter TypesTests 2>&1 | tail -10`
Expected: FAIL

**Step 3: Implement Types.swift**

```swift
// Sources/VMLXRuntime/Core/Types.swift
import Foundation
import MLX

// MARK: - Sampling Parameters

public struct SamplingParams: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var repetitionPenalty: Float
    public var stop: [String]
    public var stopTokenIds: [Int]

    public var isGreedy: Bool { temperature == 0 }

    public init(
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        topK: Int = 0,
        minP: Float = 0.0,
        repetitionPenalty: Float = 1.0,
        stop: [String] = [],
        stopTokenIds: [Int] = []
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.stop = stop
        self.stopTokenIds = stopTokenIds
    }
}

// MARK: - Request Status

public enum RequestStatus: Int, Sendable, Comparable {
    case waiting = 0
    case running = 1
    case preempted = 2
    case finishedStopped = 3
    case finishedLengthCapped = 4
    case finishedAborted = 5

    public var isFinished: Bool { self.rawValue >= RequestStatus.finishedStopped.rawValue }

    public static func < (lhs: RequestStatus, rhs: RequestStatus) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

// MARK: - Finish Reason

public enum FinishReason: String, Sendable {
    case stop = "stop"
    case length = "length"
    case abort = "abort"
    case toolCalls = "tool_calls"
}

// MARK: - Inference Request

public struct InferenceRequest: Sendable, Identifiable {
    public let id: String
    public let requestId: String
    public var promptTokenIds: [Int]
    public var samplingParams: SamplingParams
    public let arrivalTime: Date
    public var priority: Int

    // State
    public var status: RequestStatus
    public var outputTokenIds: [Int]
    public var outputText: String
    public var finishReason: FinishReason?

    // Cache state
    public var promptCache: HybridCache?
    public var cachedTokens: Int
    public var remainingTokenIds: [Int]?

    // Paged cache
    public var blockTableIds: [Int]?
    public var sharedPrefixBlocks: Int

    // Multimodal
    public var pixelValues: MLXArray?
    public var imageGridTHW: [Int]?
    public var attentionMask: MLXArray?
    public var isMultimodal: Bool

    // Reasoning
    public var enableThinking: Bool
    public var reasoningEffort: String

    public var numPromptTokens: Int { promptTokenIds.count }
    public var numOutputTokens: Int { outputTokenIds.count }
    public var numTotalTokens: Int { numPromptTokens + numOutputTokens }
    public var isFinished: Bool { status.isFinished }

    public init(
        requestId: String,
        promptTokenIds: [Int],
        samplingParams: SamplingParams = SamplingParams(),
        priority: Int = 0,
        enableThinking: Bool = false,
        reasoningEffort: String = "medium",
        isMultimodal: Bool = false
    ) {
        self.id = requestId
        self.requestId = requestId
        self.promptTokenIds = promptTokenIds
        self.samplingParams = samplingParams
        self.arrivalTime = Date()
        self.priority = priority
        self.status = .waiting
        self.outputTokenIds = []
        self.outputText = ""
        self.finishReason = nil
        self.promptCache = nil
        self.cachedTokens = 0
        self.remainingTokenIds = nil
        self.blockTableIds = nil
        self.sharedPrefixBlocks = 0
        self.pixelValues = nil
        self.imageGridTHW = nil
        self.attentionMask = nil
        self.isMultimodal = isMultimodal
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
    }

    public mutating func appendOutputToken(_ tokenId: Int) {
        outputTokenIds.append(tokenId)
    }

    public mutating func finish(reason: FinishReason) {
        switch reason {
        case .stop: status = .finishedStopped
        case .length: status = .finishedLengthCapped
        case .abort: status = .finishedAborted
        case .toolCalls: status = .finishedStopped
        }
        finishReason = reason
    }
}

// MARK: - Request Output (per-step delta)

public struct RequestOutput: Sendable {
    public let requestId: String
    public var newTokenIds: [Int]
    public var newText: String
    public var outputTokenIds: [Int]
    public var outputText: String
    public var finishReason: FinishReason?
    public var numTotalTokens: Int

    public init(
        requestId: String,
        newTokenIds: [Int] = [],
        newText: String = "",
        outputTokenIds: [Int] = [],
        outputText: String = "",
        finishReason: FinishReason? = nil,
        numTotalTokens: Int = 0
    ) {
        self.requestId = requestId
        self.newTokenIds = newTokenIds
        self.newText = newText
        self.outputTokenIds = outputTokenIds
        self.outputText = outputText
        self.finishReason = finishReason
        self.numTotalTokens = numTotalTokens
    }
}

// MARK: - Cache Detail (response metadata)

public enum CacheDetail: String, Sendable {
    case full = "full"         // No cache hit, full prefill
    case prefix = "prefix"     // Token-trie prefix cache hit
    case paged = "paged"       // Block-level paged cache hit
    case disk = "disk"         // L2 SSD cache hit
    case memory = "memory"     // Memory-aware cache hit
    case tq = "+tq"            // TurboQuant compressed
}
```

**Step 4: Run tests**

Run: `cd /Users/eric/osa-jang && swift test --package-path Packages/VMLXRuntime --filter TypesTests 2>&1 | tail -10`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add Packages/VMLXRuntime/Sources/VMLXRuntime/Core/Types.swift
git add Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Core/TypesTests.swift
git commit -m "feat(vmlx): core request types -- InferenceRequest, SamplingParams, RequestOutput"
```

---

## Phase 2: Cache Stack -- All 5 Layers

### Task 2.1: Cache Block & Free Block Queue

The foundation of paged caching -- blocks with reference counting and O(1) free list.

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheBlock.swift`
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/FreeBlockQueue.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/CacheBlockTests.swift`

**Step 1: Write the failing test**

```swift
// Tests/VMLXRuntimeTests/Cache/CacheBlockTests.swift
import Testing
import MLX
@testable import VMLXRuntime

@Suite("CacheBlock")
struct CacheBlockTests {

    @Test("Block lifecycle: allocate, fill, share, free")
    func blockLifecycle() {
        let block = CacheBlock(blockId: 1, blockSize: 64)
        #expect(block.refCount == 0)
        #expect(block.tokenCount == 0)
        #expect(!block.isFull(blockSize: 64))

        block.refCount = 1
        block.tokenCount = 64
        #expect(block.isFull(blockSize: 64))
        #expect(!block.isShared)

        // Fork (COW)
        block.refCount = 2
        #expect(block.isShared)

        // Release one reference
        block.refCount = 1
        #expect(!block.isShared)
    }

    @Test("Block hash chain")
    func blockHashChain() {
        let hash1 = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [1, 2, 3])
        let hash2 = CacheBlock.computeBlockHash(parentHash: hash1, tokenIds: [4, 5, 6])

        #expect(hash1 != hash2)
        #expect(hash1.count == 32)  // SHA-256 = 32 bytes

        // Same inputs produce same hash
        let hash1b = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [1, 2, 3])
        #expect(hash1 == hash1b)

        // Different parent produces different hash
        let hash2b = CacheBlock.computeBlockHash(parentHash: nil, tokenIds: [4, 5, 6])
        #expect(hash2 != hash2b)
    }
}

@Suite("FreeBlockQueue")
struct FreeBlockQueueTests {

    @Test("FIFO order: popleft returns LRU")
    func fifoOrder() {
        let queue = FreeBlockQueue()
        let b1 = CacheBlock(blockId: 1, blockSize: 64)
        let b2 = CacheBlock(blockId: 2, blockSize: 64)
        let b3 = CacheBlock(blockId: 3, blockSize: 64)

        queue.append(b1)
        queue.append(b2)
        queue.append(b3)
        #expect(queue.count == 3)

        let popped = queue.popleft()
        #expect(popped?.blockId == 1)  // LRU first
        #expect(queue.count == 2)
    }

    @Test("Remove from middle")
    func removeMiddle() {
        let queue = FreeBlockQueue()
        let b1 = CacheBlock(blockId: 1, blockSize: 64)
        let b2 = CacheBlock(blockId: 2, blockSize: 64)
        let b3 = CacheBlock(blockId: 3, blockSize: 64)

        queue.append(b1)
        queue.append(b2)
        queue.append(b3)

        queue.remove(b2)
        #expect(queue.count == 2)

        #expect(queue.popleft()?.blockId == 1)
        #expect(queue.popleft()?.blockId == 3)
    }

    @Test("Batch allocate popleftN")
    func batchAllocate() {
        let queue = FreeBlockQueue()
        for i in 0..<10 {
            queue.append(CacheBlock(blockId: i, blockSize: 64))
        }

        let batch = queue.popleftN(5)
        #expect(batch.count == 5)
        #expect(batch[0].blockId == 0)
        #expect(batch[4].blockId == 4)
        #expect(queue.count == 5)
    }

    @Test("Popleft from empty returns nil")
    func emptyQueue() {
        let queue = FreeBlockQueue()
        #expect(queue.popleft() == nil)
        #expect(queue.count == 0)
    }
}
```

**Step 2: Run to verify fail**

**Step 3: Implement CacheBlock.swift**

```swift
// Sources/VMLXRuntime/Cache/CacheBlock.swift
import Foundation
import CryptoKit
import MLX

/// A fixed-size block of KV cache data. The fundamental unit of paged cache management.
/// Reference-counted for Copy-on-Write sharing across requests with common prefixes.
public final class CacheBlock: @unchecked Sendable {
    public let blockId: Int
    public let blockSize: Int

    public var refCount: Int = 0
    public var blockHash: BlockHash?
    public var tokenCount: Int = 0

    // Doubly-linked list pointers (for FreeBlockQueue)
    var prevFreeBlock: CacheBlock?
    var nextFreeBlock: CacheBlock?

    // Per-layer cache data: [(keys, values)] for attention layers only
    // SSM state stored separately in SSMStateCache companion
    public var cacheData: [(keys: MLXArray, values: MLXArray)]?

    public var lastAccess: Date = Date()

    public init(blockId: Int, blockSize: Int) {
        self.blockId = blockId
        self.blockSize = blockSize
    }

    public func isFull(blockSize: Int) -> Bool {
        tokenCount >= blockSize
    }

    public var isShared: Bool {
        refCount > 1
    }

    public func touch() {
        lastAccess = Date()
    }

    public func reset() {
        refCount = 0
        blockHash = nil
        tokenCount = 0
        cacheData = nil
        prevFreeBlock = nil
        nextFreeBlock = nil
    }

    // MARK: - Block Hash Chain

    public static func computeBlockHash(
        parentHash: BlockHash?,
        tokenIds: [Int],
        extraKeys: [MLXArray]? = nil
    ) -> BlockHash {
        var hasher = SHA256()

        if let parent = parentHash {
            hasher.update(data: parent.data)
        }

        tokenIds.withUnsafeBufferPointer { buffer in
            hasher.update(bufferPointer: UnsafeRawBufferPointer(buffer))
        }

        if let extras = extraKeys {
            for array in extras {
                let shape = array.shape
                shape.withUnsafeBufferPointer { buffer in
                    hasher.update(bufferPointer: UnsafeRawBufferPointer(buffer))
                }
            }
        }

        let digest = hasher.finalize()
        return BlockHash(Data(digest))
    }
}

/// 32-byte SHA-256 hash identifying a block's content within its chain.
public struct BlockHash: Hashable, Sendable {
    public let data: Data

    public init(_ data: Data) {
        self.data = data
    }

    public var count: Int { data.count }

    public var hexString: String {
        data.map { String(format: "%02x", $0) }.joined()
    }
}
```

**Step 4: Implement FreeBlockQueue.swift**

```swift
// Sources/VMLXRuntime/Cache/FreeBlockQueue.swift
import Foundation

/// O(1) doubly-linked list of free cache blocks, ordered LRU (front) to MRU (back).
public final class FreeBlockQueue: @unchecked Sendable {
    private let head: CacheBlock
    private let tail: CacheBlock
    public private(set) var count: Int = 0

    public init() {
        self.head = CacheBlock(blockId: -1, blockSize: 0)
        self.tail = CacheBlock(blockId: -2, blockSize: 0)
        head.nextFreeBlock = tail
        tail.prevFreeBlock = head
    }

    public func popleft() -> CacheBlock? {
        guard let block = head.nextFreeBlock, block !== tail else { return nil }
        unlink(block)
        count -= 1
        return block
    }

    public func popleftN(_ n: Int) -> [CacheBlock] {
        var result: [CacheBlock] = []
        result.reserveCapacity(min(n, count))
        for _ in 0..<n {
            guard let block = popleft() else { break }
            result.append(block)
        }
        return result
    }

    public func append(_ block: CacheBlock) {
        let prev = tail.prevFreeBlock!
        prev.nextFreeBlock = block
        block.prevFreeBlock = prev
        block.nextFreeBlock = tail
        tail.prevFreeBlock = block
        count += 1
    }

    public func appendN(_ blocks: [CacheBlock]) {
        for block in blocks {
            append(block)
        }
    }

    public func remove(_ block: CacheBlock) {
        guard block.prevFreeBlock != nil, block.nextFreeBlock != nil else { return }
        unlink(block)
        count -= 1
    }

    private func unlink(_ block: CacheBlock) {
        block.prevFreeBlock?.nextFreeBlock = block.nextFreeBlock
        block.nextFreeBlock?.prevFreeBlock = block.prevFreeBlock
        block.prevFreeBlock = nil
        block.nextFreeBlock = nil
    }
}
```

**Step 5: Run tests, Step 6: Commit**

```bash
git commit -m "feat(vmlx): CacheBlock with ref counting + FreeBlockQueue O(1) LRU"
```

---

### Task 2.2: Block Hash Map & Block Table

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/BlockHashMap.swift`
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/BlockTable.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/BlockHashMapTests.swift`

Implement `BlockHashMap` (hash->block lookup) and `BlockTable` (request->blockIds mapping). See PagedCacheManager tests for integration verification.

---

### Task 2.3: Paged Cache Manager

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/PagedCacheManager.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/PagedCacheManagerTests.swift`

Core paged cache: block pool, allocation, COW forking, hash-based prefix reuse, LRU eviction. Thread-safe via `OSAllocatedUnfairLock`.

---

### Task 2.4: Prefix Cache (Trie-based)

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/PrefixCache.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/PrefixCacheTests.swift`

Token-trie with LRU eviction. Ports `PrefixCacheManager`. Handles exact match, shorter prefix, longer prefix (truncation). Returns nil for hybrid models (uses `HybridCache.canTruncate`).

---

### Task 2.5: Memory-Aware Cache

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/MemoryCache.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/MemoryCacheTests.swift`

RAM-aware LRU. Ports `MemoryAwarePrefixCache`. Checks `os_proc_available_memory()` every 60s, shrinks budget if available < 20%. TTL support. Skips hybrid SSM models (can't cache state-dependent operations via prefix).

---

### Task 2.6: Disk Cache (L2 SSD)

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/DiskCache.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/DiskCacheTests.swift`

SQLite-indexed safetensors storage. Ports `DiskCacheManager`. Token hash as PK, WAL mode, connection pool, background writer via detached Task (pre-serialize arrays on calling thread to avoid Metal threading issues), LRU eviction by `last_accessed`.

---

### Task 2.7: TurboQuant Disk Store

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/TQDiskStore.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/TQDiskStoreTests.swift`

26x compressed TQ-native serialization. Ports `tq_disk_store.py`. Stores packed uint32 indices + float16 norms to safetensors with `__tq_native__` marker. Depends on Phase 3 EncodedKeys/Values types.

---

### Task 2.8: SSM State Cache + Checkpoint System

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/SSMStateCache.swift`
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Core/SSMCheckpoint.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/SSMStateCacheTests.swift`

LRU companion for SSM layer state with mid-prefill checkpointing support. Max 50 entries, keyed by token hash + boundary position. Critical: empty array == MISS, not just nil. Stores `SSMCheckpoint` objects that capture SSM state at stable boundaries (before gen_prompt_len) for thinking model support. Deep-copy on fetch (SSM state is mutable). Disk checkpoint support for persistence across eviction.

### Task 2.8b: SSM Re-Deriver

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/SSMReDeriver.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/SSMReDeriverTests.swift`

Actor that manages async re-derivation of SSM state when checkpoint is missing but KV blocks exist. Runs full forward pass on cached tokens, checkpoints SSM at stable boundary, refreshes attention KV (TQ compressed) as side effect. Deduplicates concurrent re-derive requests for same token hash. Decision logic: sync re-derive for short sequences (< 512 tokens), async + full prefill for longer ones.

---

### Task 2.9: Cache Coordinator

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Cache/CacheCoordinator.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Cache/CacheCoordinatorTests.swift`

Orchestrates all 5 layers. Fetch cascade: paged -> memory -> disk -> MISS. Store cascade: memory + disk + paged hashes + SSM companion. Config struct for all cache knobs.

---

## Phase 3: TurboQuant -- 3-bit KV Compression

### Task 3.1: TurboQuant Config

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/TurboQuantConfig.swift`

Per-layer bit widths, critical layer overrides, hybrid-aware (returns nil for SSM layers).

### Task 3.2: Encoded Keys & Values

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/EncodedKeys.swift`
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/EncodedValues.swift`

Packed codebook indices (uint32), QJL sign bits, norms. Stays compressed in GPU.

### Task 3.3: TurboQuantKVCache

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/TurboQuantKVCache.swift`

Two phases: FILL (float16, zero overhead) -> COMPRESS (3-bit, after prefill). Stays compressed during decode.

### Task 3.4: TurboQuant Encode/Decode

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/TurboQuant.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Quantization/TurboQuantTests.swift`

Core compression via MLX Metal ops. Port Python encoder/decoder.

---

## Phase 4: JANG Model Loading

### Task 4.1: JANG Loader

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Quantization/JangLoader.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Quantization/JangLoaderTests.swift`

Auto-detect JANG models, load v2 format (MLX-native safetensors, mmap), patch makeCache for TQ, detect hybrid patterns, handle MLA dimensions.

---

## Phase 5: Continuous Batching Scheduler

### Task 5.1: Scheduler Config

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Scheduler/SchedulerConfig.swift`

All knobs: max_num_seqs, batch sizes, cache flags, KV quant, disk paths.

### Task 5.2: Request Queue

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Scheduler/RequestQueue.swift`

FCFS deque with priority, waiting + running state.

### Task 5.3: Scheduler

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Scheduler/Scheduler.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Scheduler/SchedulerTests.swift`

Core loop: accept -> schedule -> cache check -> batch build -> generate -> collect -> cleanup.

### Task 5.4: Batch Builder

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Scheduler/BatchBuilder.swift`

Construct batched tensors from multiple requests. Handle variable-length with padding. Merge hybrid caches.

### Task 5.5: MLLM Scheduler

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Scheduler/MLLMScheduler.swift`

Vision-aware: preprocess images/video, manage embedding cache, mixed text+vision batches.

---

## Phase 6: Generation Engine

### Task 6.1: Sampler

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Generation/Sampler.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Generation/SamplerTests.swift`

Temperature, top-p, top-k, min-p, repetition penalty. Greedy when temperature=0.

### Task 6.2: Stop Sequence Detector

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Generation/StopSequenceDetector.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Generation/StopSequenceDetectorTests.swift`

Sliding window. Hold back maxStopLen chars, emit safe prefix. Cross-token-boundary detection.

### Task 6.3: Stream Accumulator

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Generation/StreamAccumulator.swift`
- Test: `Packages/VMLXRuntime/Tests/VMLXRuntimeTests/Generation/StreamAccumulatorTests.swift`

AsyncSequence: token IDs -> typed events (.tokens, .toolInvocation, .thinking). Incremental decode with sliding 8-token context. Integrates tool parser + stop detector.

### Task 6.4: Generation Engine

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Generation/GenerationEngine.swift`

Prefill + decode loop. Cache reuse via CacheCoordinator. TQ recompress after prefill (skip SSM). Two-phase prefill for hybrid. Materialize before store.

---

## Phase 7: Vision-Language Pipeline

### Task 7.1: Vision Processor

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Vision/VisionProcessor.swift`

Image: resize 1024x1024 max, normalize, CoreImage -> MLXArray. Video: smart frame extraction 8-64 frames. PNG/JPEG/WebP.

### Task 7.2: Vision Embedding Cache

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Vision/VisionEmbeddingCache.swift`

Cache preprocessed image embeddings by data hash. Avoid re-encoding on cache hit.

### Task 7.3: VLM Model Wrapper

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Vision/VLMModelWrapper.swift`

Bridge vision encoder + LLM. Qwen-VL, Pixtral, InternVL, LLaVA, Gemma 3n, Phi-3-Vision formats. Grid THW for variable resolution.

---

## Phase 8: Tool & Reasoning Parsers

### Task 8.1: Tool Parser Protocol + Auto-Detect

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Parsers/ToolCallParser.swift`

Protocol with processToken/finalize/reset. Auto-detect from model name and chat template.

### Task 8.2: 14 Tool Parser Implementations

One file each in `ToolParsers/`. Qwen, Llama, Mistral, DeepSeek, Hermes, Functionary, Granite, GLM, MiniMax, Nemotron, xLAM, Moonshot, StepFun, Generic (JSON fallback).

### Task 8.3: Reasoning Parser + 4 Implementations

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Parsers/ReasoningParser.swift`
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Parsers/ReasoningParsers/*.swift`

Extract think blocks. Qwen3, DeepSeek-R1, GPT-OSS, Mistral formats.

---

## Phase 9: Osaurus Integration

### Task 9.1: ChatMessage Mapper

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/ChatMessageMapper.swift`

Map Osaurus ChatMessage (OpenAI format + multimodal parts) -> InferenceRequest. Handle text, images, tool calls, tool results.

### Task 9.2: VMLXRuntime Actor

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXRuntime.swift`

Singleton actor replacing ModelRuntime. Owns model loading, scheduler, cache coordinator, generation engine.

### Task 9.3: VMLXService (Drop-in Replacement)

**Files:**
- Create: `Packages/VMLXRuntime/Sources/VMLXRuntime/Integration/VMLXService.swift`

Conforms to ToolCapableService. Drop-in for MLXService with id "vmlx".

### Task 9.4: Wire Into Osaurus

**Files:**
- Modify: `Packages/OsaurusCore/Sources/Services/Chat/ChatEngine.swift`

Replace MLXService.shared with VMLXService.shared in service array. Everything else keeps working.

---

## Phase 10: Extended Capabilities (Post-MVP)

### Task 10.1: Image Generation
Port Flux/Z-Image/FIBO. Endpoint: `POST /v1/images/generations`

### Task 10.2: Audio (TTS/STT)
Port Kokoro TTS + Whisper STT. Endpoints: `/v1/audio/speech`, `/v1/audio/transcriptions`

### Task 10.3: Embeddings & Reranking
Endpoints: `POST /v1/embeddings`, `POST /v1/rerank`

### Task 10.4: Anthropic Messages API
Port anthropic_adapter. Endpoint: `POST /v1/messages`

### Task 10.5: Model Config Registry
Port 65+ model family configs. Auto-detect reasoning/tool formats per model.

---

## Dependency Graph

```
Phase 1 (Foundation)
  |-- Task 1.1: Package scaffold
  |-- Task 1.2: LayerCache + HybridCache  <-- EVERYTHING depends on this
  +-- Task 1.3: Core types (Request, SamplingParams)

Phase 2 (Cache Stack) <-- depends on Phase 1
  |-- Task 2.1: CacheBlock + FreeBlockQueue
  |-- Task 2.2: BlockHashMap + BlockTable
  |-- Task 2.3: PagedCacheManager  <-- depends on 2.1, 2.2
  |-- Task 2.4: PrefixCache (trie)
  |-- Task 2.5: MemoryCache
  |-- Task 2.6: DiskCache (L2 SSD)
  |-- Task 2.7: TQDiskStore  <-- depends on Phase 3.2
  |-- Task 2.8: SSMStateCache
  +-- Task 2.9: CacheCoordinator  <-- depends on all above

Phase 3 (TurboQuant) <-- depends on Phase 1
  |-- Task 3.1: TurboQuantConfig
  |-- Task 3.2: EncodedKeys + EncodedValues
  |-- Task 3.3: TurboQuantKVCache  <-- depends on 3.1, 3.2
  +-- Task 3.4: TurboQuant encode/decode  <-- depends on 3.1, 3.2, 3.3

Phase 4 (JANG) <-- depends on Phase 3
  +-- Task 4.1: JangLoader

Phase 5 (Scheduler) <-- depends on Phase 2
  |-- Task 5.1: SchedulerConfig
  |-- Task 5.2: RequestQueue
  |-- Task 5.3: Scheduler  <-- depends on 5.1, 5.2, Phase 2.9
  |-- Task 5.4: BatchBuilder
  +-- Task 5.5: MLLMScheduler  <-- depends on Phase 7

Phase 6 (Generation) <-- depends on Phase 3, 5
  |-- Task 6.1: Sampler
  |-- Task 6.2: StopSequenceDetector
  |-- Task 6.3: StreamAccumulator  <-- depends on Phase 8
  +-- Task 6.4: GenerationEngine  <-- depends on all above

Phase 7 (Vision) <-- depends on Phase 1
  |-- Task 7.1: VisionProcessor
  |-- Task 7.2: VisionEmbeddingCache
  +-- Task 7.3: VLMModelWrapper

Phase 8 (Parsers) <-- depends on Phase 1
  |-- Task 8.1: ToolCallParser protocol
  |-- Task 8.2: 14 tool parser implementations
  +-- Task 8.3: ReasoningParser + 4 implementations

Phase 9 (Integration) <-- depends on Phase 6
  |-- Task 9.1: ChatMessageMapper
  |-- Task 9.2: VMLXRuntimeActor
  |-- Task 9.3: VMLXService (ToolCapableService)
  +-- Task 9.4: Wire into Osaurus

Phase 10 (Extended) <-- depends on Phase 9
  |-- Task 10.1: Image Generation
  |-- Task 10.2: Audio TTS/STT
  |-- Task 10.3: Embeddings + Reranking
  |-- Task 10.4: Anthropic Messages API
  +-- Task 10.5: Model Config Registry
```

## Parallelization Opportunities

These phases/tasks can run concurrently on separate worktrees:

| Track A (Cache) | Track B (Quantization) | Track C (Parsers) | Track D (Vision) |
|---|---|---|---|
| Phase 2.1-2.6 | Phase 3.1-3.4 | Phase 8.1-8.3 | Phase 7.1-7.3 |
| Phase 2.7 (needs 3.2) | Phase 4.1 | | |
| Phase 2.8-2.9 | | | |

After all tracks complete: Phase 5 (Scheduler) -> Phase 6 (Generation) -> Phase 9 (Integration)

## SSM Re-Derivation Architecture (Thinking Models + Hybrid SSM)

### The Problem

In VMLX Python, SSM companion caching is SKIPPED entirely for thinking models (`gen_prompt_len > 0`). The SSM state after generation is "contaminated" by gen_prompt + output tokens, making it unsafe to store at the pre-gen_prompt cache key. Result: every turn of a thinking model with hybrid SSM does full prefill, even with cached KV blocks.

### Solution: Mid-Prefill SSM Checkpointing + Async Re-Derive

**Core Insight:** Checkpoint SSM state at the STABLE BOUNDARY (before gen_prompt_len) during prefill, not after generation. This checkpoint is safe because it matches the truncated KV cache key.

#### Prefill Flow for Thinking Hybrid Models

```
Tokens: [system, history, ..., <|im_start|>assistant\n<think>\n]
                                 ^                              ^
                                 stable_boundary                gen_prompt starts

Phase 1: Forward pass tokens[0:stable_boundary] through all layers
  - Attention layers: compute K,V, store in paged cache
  - SSM layers: compute cumulative state
  - CHECKPOINT: snapshot SSM state here
  - Store SSM checkpoint keyed by hash(tokens[0:stable_boundary])

Phase 2: Continue forward pass tokens[stable_boundary:] (gen_prompt)
  - SSM state advances past checkpoint (now "contaminated")
  - This state is NOT stored
  - Generation proceeds normally

Next turn (same conversation):
  - KV cache hit at stable boundary (paged blocks)
  - SSM checkpoint hit at stable boundary
  - Combine into full HybridCache
  - Only process new turn's gen_prompt through all layers
  - SSM correctly continues from clean checkpoint
```

#### Async Re-Derive (SSM Evicted)

When SSM checkpoint has been evicted but KV blocks still exist:

1. Token sequence is known (stored with KV block metadata)
2. Dispatch async re-derive: run model forward on cached tokens
3. All layers must run (each depends on previous output) -- cost = full prefill
4. During re-derive, attention KV gets TQ-compressed and cached (double duty)
5. Checkpoint SSM at stable boundary
6. Store both refreshed KV and SSM checkpoint

The async aspect is key: current request can proceed with full prefill while re-derive runs in background. Once complete, ALL subsequent requests benefit.

#### CacheCoordinator Fetch Cascade (Updated for SSM Re-Derive)

```
1. PagedCache.fetch(tokens) -> attention KV blocks (TQ compressed)
   - MISS -> full prefill, no re-derive needed

2. If attention HIT and model is hybrid:
   a. SSMStateCache.fetch(tokenHash, boundary) -> SSM checkpoint?
      - HIT -> combine KV + SSM = full HybridCache, skip prefill

   b. SSMDiskCheckpoint.fetch(tokenHash) -> SSM checkpoint on disk?
      - HIT -> load, promote to memory, combine = full HybridCache

   c. MISS -> decision point:
      - Estimate re-derive cost (proportional to token count)
      - If tokens < threshold (e.g., 512): sync re-derive, wait
      - If tokens >= threshold: full prefill now + dispatch async re-derive
      - Async re-derive stores SSM checkpoint for next request
```

#### Types

```swift
/// SSM state snapshot at a stable boundary (before gen_prompt_len).
/// Safe to store/fetch because key matches truncated KV cache key.
struct SSMCheckpoint: Sendable {
    let ssmStates: [SSMStateLayer]   // One per SSM layer in model
    let boundary: Int                 // Token position of checkpoint
    let tokenHash: String             // SHA-256 of tokens[:boundary]
    let timestamp: Date
}

/// Manages async re-derivation of SSM state from token sequences.
actor SSMReDeriver {
    /// Checkpoint SSM state during prefill at stable boundary.
    func checkpoint(
        ssmStates: [SSMStateLayer],
        tokens: [Int],
        boundary: Int
    ) -> SSMCheckpoint

    /// Async re-derive: run model forward to recover SSM state.
    /// Returns checkpoint at stableBoundary.
    /// Also refreshes attention KV cache (TQ compressed) as side effect.
    func reDerive(
        tokens: [Int],
        stableBoundary: Int,
        model: ModelContainer,
        cacheCoordinator: CacheCoordinator
    ) async throws -> SSMCheckpoint

    /// Check if re-derive is already in progress for this token hash.
    func isReDeriving(tokenHash: String) -> Bool
}
```

#### TurboQuant Interaction

- TQ compresses attention KV cache (3-bit) -- SSM state is NEVER TQ'd
- During re-derive, fresh attention KV gets TQ compressed before storage
- SSM checkpoints are stored independently (small relative to KV, not worth compressing)
- On fetch: TQ attention KV decompresses on-the-fly, SSM checkpoint loads directly
- TQ recompress after prefill Phase 1 SKIPS SSM layers (uses layerPattern)

#### Why This Is Better Than VMLX Python

| Aspect | VMLX Python (current) | VMLXRuntime Swift (new) |
|--------|----------------------|------------------------|
| Thinking + hybrid SSM | SSM storage SKIPPED, full prefill every turn | Mid-prefill checkpoint, instant SSM hit |
| SSM eviction | Full prefill, no recovery | Async re-derive in background |
| Multi-turn thinking | O(n) prefill per turn | O(gen_prompt_len) per turn after first |
| TQ + SSM | TQ only on attention, SSM state lost | TQ attention + SSM checkpoint, both preserved |

---

## Critical Invariants

These rules must hold at all times. Violations caused production crashes in VMLX:

1. **SSM state is path-dependent.** Never truncate, slice, or skip prefix computation for SSM layers. The `HybridCache.canTruncate` gate enforces this.

2. **Materialize before caching.** Call `MLXArray.eval()` on all arrays after generation before storing in cache. Without this, the lazy computation graph replays on next access (full re-prefill). `MLXArray.eval()` forces lazy Metal GPU computation to execute, converting the computation graph into concrete GPU buffer values. This is NOT arbitrary code execution -- it is the MLX framework's standard mechanism for materializing lazy tensor operations.

3. **Pre-serialize on calling thread.** Metal GPU operations MUST happen on the thread that owns the Metal context. Disk cache background writers must receive pre-serialized data, never raw MLXArrays.

4. **TQ recompress skips SSM layers.** Only attention layers have KV to compress. The `TurboQuantConfig.keyBits(forLayer:)` returns nil for SSM layers.

5. **Paged block ref count on abort.** When a request is aborted, call `deleteBlockTable()` to decrement ref counts. Leaking refs = OOM.

6. **SSM companion fetch: empty == MISS.** An empty `[MLXArray]` is a miss, not just `nil`. Bug from ba07392.

7. **Two-phase prefill for non-trimmable caches.** If `HybridCache.canTruncate == false`, use the two-phase approach: prefill stable boundary, snapshot, then feed generation prefix separately.

8. **MLA head count awareness.** For models with `kv_lora_rank > 0` (DeepSeek MLA), key_dim = `qk_nope_head_dim + qk_rope_head_dim`, value_dim = `v_head_dim`. TQ must use these, not the standard head_dim.

9. **Mid-prefill SSM checkpoint for thinking models.** When `gen_prompt_len > 0` on a hybrid model, checkpoint SSM state at `tokens[:stable_boundary]` DURING prefill (Phase 1), not after generation. Post-generation SSM state is contaminated by gen_prompt + output tokens and MUST NOT be stored.

10. **Async re-derive is full forward pass.** SSM layers cannot run independently — each layer depends on the previous layer's output. Re-derive cost = full prefill cost. The benefit is amortization across turns and background execution.
