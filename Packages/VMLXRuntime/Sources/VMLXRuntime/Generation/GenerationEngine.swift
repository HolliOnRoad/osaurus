import Foundation
import MLX

/// Protocol for the actual model forward pass.
/// Implemented by the model loader when a real model is available.
public protocol ModelForwardPass: Sendable {
    /// Run prefill: process multiple tokens, populate cache.
    /// Returns logits for the last token.
    func prefill(inputIds: MLXArray, cache: inout [MLXArray], mask: MLXArray?) async throws -> MLXArray

    /// Run decode: process a single token, update cache.
    /// Returns logits for the next token.
    func decode(tokenId: Int, cache: inout [MLXArray]) async throws -> MLXArray

    /// Get the model's vocabulary size.
    var vocabSize: Int { get }

    /// Get the model's layer count.
    var layerCount: Int { get }
}

/// Configuration for a generation run.
public struct GenerationConfig: Sendable {
    /// Sampling parameters.
    public let samplingParams: SamplingParams

    /// Maximum tokens to generate.
    public let maxTokens: Int

    /// Stop sequences.
    public let stopSequences: [String]

    /// Whether this model is hybrid (SSM+attention).
    public let isHybrid: Bool

    /// Whether TurboQuant is active.
    public let enableTQ: Bool

    /// TurboQuant configuration (if TQ active).
    public let tqConfig: TurboQuantConfig?

    /// Whether thinking/reasoning mode is enabled.
    public let enableThinking: Bool

    /// Generation prompt length (tokens injected by chat template, e.g., <think>).
    /// Used for mid-prefill SSM checkpointing.
    public let genPromptLen: Int

    public init(
        samplingParams: SamplingParams = SamplingParams(),
        maxTokens: Int = 2048,
        stopSequences: [String] = [],
        isHybrid: Bool = false,
        enableTQ: Bool = false,
        tqConfig: TurboQuantConfig? = nil,
        enableThinking: Bool = false,
        genPromptLen: Int = 0
    ) {
        self.samplingParams = samplingParams
        self.maxTokens = maxTokens
        self.stopSequences = stopSequences
        self.isHybrid = isHybrid
        self.enableTQ = enableTQ
        self.tqConfig = tqConfig
        self.enableThinking = enableThinking
        self.genPromptLen = genPromptLen
    }
}

/// Result of a generation run.
public struct GenerationResult: Sendable {
    /// Generated token IDs.
    public let tokenIds: [Int]

    /// Generated text.
    public let text: String

    /// Finish reason.
    public let finishReason: FinishReason

    /// Number of prompt tokens (after cache deduction).
    public let promptTokens: Int

    /// Number of tokens from cache.
    public let cachedTokens: Int

    /// Number of generated tokens.
    public let completionTokens: Int

    /// Cache detail (which layer hit).
    public let cacheDetail: CacheDetail

    /// SSM checkpoint (if hybrid model, captured at stable boundary).
    public let ssmCheckpoint: SSMCheckpoint?
}

/// The core generation engine.
/// Orchestrates: cache lookup -> prefill -> TQ compress -> decode loop -> cache store.
///
/// Two-phase prefill for hybrid thinking models:
///   Phase 1: tokens[0:stable_boundary] -> SSM checkpoint
///   Phase 2: tokens[stable_boundary:] (gen_prompt) -> continue to decode
public struct GenerationEngine: Sendable {

    /// Run generation with a model, cache, and configuration.
    /// Returns a stream of events (tokens, thinking, tool calls).
    ///
    /// - Parameter model: The model to use for forward passes. When `nil`, the engine
    ///   runs in stub mode (no actual generation) — useful for testing cache/scheduling logic.
    public static func generate(
        model: (any ModelForwardPass)? = nil,
        promptTokenIds: [Int],
        existingCache: HybridCache?,
        cachedTokenCount: Int,
        config: GenerationConfig,
        cacheCoordinator: CacheCoordinator,
        accumulator: inout StreamAccumulator,
        onEvent: @Sendable (StreamEvent) -> Void
    ) async throws -> GenerationResult {

        let uncachedTokens: [Int]
        let cacheDetail: CacheDetail

        if let _ = existingCache, cachedTokenCount > 0 {
            uncachedTokens = Array(promptTokenIds[cachedTokenCount...])
            cacheDetail = .prefix  // Simplified — coordinator determines actual detail
        } else {
            uncachedTokens = promptTokenIds
            cacheDetail = .full
        }

        // Determine stable boundary for SSM checkpoint (thinking models)
        let stableBoundary: Int?
        if config.isHybrid && config.enableThinking && config.genPromptLen > 0 {
            // Stable boundary = total prompt minus gen_prompt_len
            stableBoundary = promptTokenIds.count - config.genPromptLen
        } else {
            stableBoundary = nil
        }

        // ---- PHASE 1: Prefill ----
        // In production: run model.prefill() on uncachedTokens
        // For hybrid models with thinking: checkpoint SSM at stableBoundary

        var ssmCheckpoint: SSMCheckpoint? = nil

        if let boundary = stableBoundary, config.isHybrid {
            // Two-phase prefill:
            // Phase 1: prefill tokens[0:boundary] (stable content)
            // -> Checkpoint SSM state here (safe, no gen_prompt contamination)
            // Phase 2: prefill tokens[boundary:] (gen_prompt)
            // -> SSM state after this is "contaminated", don't store

            let stableTokens = Array(promptTokenIds.prefix(boundary))
            let tokenHash = SSMStateCache.hashTokens(stableTokens, count: boundary)

            // TODO: After actual prefill phase 1, capture SSM states
            // ssmCheckpoint = SSMCheckpoint(ssmStates: extractedSSMStates, boundary: boundary, tokenHash: tokenHash)
            // cacheCoordinator.storeSSMCheckpoint(ssmCheckpoint!)

            // Placeholder checkpoint
            ssmCheckpoint = SSMCheckpoint(ssmStates: [], boundary: boundary, tokenHash: tokenHash)
        }

        // ---- TQ Recompress (after prefill, before decode) ----
        if config.enableTQ, let tqConfig = config.tqConfig {
            // Compress attention KV cache to TQ format
            // Skip SSM layers (tqConfig.keyBits returns nil for SSM)
            // TODO: Iterate attention layers, compress each via TurboQuantKVCache.compress()
            _ = tqConfig  // Acknowledge config
        }

        // ---- PHASE 2: Decode Loop ----
        let generatedTokenIds: [Int] = []
        let generatedText = ""
        let finishReason: FinishReason = .length

        for _ in 0..<config.maxTokens {
            // TODO: In production, call model.decode() to get logits
            // let logits = try await model.decode(tokenId: lastToken, cache: &cache)
            // let tokenId = Sampler.sample(logits: logits, params: config.samplingParams, previousTokens: generatedTokenIds)

            // Placeholder: no actual generation without a model
            // This will be filled when model loading is implemented
            break
        }

        // ---- Finalize ----
        let finalEvents = accumulator.finalize()
        for event in finalEvents {
            onEvent(event)
        }

        // ---- Store cache ----
        // After generation, materialize cache and store
        // if let cache = generatedCache {
        //     let materialized = cache.materialized()
        //     cacheCoordinator.store(tokens: promptTokenIds + generatedTokenIds, cache: materialized)
        // }

        return GenerationResult(
            tokenIds: generatedTokenIds,
            text: generatedText,
            finishReason: finishReason,
            promptTokens: uncachedTokens.count,
            cachedTokens: cachedTokenCount,
            completionTokens: generatedTokenIds.count,
            cacheDetail: cacheDetail,
            ssmCheckpoint: ssmCheckpoint
        )
    }

    /// Common prefix detection between new prompt tokens and cached tokens.
    /// Returns the number of tokens that match.
    public static func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
        let minLen = min(a.count, b.count)
        for i in 0..<minLen {
            if a[i] != b[i] { return i }
        }
        return minLen
    }
}
