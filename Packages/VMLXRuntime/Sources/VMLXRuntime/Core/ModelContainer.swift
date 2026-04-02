import Foundation
import MLX
import MLXNN
import Tokenizers

public struct VMLXLiveCacheConfig: Sendable {
    public let kvBits: Int?
    public let kvGroupSize: Int
    public let turboQuantConfig: TurboQuantConfig?

    public init(
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        turboQuantConfig: TurboQuantConfig? = nil
    ) {
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.turboQuantConfig = turboQuantConfig
    }
}

/// Container wrapping a loaded model with runtime configuration.
/// This is what gets passed around during inference.
public final class VMLXModelContainer: @unchecked Sendable {

    /// The loaded model (weights + tokenizer + config).
    public let model: LoadedModel

    /// The native model for direct forward pass access.
    public let nativeModel: any VMLXNativeModel & Module

    /// Tokenizer reference.
    public let tokenizer: any Tokenizer

    /// Model name/path.
    public let name: String

    /// Whether this is a JANG model.
    public var isJang: Bool { model.detected.isJang }

    /// Whether this is a hybrid model (SSM + attention).
    public var isHybrid: Bool { model.detected.isHybrid }

    /// Whether this model supports vision.
    public var hasVision: Bool { model.detected.hasVision }

    /// TurboQuant config (nil if not a JANG model or TQ disabled).
    public let turboQuantConfig: TurboQuantConfig?

    /// Layer pattern for hybrid models.
    public let layerPattern: [LayerType]?

    /// EOS token IDs for stop detection.
    public let eosTokenIds: Set<Int>

    /// Model family config (tool format, reasoning format, etc.).
    public let familyConfig: ModelFamilyConfig

    /// Map config.json layer_type strings to LayerType enum.
    /// Handles Qwen3.5 ("linear_attention"/"full_attention"),
    /// Nemotron-H ("mamba"/"attention"), Gemma 4 ("sliding_attention"),
    /// and generic variants.
    private static func parseLayerTypeString(_ str: String) -> LayerType {
        let lower = str.lowercased()
        switch lower {
        case "full_attention", "sliding_attention", "attention", "attn", "self_attention":
            return .attention
        case "linear_attention", "ssm", "mamba", "recurrent", "gated_delta":
            return .ssm
        case "expert", "moe":
            return .expert
        default:
            return .attention
        }
    }

    private init(model: LoadedModel,
                 turboQuantConfig: TurboQuantConfig?, layerPattern: [LayerType]?) {
        self.model = model
        self.nativeModel = model.nativeModel
        self.tokenizer = model.tokenizer
        self.name = model.detected.name
        self.eosTokenIds = model.eosTokenIds
        // Use config.json model_type for family detection — no name matching
        if let modelType = model.detected.modelType,
           let config = ModelConfigRegistry.configForModelType(modelType) {
            self.familyConfig = config
        } else {
            // Fallback: generic config (unknown model type)
            self.familyConfig = ModelFamilyConfig(family: "generic", toolCallFormat: .generic, defaultContextWindow: 8192)
        }
        self.turboQuantConfig = turboQuantConfig
        self.layerPattern = layerPattern
    }

    /// Factory method. Builds TQ/hybrid configuration from detected model properties.
    ///
    /// TurboQuant is available for ALL models going through VMLXRuntime, not just JANG.
    /// - JANG models: TQ config is built from the JANG quantization profile (custom bit widths)
    /// - Non-JANG models: TQ config uses sensible defaults (3-bit keys/values, 4-bit critical layers)
    /// The user must still enable TQ via settings (`enableTurboQuant`) for it to activate at runtime.
    public static func create(model: LoadedModel) -> VMLXModelContainer {
        // Metal kernels DISABLED — crash with EXC_BAD_ACCESS even during serialized
        // single-threaded prewarm. Root cause is a kernel template issue, NOT concurrency.
        // Prewarm skipped until kernel bug is resolved.
        // vmlxPrewarmCustomKernels()

        // Detect layer pattern (hybrid architecture) from config.json
        let detectedLayerPattern: [LayerType]?
        if let patternStr = model.detected.hybridOverridePattern {
            detectedLayerPattern = parseHybridPattern(patternStr)
        } else if let layerTypeStrs = model.detected.layerTypes {
            detectedLayerPattern = layerTypeStrs.map { parseLayerTypeString($0) }
        } else {
            detectedLayerPattern = nil
        }

        // Build TQ config
        let turboQuantConfig: TurboQuantConfig?
        if model.detected.isJang,
           JangLoader.isJangModel(at: model.detected.modelPath),
           let jangConfig = try? JangLoader.loadConfig(at: model.detected.modelPath) {
            // JANG models: custom TQ config from quantization profile
            turboQuantConfig = JangLoader.buildTQConfig(
                from: jangConfig,
                layerPattern: detectedLayerPattern,
                kvLoraRank: model.detected.kvLoraRank,
                qkNopeHeadDim: model.detected.qkNopeHeadDim,
                qkRopeHeadDim: model.detected.qkRopeHeadDim
            )
        } else {
            // Non-JANG MLX models: default TQ config with sensible defaults.
            // Includes MLA dimensions from config.json if available (Mistral4, DeepSeek).
            var defaultTQ = TurboQuantConfig(layerPattern: detectedLayerPattern)
            if let rank = model.detected.kvLoraRank, rank > 0 {
                if let nope = model.detected.qkNopeHeadDim, let rope = model.detected.qkRopeHeadDim {
                    defaultTQ.mlaKeyDim = nope + rope
                }
            }
            turboQuantConfig = defaultTQ
        }

        return VMLXModelContainer(
            model: model,
            turboQuantConfig: turboQuantConfig,
            layerPattern: detectedLayerPattern
        )
    }

    // MARK: - Tokenization

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    /// Decode token IDs to text.
    public func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }

    /// Apply chat template to messages and encode.
    public func applyChatTemplate(
        messages: [VMLXChatMessage],
        addGenerationPrompt: Bool = true,
        enableThinking: Bool = true,
        reasoningEffort: String? = nil
    ) throws -> [Int] {
        let chatMessages: [Message] = messages.map { msg in
            ["role": msg.role, "content": msg.textContent]
        }

        if tokenizer.hasChatTemplate {
            var context: [String: any Sendable] = ["enable_thinking": enableThinking]
            if let effort = reasoningEffort {
                context["reasoning_effort"] = effort
            }
            // Auto-map enable_thinking → reasoning_effort for Mistral 4.
            // Mistral 4's template uses reasoning_effort ("none"/"high"), not enable_thinking.
            // Matches Python VMLX server.py behavior.
            if context["reasoning_effort"] == nil {
                if enableThinking {
                    context["reasoning_effort"] = "high"
                } else {
                    context["reasoning_effort"] = "none"
                }
            }
            return try tokenizer.applyChatTemplate(
                messages: chatMessages,
                chatTemplate: nil,
                addGenerationPrompt: addGenerationPrompt,
                truncation: false,
                maxLength: nil,
                tools: nil,
                additionalContext: context
            )
        }

        let fullText = messages.map { msg in
            "\(msg.role): \(msg.textContent)"
        }.joined(separator: "\n")

        return encode(fullText)
    }

    /// Compute gen_prompt_len: difference between encoding with and without generation prompt.
    public func computeGenPromptLen(messages: [VMLXChatMessage]) -> Int {
        guard tokenizer.hasChatTemplate else { return 0 }
        do {
            let withGen = try applyChatTemplate(messages: messages, addGenerationPrompt: true)
            let withoutGen = try applyChatTemplate(messages: messages, addGenerationPrompt: false)
            return max(withGen.count - withoutGen.count, 0)
        } catch {
            return 0
        }
    }

    // MARK: - Inference

    /// Run the model forward pass (tokens in, logits out).
    public func forward(_ tokens: MLXArray, cache: [VMLXKVCache]?) -> MLXArray {
        nativeModel(tokens, cache: cache)
    }

    /// Create fresh caches for inference.
    public func newCache() -> [VMLXKVCache] {
        nativeModel.newCache()
    }

    /// Create fresh caches with the requested live-cache policy.
    /// TurboQuant takes precedence over q4/q8 live KV quantization on layers where
    /// the model config declares TurboQuant-eligible attention KV.
    public func newCache(config: VMLXLiveCacheConfig) -> [VMLXKVCache] {
        let baseCaches = nativeModel.newCache()
        return baseCaches.enumerated().map { index, cache in
            if let turboQuantConfig = config.turboQuantConfig,
               cache is VMLXKVCacheSimple,
               turboQuantConfig.keyBits(forLayer: index, totalLayers: baseCaches.count) != nil,
               turboQuantConfig.valueBits(forLayer: index, totalLayers: baseCaches.count) != nil {
                return TurboQuantKVCache(
                    config: turboQuantConfig,
                    layerIndex: index,
                    totalLayers: baseCaches.count
                )
            }

            if let bits = config.kvBits, bits < 16, cache is VMLXKVCacheSimple {
                return VMLXQuantizedKVCache(bits: bits, groupSize: config.kvGroupSize)
            }

            return cache
        }
    }

    /// Create fresh caches with KV quantization applied.
    /// Replaces VMLXKVCacheSimple with VMLXQuantizedKVCache when kvBits < 16.
    /// SSM caches (VMLXMambaCache) are not affected — SSM state is not quantizable.
    public func newCache(kvBits: Int?, kvGroupSize: Int) -> [VMLXKVCache] {
        newCache(config: VMLXLiveCacheConfig(kvBits: kvBits, kvGroupSize: kvGroupSize))
    }
}
