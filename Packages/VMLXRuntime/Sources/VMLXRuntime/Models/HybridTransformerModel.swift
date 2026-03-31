import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Hybrid Block

/// A hybrid transformer block -- either attention or SSM based on layer type.
/// Used in models like Nemotron-H, Jamba, and Qwen3.5-A3B that interleave
/// attention and Mamba layers.
public enum HybridBlock {
    case attention(TransformerBlock)
    case ssm(MambaResidualBlock)
}

// MARK: - HybridTransformerConfig

/// Extended configuration for hybrid transformer models.
/// Combines transformer config (for attention layers) with Mamba config (for SSM layers).
public struct HybridTransformerConfig: Sendable {
    /// Base transformer config (used for attention layers and shared parameters).
    public let transformer: TransformerConfig

    /// Mamba config (used for SSM layers).
    public let mamba: MambaConfig

    /// Per-layer type pattern describing which layers are attention vs SSM.
    /// Length must equal transformer.numLayers.
    public let layerPattern: [LayerType]

    public init(transformer: TransformerConfig, mamba: MambaConfig, layerPattern: [LayerType]) {
        self.transformer = transformer
        self.mamba = mamba
        self.layerPattern = layerPattern
    }

    /// Number of attention layers in the pattern.
    public var numAttentionLayers: Int {
        layerPattern.filter { $0 == .attention || $0 == .expert }.count
    }

    /// Number of SSM layers in the pattern.
    public var numSSMLayers: Int {
        layerPattern.filter { $0 == .ssm }.count
    }

    /// Parse from a model's config.json dictionary.
    /// Requires the config to contain hybrid_override_pattern or a known hybrid family
    /// pattern string that can be parsed into layer types.
    public static func from(config: [String: Any]) -> HybridTransformerConfig? {
        let transformerCfg = TransformerConfig.from(config: config)

        // Parse Mamba config from the same dict (or nested ssm_cfg)
        guard let mambaCfg = MambaConfig.from(config: config) else {
            return nil
        }

        // Get layer pattern from config
        let pattern: [LayerType]
        if let patternStr = config["hybrid_override_pattern"] as? String {
            // Repeat pattern to fill all layers
            let basePattern = parseHybridPattern(patternStr)
            if basePattern.isEmpty {
                return nil
            }
            pattern = expandPattern(basePattern, to: transformerCfg.numLayers)
        } else if let patternArray = config["layer_types"] as? [String] {
            // Explicit per-layer type array
            pattern = patternArray.map { str -> LayerType in
                switch str.lowercased() {
                case "mamba", "ssm", "m":
                    return .ssm
                case "attention", "attn", "*":
                    return .attention
                case "expert", "moe", "e":
                    return .expert
                default:
                    return .attention
                }
            }
        } else {
            // Default: all attention (not a hybrid model)
            return nil
        }

        guard pattern.count == transformerCfg.numLayers else {
            return nil
        }

        return HybridTransformerConfig(
            transformer: transformerCfg,
            mamba: mambaCfg,
            layerPattern: pattern
        )
    }

    /// Expand a base pattern to fill `targetCount` layers by repeating.
    /// e.g., [.ssm, .ssm, .ssm, .attention] repeated to fill 32 layers.
    private static func expandPattern(_ base: [LayerType], to targetCount: Int) -> [LayerType] {
        guard !base.isEmpty else { return [] }
        var result: [LayerType] = []
        while result.count < targetCount {
            result.append(contentsOf: base)
        }
        return Array(result.prefix(targetCount))
    }
}

// MARK: - HybridTransformerModelInner

/// The inner `model` wrapper for hybrid architectures.
/// Maps to `model.embed_tokens`, `model.layers`, `model.norm`.
///
/// Each layer is either a TransformerBlock (attention) or MambaResidualBlock (SSM)
/// based on the layer pattern from the config.
public class HybridTransformerModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let blocks: [HybridBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    /// Indices of attention layers (for KV cache mapping).
    let attentionIndices: [Int]
    /// Indices of SSM layers (for SSM state mapping).
    let ssmIndices: [Int]

    public init(_ config: HybridTransformerConfig) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.transformer.vocabSize,
            dimensions: config.transformer.hiddenSize
        )

        var attnIdx: [Int] = []
        var ssmIdx: [Int] = []
        var layerBlocks: [HybridBlock] = []

        for (i, layerType) in config.layerPattern.enumerated() {
            switch layerType {
            case .attention, .expert:
                layerBlocks.append(.attention(TransformerBlock(config.transformer)))
                attnIdx.append(i)
            case .ssm:
                layerBlocks.append(.ssm(MambaResidualBlock(config.mamba)))
                ssmIdx.append(i)
            }
        }

        self.blocks = layerBlocks
        self.attentionIndices = attnIdx
        self.ssmIndices = ssmIdx

        self._norm.wrappedValue = RMSNorm(
            dimensions: config.transformer.hiddenSize,
            eps: config.transformer.rmsNormEps
        )
    }

    /// Forward pass: embed -> N hybrid layers -> final norm.
    ///
    /// - Parameters:
    ///   - inputIds: token IDs [B, seqLen]
    ///   - mask: optional attention mask (only used by attention layers)
    ///   - kvCaches: per-attention-layer KV caches (indexed by attentionIndices order)
    ///   - ssmStates: per-SSM-layer states (indexed by ssmIndices order)
    /// - Returns: hidden states [B, seqLen, hiddenSize]
    public func callAsFunction(
        _ inputIds: MLXArray,
        mask: MLXArray? = nil,
        kvCaches: [KVCache]? = nil,
        ssmStates: [MambaState]? = nil
    ) -> MLXArray {
        // Position offset from existing attention cache
        let offset: Int
        if let caches = kvCaches, !caches.isEmpty {
            offset = caches[0].sequenceLength
        } else {
            offset = 0
        }

        var h = embedTokens(inputIds)
        var attnCacheIdx = 0
        var ssmStateIdx = 0

        for (_, block) in blocks.enumerated() {
            switch block {
            case .attention(let attnBlock):
                let cache = kvCaches?[attnCacheIdx]
                h = attnBlock(h, mask: mask, cache: cache, offset: offset)
                attnCacheIdx += 1
            case .ssm(let ssmBlock):
                let state = ssmStates?[ssmStateIdx] ?? MambaState()
                h = ssmBlock(h, state: state)
                ssmStateIdx += 1
            }
        }

        return norm(h)
    }
}

// MARK: - HybridTransformerModel

/// Complete hybrid decoder-only model with interleaved attention and SSM layers.
///
/// Extends the standard transformer architecture to support mixed layer types.
/// Attention layers use TransformerBlock with KV caching.
/// SSM layers use MambaResidualBlock with cumulative state.
///
/// Key path hierarchy matches HuggingFace weight names:
/// ```
/// model.embed_tokens.weight
/// model.layers.0.self_attn.{q,k,v,o}_proj.weight    (attention layers)
/// model.layers.0.mlp.{gate,up,down}_proj.weight      (attention layers)
/// model.layers.0.mixer.in_proj.weight                 (SSM layers)
/// model.layers.0.mixer.conv1d.weight                  (SSM layers)
/// model.layers.0.norm.weight                          (both)
/// model.norm.weight
/// lm_head.weight
/// ```
public class HybridTransformerModel: Module {

    @ModuleInfo(key: "model") var model: HybridTransformerModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: HybridTransformerConfig
    public let layerPattern: [LayerType]

    public init(_ config: HybridTransformerConfig) {
        self.config = config
        self.layerPattern = config.layerPattern
        self._model.wrappedValue = HybridTransformerModelInner(config)
        self._lmHead.wrappedValue = Linear(
            config.transformer.hiddenSize,
            config.transformer.vocabSize,
            bias: false
        )
    }

    /// Run the full model: embedding -> hybrid layers -> norm -> LM head.
    public func callAsFunction(
        _ inputIds: MLXArray,
        mask: MLXArray? = nil,
        kvCaches: [KVCache]? = nil,
        ssmStates: [MambaState]? = nil
    ) -> MLXArray {
        let hidden = model(inputIds, mask: mask, kvCaches: kvCaches, ssmStates: ssmStates)
        return lmHead(hidden)
    }

    /// Create a causal attention mask (delegates to TransformerModel).
    public static func createCausalMask(seqLen: Int, offset: Int = 0, dtype: DType = .float16) -> MLXArray {
        TransformerModel.createCausalMask(seqLen: seqLen, offset: offset, dtype: dtype)
    }

    /// Create empty KV caches for attention layers only.
    public func createKVCaches() -> [KVCache] {
        (0..<config.numAttentionLayers).map { _ in KVCache() }
    }

    /// Create empty MambaStates for SSM layers only.
    public func createSSMStates() -> [MambaState] {
        (0..<config.numSSMLayers).map { _ in MambaState() }
    }

    /// Load weights from a flat dictionary (e.g., from safetensors).
    public func loadWeights(_ weights: [String: MLXArray]) {
        // Handle conv1d weight transposition (some checkpoints have [out, width, in] vs [out, in, width])
        var sanitized = weights
        for (key, value) in sanitized {
            if key.contains("conv1d.weight") && value.ndim == 3 && value.dim(2) != 1 {
                sanitized[key] = value.transposed(0, 2, 1)
            }
        }

        let parameters = ModuleParameters.unflattened(sanitized)
        self.update(parameters: parameters)
    }

    /// Number of attention layers.
    public var numAttentionLayers: Int { config.numAttentionLayers }

    /// Number of SSM layers.
    public var numSSMLayers: Int { config.numSSMLayers }

    /// Whether this model has any SSM layers.
    public var hasSSMLayers: Bool { config.numSSMLayers > 0 }
}

