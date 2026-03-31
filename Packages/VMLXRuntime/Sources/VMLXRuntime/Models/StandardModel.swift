//
//  StandardModel.swift
//  VMLXRuntime
//
//  Generic decoder-only transformer for standard MLX models:
//  Llama 2/3/4, Qwen 2/2.5/3, Mistral (non-MLA), Gemma 2/3, Phi 3/4,
//  StarCoder 2, InternLM 2, Granite, Cohere, etc.
//
//  Uses VMLXKVCacheSimple for KV caching, conforms to VMLXNativeModel
//  for integration with VMLXRuntimeActor's native generation loop.
//
//  Weight key hierarchy matches HuggingFace naming:
//    model.embed_tokens.weight
//    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//    model.layers.{i}.mlp.{gate,up,down}_proj.weight
//    model.layers.{i}.input_layernorm.weight
//    model.layers.{i}.post_attention_layernorm.weight
//    model.norm.weight
//    lm_head.weight
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct StandardModelConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int?
    var vocabularySize: Int = 32000
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 10000.0
    var maxPositionEmbeddings: Int = 8192
    var headDim: Int?
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var mlpBias: Bool = false
    var useQKNorm: Bool = false
    var ropeScaling: [String: StringOrNumber]?
    var rotaryDim: Int?

    // MoE fields
    var numLocalExperts: Int = 0
    var numExpertsPerTok: Int = 0

    /// Resolved head dimension (computed from config).
    var resolvedHeadDim: Int {
        headDim ?? (hiddenSize / attentionHeads)
    }

    /// Resolved KV heads (defaults to attentionHeads for MHA).
    var resolvedKVHeads: Int {
        kvHeads ?? attentionHeads
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case headDim = "head_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case useQKNorm = "use_qk_norm"
        case ropeScaling = "rope_scaling"
        case rotaryDim = "rotary_dim"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerTok = "num_experts_per_tok"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads)
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 32000
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)

        // Model-type-aware defaults for bias.
        // Qwen2/2.5 has attention_bias=true by default (Q/K/V/O all have .bias).
        // Most other models (Llama, Mistral, Gemma) default to false.
        let biasDefault = (modelType == "qwen2" || modelType == "qwen3")
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? biasDefault
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        useQKNorm = try c.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? false
        rotaryDim = try c.decodeIfPresent(Int.self, forKey: .rotaryDim)
        numLocalExperts = try c.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? 0
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
    }

    /// Resolved RoPE dimensions (some models like MiniMax use partial rotation).
    var resolvedRopeDims: Int {
        rotaryDim ?? resolvedHeadDim
    }

    /// Whether this config describes a Mixture-of-Experts model.
    var isMoE: Bool { numLocalExperts > 0 }
}

// MARK: - Attention

final class StandardAttention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: VMLXRoPELayer

    init(_ args: StandardModelConfiguration) {
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.resolvedKVHeads
        self.headDim = args.resolvedHeadDim
        self.scale = pow(Float(args.resolvedHeadDim), -0.5)

        let hasBias = args.attentionBias

        _qProj.wrappedValue = Linear(
            args.hiddenSize, attentionHeads * headDim, bias: hasBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDim, bias: hasBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDim, bias: hasBias)
        _oProj.wrappedValue = Linear(
            attentionHeads * headDim, args.hiddenSize, bias: hasBias)

        if args.useQKNorm {
            _qNorm.wrappedValue = RMSNorm(dimensions: attentionHeads * headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: kvHeads * headDim, eps: args.rmsNormEps)
        }

        self.rope = vmlxInitializeRope(
            dims: args.resolvedRopeDims,
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: VMLXKVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var qOut = qProj(x)
        var kOut = kProj(x)

        // Apply Q/K normalization BEFORE reshape (operates on full projected dim)
        if let qNorm { qOut = qNorm(qOut) }
        if let kNorm { kOut = kNorm(kOut) }

        var queries = qOut.reshaped(B, L, attentionHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kOut.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let output = vmlxAttentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - Compiled SwiGLU

/// Fused SiLU(gate) * up kernel. Matches Python's @mx.compile(shapeless=True) swiglu.
/// Compiles to a single GPU kernel instead of 2 separate ops (silu + multiply).
let compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) {
    (gate: MLXArray, x: MLXArray) -> MLXArray in
    silu(gate) * x
}

// MARK: - MLP (SwiGLU)

final class StandardMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(_ args: StandardModelConfiguration) {
        let hasBias = args.mlpBias
        _gateProj.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: hasBias)
        _downProj.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: hasBias)
        _upProj.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: hasBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(compiledSwiGLU(gateProj(x), upProj(x)))
    }
}

// MARK: - Sparse MoE Block

/// Standard MoE block for models like MiniMax M2.5.
/// Weight key: `block_sparse_moe.gate` + `block_sparse_moe.switch_mlp`.
final class StandardSparseMoeBlock: Module, UnaryLayer {
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: VMLXSwitchGLU
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ args: StandardModelConfiguration) {
        self.numExperts = args.numLocalExperts
        self.topK = args.numExpertsPerTok

        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([args.numLocalExperts])
        _gate.wrappedValue = Linear(args.hiddenSize, args.numLocalExperts, bias: false)
        _switchMLP.wrappedValue = VMLXSwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.intermediateSize,
            numExperts: args.numLocalExperts
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Cast to float32 for numerical stability in gate routing (matches Python mlx-lm).
        // Prevents float16/bfloat16 overflow in sigmoid for large expert counts.
        let gates = gate(x.asType(.float32))
        let scores = sigmoid(gates)

        // Apply e_score_correction_bias for expert SELECTION (top-k routing),
        // but use original scores (without bias) for weighting.
        let biasedScores = scores + eScoreCorrectionBias

        let k = topK
        let inds = MLX.argPartition(-biasedScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        let selectedScores = MLX.takeAlong(scores, inds, axis: -1)

        // Normalize with epsilon guard, cast back to input dtype
        let normalizedScores = (selectedScores / (selectedScores.sum(axis: -1, keepDims: true) + 1e-20))
            .asType(x.dtype)

        let y = switchMLP(x, inds)
        return (y * normalizedScores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

// MARK: - Decoder Layer

final class StandardDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: StandardAttention
    @ModuleInfo(key: "mlp") var mlp: StandardMLP?
    @ModuleInfo(key: "block_sparse_moe") var moe: StandardSparseMoeBlock?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: StandardModelConfiguration) {
        _selfAttn.wrappedValue = StandardAttention(args)

        if args.isMoE {
            _moe.wrappedValue = StandardSparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = StandardMLP(args)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: VMLXKVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let ffnInput = postAttentionLayerNorm(h)
        if let moe {
            return h + moe(ffnInput)
        } else {
            return h + mlp!(ffnInput)
        }
    }
}

// MARK: - Model Inner (model.embed_tokens + model.layers + model.norm)

final class StandardModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [StandardDecoderLayer]
    let norm: RMSNorm

    init(_ args: StandardModelConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )
        self.layers = (0 ..< args.hiddenLayers).map { _ in
            StandardDecoderLayer(args)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [VMLXKVCache?]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = vmlxCreateAttentionMask(h: h, cache: cache?.first.flatMap { $0 })

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

// MARK: - Standard Transformer Model

/// Generic decoder-only transformer that handles all standard HF model architectures.
///
/// Supports: Llama 2/3/4, Qwen 2/2.5/3, Mistral (v0.x), Gemma 2/3, Phi 3/4,
/// StarCoder 2, InternLM 2, Granite, Cohere, and any model following the
/// standard {model.embed_tokens, model.layers.N.self_attn/mlp, model.norm, lm_head}
/// weight key convention with SwiGLU MLP and RoPE attention.
public class StandardTransformerModel: Module {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: StandardModelConfiguration

    @ModuleInfo(key: "model") var model: StandardModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: StandardModelConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.resolvedKVHeads, count: args.hiddenLayers)

        _model.wrappedValue = StandardModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [VMLXKVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache() -> [VMLXKVCache] {
        (0 ..< configuration.hiddenLayers).map { _ in VMLXKVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            var key = key

            // Skip vision/multimodal weights (VL models like Mistral-Small-4)
            if key.hasPrefix("vision_tower") || key.hasPrefix("multi_modal_projector")
                || key.hasPrefix("model.visual") {
                continue
            }

            // Remap language_model.model.* -> model.* (VL model wrapper)
            if key.hasPrefix("language_model.model.") {
                key = String(key.dropFirst("language_model.".count))
            } else if key.hasPrefix("language_model.lm_head") {
                key = String(key.dropFirst("language_model.".count))
            } else if key.hasPrefix("language_model.") {
                key = String(key.dropFirst("language_model.".count))
            }

            sanitized[key] = value
        }

        // Remove lm_head.weight if tied to embeddings
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        return sanitized
    }
}

// MARK: - VMLXNativeModel + VMLXSanitizable

extension StandardTransformerModel: VMLXNativeModel, VMLXSanitizable {}
