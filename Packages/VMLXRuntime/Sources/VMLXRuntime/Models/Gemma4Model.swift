//
//  Gemma4Model.swift
//  VMLXRuntime
//
//  Gemma 4 text model: 30-layer MoE with mixed sliding/full attention.
//  128 experts top-8, GELU activation, logit softcapping, per-layer scaling.
//
//  Weight prefix: model.language_model.layers.{i}.*
//  Vision tower weights are skipped (text-only inference).
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    var modelType: String = "gemma4_text"
    var hiddenSize: Int = 2816
    var numHiddenLayers: Int = 30
    var numAttentionHeads: Int = 16
    var numKeyValueHeads: Int = 8
    var numGlobalKeyValueHeads: Int = 2
    var headDim: Int = 256
    var globalHeadDim: Int = 512
    var intermediateSize: Int = 2112
    var moeIntermediateSize: Int = 704
    var numExperts: Int = 128
    var topKExperts: Int = 8
    var vocabSize: Int = 262144
    var rmsNormEps: Float = 1e-6
    var slidingWindow: Int = 1024
    var layerTypes: [String] = []
    var finalLogitSoftcapping: Float? = 30.0
    var tieWordEmbeddings: Bool = true
    var attentionBias: Bool = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
    }
}

/// Top-level config with nested text_config
struct Gemma4TopLevelConfig: Codable {
    var textConfig: Gemma4TextConfiguration?
    var modelType: String?
    var tieWordEmbeddings: Bool?
    var quantization: QuantizationConfig?

    struct QuantizationConfig: Codable {
        var groupSize: Int?
        var bits: Int?
        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case modelType = "model_type"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantization
    }
}

// MARK: - Attention

final class Gemma4Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: VMLXRoPELayer

    init(_ config: Gemma4TextConfiguration, layerIndex: Int) {
        let layerType = layerIndex < config.layerTypes.count
            ? config.layerTypes[layerIndex] : "sliding_attention"
        let isSliding = layerType != "full_attention"

        if isSliding {
            self.numHeads = config.numAttentionHeads
            self.numKVHeads = config.numKeyValueHeads
            self.headDim = config.headDim
        } else {
            self.numHeads = config.numAttentionHeads
            self.numKVHeads = config.numGlobalKeyValueHeads
            self.headDim = config.globalHeadDim
        }
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        // RoPE config per layer type
        if isSliding {
            self.rope = vmlxInitializeRope(dims: headDim, base: 10000.0, traditional: false, scalingConfig: nil, maxPositionEmbeddings: 262144)
        } else {
            let ropeDims = max(1, Int(Float(config.globalHeadDim) * 0.25))
            self.rope = vmlxInitializeRope(dims: ropeDims, base: 1000000.0, traditional: false, scalingConfig: nil, maxPositionEmbeddings: 262144)
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: VMLXKVCache?) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        queries = qNorm(queries)
        keys = kNorm(keys)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let attnOut = vmlxAttentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(attnOut)
    }
}

// MARK: - Dense MLP

final class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4TextConfiguration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Router

final class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "scale") var routerScale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    let numExperts: Int
    let topK: Int

    init(_ config: Gemma4TextConfiguration) {
        self.numExperts = config.numExperts
        self.topK = config.topKExperts
        _proj.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _routerScale.wrappedValue = MLXArray(1.0)
        _perExpertScale.wrappedValue = MLXArray.ones([config.numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        var logits = proj(x).asType(.float32)
        logits = logits * routerScale
        let probs = sigmoid(logits) * perExpertScale.asType(.float32)

        let topKIndices = argPartition(probs, kth: numExperts - topK, axis: -1)[
            .ellipsis, (numExperts - topK)...]
        let topKWeights = takeAlong(probs, topKIndices, axis: -1)
        let weightSum = topKWeights.sum(axis: -1, keepDims: true)
        let normalizedWeights = topKWeights / (weightSum + 1e-8)

        return (indices: topKIndices, weights: normalizedWeights)
    }
}

// MARK: - Decoder Layer

final class Gemma4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo(key: "mlp") var mlp: Gemma4MLP
    @ModuleInfo(key: "switch_mlp") var switchMLP: VMLXSwitchGLU
    @ModuleInfo(key: "router") var router: Gemma4Router

    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayernorm2: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayernorm1: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayernorm2: RMSNorm

    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIndex: Int) {
        _selfAttn.wrappedValue = Gemma4Attention(config, layerIndex: layerIndex)
        _mlp.wrappedValue = Gemma4MLP(config)
        _switchMLP.wrappedValue = VMLXSwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            activation: { geluApproximate($0) },
            isSilu: false,
            isGelu: true,
            bias: false
        )
        _router.wrappedValue = Gemma4Router(config)

        _inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayernorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayernorm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayernorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        _layerScalar.wrappedValue = MLXArray(1.0)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: VMLXKVCache?) -> MLXArray {
        var h = x

        // 1. Attention block
        let attnOut = selfAttn(inputLayernorm(h), mask: mask, cache: cache)
        h = h + postAttentionLayernorm(attnOut)

        // 2. Feedforward: dense MLP and MoE are PARALLEL branches
        let residual = h
        let mlpOut = mlp(preFeedforwardLayernorm(h))

        // Path 1: dense MLP → post_feedforward_layernorm_1
        let path1 = postFeedforwardLayernorm1(mlpOut)

        // Path 2: router on residual → experts → post_feedforward_layernorm_2
        let flatResidual = residual.reshaped(-1, residual.dim(-1))
        let (expertIndices, expertWeights) = router(flatResidual)
        let moeIn = preFeedforwardLayernorm2(flatResidual)
        let moeOut = switchMLP(moeIn, expertIndices)
        let weightedMoeOut = (moeOut * expertWeights.expandedDimensions(axis: -1)).sum(axis: -2)
        let path2 = postFeedforwardLayernorm2(weightedMoeOut.reshaped(residual.shape))

        // Combine parallel paths → post_feedforward_layernorm → residual add
        let combined = postFeedforwardLayernorm(path1 + path2)
        h = residual + combined

        // layer_scalar applied to entire layer delta
        h = h * layerScalar.asType(h.dtype)

        return h
    }
}

// MARK: - Text Model

public class Gemma4TextModel: Module, VMLXNativeModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let config: Gemma4TextConfiguration

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0..<config.numHiddenLayers).map { i in
            let lt = i < config.layerTypes.count ? config.layerTypes[i] : "sliding_attention"
            return lt == "full_attention" ? config.numGlobalKeyValueHeads : config.numKeyValueHeads
        }

        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { Gemma4DecoderLayer(config, layerIndex: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [VMLXKVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        h = h * MLXArray(Float(sqrt(Double(config.hiddenSize))))

        let t = h.dim(1)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if t > 1 {
            let offset = cache?.first?.offset ?? 0
            mask = offset == 0 ? .causal : .array(vmlxCreateCausalMask(n: t, offset: offset))
        } else {
            mask = .none
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        var out = embedTokens.asLinear(norm(h))
        if let cap = config.finalLogitSoftcapping, cap > 0 {
            out = tanh(out / cap) * cap
        }
        return out
    }

    public func newCache() -> [VMLXKVCache] {
        (0..<config.numHiddenLayers).map { _ in VMLXKVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            var k = key
            // Strip language_model prefix
            if k.hasPrefix("model.language_model.") {
                k = String(k.dropFirst("model.language_model.".count))
            }
            // Also handle "model." prefix for embed_tokens
            if k.hasPrefix("model.") && !k.hasPrefix("model.vision_tower") && !k.hasPrefix("model.embed_vision") {
                k = String(k.dropFirst("model.".count))
            }
            // Skip vision tower weights entirely
            if k.contains("vision_tower") || k.contains("embed_vision") {
                continue
            }
            sanitized[k] = value
        }
        return sanitized
    }
}
