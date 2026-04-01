//
//  Mistral4Model.swift
//  VMLXRuntime
//
//  Native VMLXRuntime implementation of Mistral Small 4 (119B MoE with MLA).
//
//  Architecture: Multi-head Latent Attention (MLA) + Mixture of Experts (MoE)
//  - 36 decoder layers, 32 attention heads, 128 routed experts + 1 shared expert
//  - YARn RoPE with Llama 4 position-dependent scaling
//  - FP8 weights (float8_e4m3fn) with weight_scale_inv dequantization
//
//  Ported from:
//  - Python: mlx_lm/models/mistral4.py
//  - Swift: mlx-swift-lm DeepseekV3.swift (MLA reference)
//

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// MARK: - Configuration

/// Text model configuration for Mistral Small 4.
/// Parsed from the `text_config` key in config.json (top-level model_type is "mistral3").
public struct Mistral4TextConfiguration: Codable, Sendable {
    var modelType: String = "mistral4"
    var vocabSize: Int = 131072
    var hiddenSize: Int = 4096
    var intermediateSize: Int = 12288
    var moeIntermediateSize: Int = 2048
    var numHiddenLayers: Int = 36
    var numAttentionHeads: Int = 32
    var numKeyValueHeads: Int = 32
    var nSharedExperts: Int? = 1
    var nRoutedExperts: Int = 128
    var numExpertsPerTok: Int = 4
    var routedScalingFactor: Float = 1.0
    var kvLoraRank: Int = 256
    var qLoraRank: Int? = 1024
    var qkRopeHeadDim: Int = 64
    var vHeadDim: Int = 128
    var qkNopeHeadDim: Int = 64
    var headDim: Int = 128
    var nGroup: Int = 1
    var topkGroup: Int = 1
    var firstKDenseReplace: Int = 0
    var moeLayerFreq: Int = 1
    var maxPositionEmbeddings: Int = 1048576
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 10000.0
    var ropeScaling: [String: StringOrNumber]? = nil
    var ropeParameters: [String: StringOrNumber]? = nil
    var attentionBias: Bool = false
    var normTopkProb: Bool = true
    var tieWordEmbeddings: Bool = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case headDim = "head_dim"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case firstKDenseReplace = "first_k_dense_replace"
        case moeLayerFreq = "moe_layer_freq"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case normTopkProb = "norm_topk_prob"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mistral4"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 131072
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12288
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2048
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 36
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 32
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 128
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 4
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 256
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 64
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1048576
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        ropeParameters = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }

    /// Total Q/K head dimension (nope + rope).
    var qHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }

    /// Merge rope_parameters into rope_scaling if needed (matches Python __post_init__).
    public mutating func resolveRopeScaling() {
        if let rp = ropeParameters, ropeScaling == nil {
            var scaling = [String: StringOrNumber]()
            if let t = rp["type"] ?? rp["rope_type"] {
                scaling["type"] = t
            } else {
                scaling["type"] = .string("yarn")
            }
            scaling["factor"] = rp["factor"] ?? .float(128.0)
            scaling["original_max_position_embeddings"] =
                rp["original_max_position_embeddings"] ?? .int(8192)
            scaling["beta_fast"] = rp["beta_fast"] ?? .float(32.0)
            scaling["beta_slow"] = rp["beta_slow"] ?? .float(1.0)
            scaling["mscale"] = rp["mscale"] ?? .float(1.0)
            scaling["mscale_all_dim"] = rp["mscale_all_dim"] ?? .float(1.0)
            scaling["llama_4_scaling_beta"] = rp["llama_4_scaling_beta"] ?? .float(0.0)
            if let theta = rp["rope_theta"]?.asFloat() {
                self.ropeTheta = theta
            }
            self.ropeScaling = scaling
        }
    }
}

/// Top-level configuration wrapper.
/// The actual config.json has model_type "mistral3" with text config nested.
public struct Mistral4Configuration: Codable, Sendable {
    var modelType: String
    var textConfig: Mistral4TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)

        if let textConfig = try container.decodeIfPresent(
            Mistral4TextConfiguration.self, forKey: .textConfig)
        {
            self.textConfig = textConfig
        } else {
            // Flat config (text_config fields at top level)
            self.textConfig = try Mistral4TextConfiguration(from: decoder)
        }
        self.textConfig.resolveRopeScaling()
    }
}

// MARK: - YARn RoPE

/// YARn RoPE for Mistral 4 with interleaved (non-traditional) mode.
/// For Mistral 4, mscale == mscale_all_dim == 1.0, so the mscale factor is exactly 1.0.
class Mistral4YarnRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dimensions: Int
    private let _mscale: Float
    private let _freqs: MLXArray

    init(
        dim: Int,
        maxPositionEmbeddings: Int = 1048576,
        base: Float = 10000,
        scalingFactor: Float = 128.0,
        originalMaxPositionEmbeddings: Int = 8192,
        betaFast: Float = 32,
        betaSlow: Float = 1,
        mscale: Float = 1,
        mscaleAllDim: Float = 1
    ) {
        precondition(dim % 2 == 0, "Dimensions must be even")
        self.dimensions = dim

        func yarnGetMscale(scale: Float, ms: Float) -> Float {
            if scale <= 1 { return 1.0 }
            return 0.1 * ms * log(scale) + 1.0
        }

        func yarnFindCorrectionDim(numRotations: Float) -> Float {
            Float(dim) * log(Float(originalMaxPositionEmbeddings) / (numRotations * 2 * Float.pi))
                / (2 * log(base))
        }

        func yarnFindCorrectionRange() -> (Int, Int) {
            let low = Int(floor(yarnFindCorrectionDim(numRotations: betaFast)))
            let high = Int(ceil(yarnFindCorrectionDim(numRotations: betaSlow)))
            return (max(low, 0), min(high, dim - 1))
        }

        // For Mistral 4: mscale=1, mscaleAllDim=1 -> _mscale = 1.0
        self._mscale =
            yarnGetMscale(scale: scalingFactor, ms: mscale)
            / yarnGetMscale(scale: scalingFactor, ms: mscaleAllDim)

        let indices = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32)
        let freqExtra = pow(base, indices / Float(dim))
        let freqInter = scalingFactor * pow(base, indices / Float(dim))

        let (low, high) = yarnFindCorrectionRange()
        var maxRange = Float(high - low)
        if maxRange == 0 { maxRange = 0.001 }
        let freqMask =
            1.0
            - clip(
                (MLXArray(0 ..< (dim / 2)).asType(.float32) - Float(low)) / maxRange,
                min: 0, max: 1)

        self._freqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        var x = x
        if _mscale != 1.0 {
            x = _mscale * x
        }
        return MLXFast.RoPE(
            x, dimensions: dimensions, traditional: false,  // interleaved RoPE: Mistral 4 uses traditional=False (GPT-NeoX style)
            base: nil, scale: 1.0, offset: offset, freqs: _freqs
        )
    }

    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        var x = x
        if _mscale != 1.0 {
            x = _mscale * x
        }
        return MLXFast.RoPE(
            x, dimensions: dimensions, traditional: false,
            base: nil, scale: 1.0, offset: offset, freqs: _freqs
        )
    }
}

// MARK: - MLA Attention

/// Multi-head Latent Attention for Mistral 4.
///
/// Q path: x -> q_a_proj -> layernorm -> q_b_proj -> reshape -> split(q_nope, q_rope)
/// KV path: x -> kv_a_proj_with_mqa -> split(compressed_kv, k_pe)
///   compressed_kv -> layernorm -> kv_b_proj -> split(k_nope, values)
///   k_pe gets RoPE, then expand 1 head -> num_heads
///
/// Llama 4 scaling: queries *= 1 + beta * log(1 + floor(pos/max_pos))
class Mistral4Attention: Module {
    let numHeads: Int
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int  // qk_nope + qk_rope
    let scale: Float

    let rope: Mistral4YarnRoPE
    let llama4Beta: Float
    let llama4MaxPos: Float

    // Q path (with LoRA compression)
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?

    // Q path (direct, when q_lora_rank is nil/0)
    @ModuleInfo(key: "q_proj") var qProj: Linear?

    // KV path
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear

    // Output
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: Mistral4TextConfiguration) {
        self.numHeads = config.numAttentionHeads
        self.qLoraRank = (config.qLoraRank != nil && config.qLoraRank! > 0) ? config.qLoraRank : nil
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim  // 128
        self.scale = pow(Float(qHeadDim), -0.5)

        // Q path
        if let qlr = qLoraRank {
            self._qAProj.wrappedValue = Linear(
                config.hiddenSize, qlr, bias: config.attentionBias)
            self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qlr, eps: config.rmsNormEps)
            self._qBProj.wrappedValue = Linear(
                qlr, numHeads * qHeadDim, bias: false)
            self._qProj.wrappedValue = nil
        } else {
            self._qAProj.wrappedValue = nil
            self._qALayerNorm.wrappedValue = nil
            self._qBProj.wrappedValue = nil
            self._qProj.wrappedValue = Linear(
                config.hiddenSize, numHeads * qHeadDim, bias: false)
        }

        // KV path
        self._kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize,
            kvLoraRank + qkRopeHeadDim,
            bias: config.attentionBias)
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)
        self._kvBProj.wrappedValue = Linear(
            kvLoraRank,
            numHeads * (qkNopeHeadDim + vHeadDim),
            bias: false)

        // Output
        self._oProj.wrappedValue = Linear(
            numHeads * vHeadDim, config.hiddenSize, bias: config.attentionBias)

        // YARn RoPE
        let ropeCfg = config.ropeScaling
        self.rope = Mistral4YarnRoPE(
            dim: qkRopeHeadDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta,
            scalingFactor: ropeCfg?["factor"]?.asFloat() ?? 128.0,
            originalMaxPositionEmbeddings:
                ropeCfg?["original_max_position_embeddings"]?.asInt() ?? 8192,
            betaFast: ropeCfg?["beta_fast"]?.asFloat() ?? 32.0,
            betaSlow: ropeCfg?["beta_slow"]?.asFloat() ?? 1.0,
            mscale: ropeCfg?["mscale"]?.asFloat() ?? 1.0,
            mscaleAllDim: ropeCfg?["mscale_all_dim"]?.asFloat() ?? 1.0)

        // Llama 4 position-dependent query scaling
        self.llama4Beta = ropeCfg?["llama_4_scaling_beta"]?.asFloat() ?? 0.0
        self.llama4MaxPos = Float(
            ropeCfg?["original_max_position_embeddings"]?.asInt() ?? 8192)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: VMLXKVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        // --- Q path ---
        var q: MLXArray
        if qLoraRank != nil, let qA = qAProj, let qNorm = qALayerNorm, let qB = qBProj {
            q = qB(qNorm(qA(x)))
        } else if let qDirect = qProj {
            q = qDirect(x)
        } else {
            fatalError("Mistral4Attention: no Q projection configured")
        }

        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = splitQ[0]
        var qPe = splitQ[1]

        // --- KV path ---
        var compressedKv = kvAProjWithMqa(x)
        let splitKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitKv[0]
        var kPe = splitKv[1]

        // k_pe is a single head: (B, L, ropeHeadDim) -> (B, 1, L, ropeHeadDim)
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        // Decompress: (B, L, kvLoraRank) -> (B, L, numHeads, nope+vHeadDim)
        var kv = kvBProj(kvALayerNorm(compressedKv))
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let splitKvDecompress = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = splitKvDecompress[0]
        var values = splitKvDecompress[1]

        // --- RoPE (interleaved) ---
        let offset = cache?.offset ?? 0
        qPe = rope(qPe, offset: offset)
        kPe = rope(kPe, offset: offset)

        // Expand k_pe from 1 head -> numHeads
        kPe = repeated(kPe, count: numHeads, axis: 1)

        // --- Assemble full keys and queries ---
        var keys = concatenated([kNope, kPe], axis: -1)
        var queries = concatenated([qNope, qPe], axis: -1)

        // --- Cache update ---
        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // --- Llama 4 position-dependent query scaling ---
        if llama4Beta > 0 {
            let currentOffset = cache?.offset ?? 0
            let l4Scale = 1.0 + llama4Beta * log(
                1.0 + floor(Float(currentOffset) / llama4MaxPos))
            if l4Scale != 1.0 {
                queries = queries * l4Scale
            }
        }

        // --- Attention ---
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - MLP (for shared expert)

class Mistral4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE Gate

/// MoE gate for Mistral 4 with softmax routing.
/// Gate weights stay as float (not quantized).
class Mistral4MoEGate: Module {
    let topK: Int
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let nGroup: Int
    let topkGroup: Int
    let normTopkProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ config: Mistral4TextConfiguration) {
        self.topK = config.numExpertsPerTok
        self.nRoutedExperts = config.nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.normTopkProb = config.normTopkProb
        self._weight.wrappedValue = MLXArray.zeros([nRoutedExperts, config.hiddenSize])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        // Gate matmul
        let gates = x.matmul(weight.T)
        var scores = softmax(gates, axis: -1, precise: true)

        // Group routing (for Mistral 4: nGroup=1, topkGroup=1 -> trivial, skip)
        if nGroup > 1, topkGroup < nGroup {
            let groupedShape = Array(scores.shape.dropLast()) + [nGroup, nRoutedExperts / nGroup]
            var scoresGrouped = scores.reshaped(groupedShape)
            let groupScores = scoresGrouped.max(axis: -1, keepDims: true)
            let k = nGroup - topkGroup
            let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
            scoresGrouped = putAlong(scoresGrouped, stopGradient(groupIdx), values: MLXArray(0.0), axis: -2)
            scores = scoresGrouped.reshaped(scores.shape)
        }

        // Top-k selection
        let k = topK
        let inds = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var weights = takeAlong(scores, inds, axis: -1)

        // Normalize weights (Mistral 4: normTopkProb=true)
        if normTopkProb {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }

        weights = weights * routedScalingFactor

        return (inds, weights)
    }
}

// MARK: - MoE Block

/// MoE block with routed experts (SwitchGLU) and optional shared expert.
class Mistral4MoE: Module {
    @ModuleInfo(key: "gate") var gate: Mistral4MoEGate
    @ModuleInfo(key: "switch_mlp") var switchMlp: VMLXSwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: Mistral4MLP?

    init(_ config: Mistral4TextConfiguration) {
        self._gate.wrappedValue = Mistral4MoEGate(config)
        self._switchMlp.wrappedValue = VMLXSwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activation: silu,
            bias: false)

        if let nShared = config.nSharedExperts, nShared > 0 {
            let sharedInter = config.moeIntermediateSize * nShared
            self._sharedExperts.wrappedValue = Mistral4MLP(
                hiddenSize: config.hiddenSize, intermediateSize: sharedInter)
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMlp(x, indices)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let shared = sharedExperts {
            y = y + shared(x)
        }
        return y
    }
}

// MARK: - Decoder Layer

class Mistral4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Mistral4Attention
    var mlp: Module  // Either Mistral4MoE or Mistral4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    private let isMoE: Bool

    init(_ config: Mistral4TextConfiguration, layerIdx: Int) {
        self._selfAttn.wrappedValue = Mistral4Attention(config)

        // Determine if this layer is MoE
        let moe = layerIdx >= config.firstKDenseReplace
            && layerIdx % config.moeLayerFreq == 0
        self.isMoE = moe

        if moe {
            self.mlp = Mistral4MoE(config)
        } else {
            self.mlp = Mistral4MLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: VMLXKVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2: MLXArray
        if isMoE, let moe = mlp as? Mistral4MoE {
            r2 = moe(postAttentionLayerNorm(h))
        } else if let dense = mlp as? Mistral4MLP {
            r2 = dense(postAttentionLayerNorm(h))
        } else {
            fatalError("Mistral4DecoderLayer: unexpected MLP type")
        }
        return h + r2
    }
}

// MARK: - Model Inner

class Mistral4ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [Mistral4DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Mistral4TextConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            Mistral4DecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [VMLXKVCache?]?) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = vmlxCreateAttentionMask(h: h, cache: cache?.first as? VMLXKVCache)

        let caches: [VMLXKVCache?] = cache ?? Array(repeating: nil, count: layers.count)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: caches[i])
        }

        return norm(h)
    }
}

// MARK: - Top-Level Transformer Model

public class Mistral4TransformerModel: Module {
    public let vocabularySize: Int
    let config: Mistral4TextConfiguration

    @ModuleInfo(key: "model") var model: Mistral4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Mistral4TextConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self._model.wrappedValue = Mistral4ModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [VMLXKVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if config.tieWordEmbeddings {
            return model.embedTokens.asLinear(out)
        }
        return lmHead!(out)
    }

    public func newCache() -> [VMLXKVCache] {
        (0 ..< config.numHiddenLayers).map { _ in VMLXKVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = [String: MLXArray]()

        // Collect all weight_scale_inv keys for FP8 dequantization
        var scaleInvKeys = Set<String>()
        for key in weights.keys {
            if key.hasSuffix(".weight_scale_inv") {
                scaleInvKeys.insert(key)
            }
        }

        for (origKey, value) in weights {
            var key = origKey

            // Skip vision-related weights (language model only)
            if key.hasPrefix("vision_tower") || key.hasPrefix("multi_modal_projector") {
                continue
            }

            // Skip activation_scale (not needed for weight-only inference)
            if key.hasSuffix(".activation_scale") {
                continue
            }

            // Skip weight_scale_inv (processed with corresponding weight)
            if key.hasSuffix(".weight_scale_inv") {
                continue
            }

            // Strip language_model. prefix
            if key.hasPrefix("language_model.") {
                key = String(key.dropFirst("language_model.".count))
            }

            // FP8 dequantization: weight * scale_inv
            let scaleInvKey: String
            if origKey.hasSuffix(".weight") {
                scaleInvKey =
                    origKey.replacingOccurrences(of: ".weight", with: ".weight_scale_inv")
            } else {
                scaleInvKey = origKey + "_scale_inv"
            }
            if scaleInvKeys.contains(scaleInvKey), let scaleInv = weights[scaleInvKey] {
                // Dequantize FP8: cast to bfloat16 then multiply by scale_inv
                let dequantized = mistral4DequantFP8(
                    weight: value, scaleInv: scaleInv)
                newWeights[key] = dequantized
                continue
            }

            newWeights[key] = value
        }

        // Convert per-expert weights into stacked SwitchGLU format
        for l in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(l)"

            // Handle separate per-expert weights -> stacked SwitchGLU
            for (_, projName) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for suffix in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(suffix)"
                    if newWeights[firstKey] != nil {
                        let joined = (0 ..< config.nRoutedExperts).map {
                            newWeights["\(prefix).mlp.experts.\($0).\(projName).\(suffix)"]!
                        }
                        newWeights["\(prefix).mlp.switch_mlp.\(projName).\(suffix)"] = stacked(
                            joined)
                        // Remove individual expert keys
                        for e in 0 ..< config.nRoutedExperts {
                            newWeights.removeValue(
                                forKey: "\(prefix).mlp.experts.\(e).\(projName).\(suffix)")
                        }
                    }
                }
            }

            // Handle fused gate_up_proj format (some HF checkpoints)
            let gateUpKey = "\(prefix).mlp.experts.gate_up_proj"
            if let gateUp = newWeights[gateUpKey] {
                let mid = gateUp.dim(-2) / 2
                newWeights["\(prefix).mlp.switch_mlp.gate_proj.weight"] = gateUp[
                    .ellipsis, ..<mid, 0...]
                newWeights["\(prefix).mlp.switch_mlp.up_proj.weight"] = gateUp[
                    .ellipsis, mid..., 0...]
                newWeights.removeValue(forKey: gateUpKey)

                let downKey = "\(prefix).mlp.experts.down_proj"
                if let down = newWeights[downKey] {
                    newWeights["\(prefix).mlp.switch_mlp.down_proj.weight"] = down
                    newWeights.removeValue(forKey: downKey)
                }
            }
        }

        // Filter out rotary_emb.inv_freq (not used, RoPE computed from config)
        return newWeights.filter { key, _ in
            !key.contains("rotary_emb.inv_freq")
        }
    }
}

// MARK: - FP8 Dequantization Helper

/// Dequantize FP8 block-scaled weights.
/// FP8 weights use per-block scale_inv with block size 128.
/// weight = cast_to_bf16(weight) * scale_inv (broadcast per block)
func mistral4DequantFP8(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
    // For switch_mlp (3D: [num_experts, out, in]) or regular (2D: [out, in])
    let ndim = weight.ndim

    if ndim == 2 {
        // Standard 2D weight: [M, N]
        let bs = 128
        let (m, n) = (weight.dim(0), weight.dim(1))
        let padBottom = (bs - m % bs) % bs
        let padSide = (bs - n % bs) % bs

        var w = weight.asType(.bfloat16)
        if padBottom > 0 || padSide > 0 {
            w = padded(w, widths: [.init((0, padBottom)), .init((0, padSide))])
        }
        w = w.reshaped((m + padBottom) / bs, bs, (n + padSide) / bs, bs)
        w = w * scaleInv[0..., .newAxis, 0..., .newAxis]
        w = w.reshaped(m + padBottom, n + padSide)
        if padBottom > 0 || padSide > 0 {
            w = w[0 ..< m, 0 ..< n]
        }
        return w
    } else if ndim == 3 {
        // Expert weight: [num_experts, out, in]
        let bs = 128
        let (e, m, n) = (weight.dim(0), weight.dim(1), weight.dim(2))
        let padBottom = (bs - m % bs) % bs
        let padSide = (bs - n % bs) % bs

        var w = weight.asType(.bfloat16)
        if padBottom > 0 || padSide > 0 {
            w = padded(w, widths: [.init((0, 0)), .init((0, padBottom)), .init((0, padSide))])
        }
        w = w.reshaped(e, (m + padBottom) / bs, bs, (n + padSide) / bs, bs)
        w = w * scaleInv[0..., 0..., .newAxis, 0..., .newAxis]
        w = w.reshaped(e, m + padBottom, n + padSide)
        if padBottom > 0 || padSide > 0 {
            w = w[0..., 0 ..< m, 0 ..< n]
        }
        return w
    } else {
        // Scalar or 1D scale: simple multiply
        return weight.asType(.bfloat16) * scaleInv
    }
}

// MARK: - VMLXNativeModel + VMLXSanitizable

extension Mistral4TransformerModel: VMLXNativeModel, VMLXSanitizable {}
