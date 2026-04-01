//
//  NemotronHModel.swift
//  VMLXRuntime
//
//  Native Nemotron-H model: hybrid Mamba2 SSM + GQA attention + MoE-MLP.
//  Layer routing via hybrid_override_pattern: M=Mamba2, *=Attention, E=MoE-MLP.
//
//  Weight naming: backbone.layers.N.mixer.* (not self_attn/mlp)
//  Mamba2 weights: in_proj, out_proj, conv1d, A_log, D, dt_bias, norm
//  Attention weights: q_proj, k_proj, v_proj, o_proj
//  MoE weights: gate, switch_mlp.up_proj/down_proj, shared_experts.up_proj/down_proj
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Configuration

public struct NemotronHConfiguration: Codable, Sendable {
    var modelType: String = "nemotron_h"
    var vocabSize: Int = 131072
    var hiddenSize: Int = 2688
    var intermediateSize: Int = 1856
    var moeIntermediateSize: Int = 1856
    var moeSharedExpertIntermediateSize: Int = 3712
    var numHiddenLayers: Int = 52
    var numAttentionHeads: Int = 32
    var numKeyValueHeads: Int = 2
    var headDim: Int = 128
    var mambaHeadDim: Int = 64
    var mambaNumHeads: Int = 64
    var ssmStateSize: Int = 128
    var convKernel: Int = 4
    var expand: Int = 2
    var nRoutedExperts: Int = 128
    var nSharedExperts: Int = 1
    var numExpertsPerTok: Int = 6
    var routedScalingFactor: Float = 2.5
    var nGroups: Int = 8
    var normEps: Float = 1e-5
    var ropeTheta: Float = 10000.0
    var maxPositionEmbeddings: Int = 262144
    var hybridOverridePattern: String = ""
    var tieWordEmbeddings: Bool = false
    var mambaHiddenAct: String = "silu"
    var mlpHiddenAct: String = "relu2"
    var useConvBias: Bool = true
    var mambaProjBias: Bool = false
    var normTopkProb: Bool = true
    var timeStepMin: Float = 0.001
    var timeStepMax: Float = 0.1
    var timeStepFloor: Float = 0.0001

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case mambaHeadDim = "mamba_head_dim"
        case mambaNumHeads = "mamba_num_heads"
        case ssmStateSize = "ssm_state_size"
        case convKernel = "conv_kernel"
        case expand = "expand"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case routedScalingFactor = "routed_scaling_factor"
        case nGroups = "n_groups"
        case normEps = "norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hybridOverridePattern = "hybrid_override_pattern"
        case tieWordEmbeddings = "tie_word_embeddings"
        case mambaHiddenAct = "mamba_hidden_act"
        case mlpHiddenAct = "mlp_hidden_act"
        case useConvBias = "use_conv_bias"
        case mambaProjBias = "mamba_proj_bias"
        case normTopkProb = "norm_topk_prob"
        case timeStepMin = "time_step_min"
        case timeStepMax = "time_step_max"
        case timeStepFloor = "time_step_floor"
    }

    /// Computed: Mamba2 intermediate size = expand * hiddenSize
    var mambaIntermediateSize: Int { expand * hiddenSize }

    /// Computed: Mamba2 conv dimension (x + B + C portions)
    var mambaConvDim: Int { mambaNumHeads * mambaHeadDim + 2 * nGroups * ssmStateSize }

    /// Computed: in_proj output = z + x + B + C + dt
    var mambaInProjSize: Int {
        mambaNumHeads * mambaHeadDim  // z (gate)
        + mambaNumHeads * mambaHeadDim  // x
        + 2 * nGroups * ssmStateSize  // B + C
        + mambaNumHeads  // dt
    }
}

// MARK: - Mamba2 Mixer

/// Mamba2 (SSD) layer for Nemotron-H.
/// Uses structured state space duality with selective scan.
final class NemotronHMamba2Mixer: Module {
    let config: NemotronHConfiguration
    let mambaNumHeads: Int
    let mambaHeadDim: Int
    let ssmStateSize: Int
    let nGroups: Int
    let convDim: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "norm") var norm: RMSNorm

    var aLog: MLXArray  // [mambaNumHeads]
    var D: MLXArray      // [mambaNumHeads]
    var dtBias: MLXArray // [mambaNumHeads]

    init(_ config: NemotronHConfiguration) {
        self.config = config
        self.mambaNumHeads = config.mambaNumHeads
        self.mambaHeadDim = config.mambaHeadDim
        self.ssmStateSize = config.ssmStateSize
        self.nGroups = config.nGroups
        self.convDim = config.mambaConvDim

        let inProjSize = config.mambaInProjSize
        _inProj.wrappedValue = Linear(config.hiddenSize, inProjSize, bias: config.mambaProjBias)
        _outProj.wrappedValue = Linear(mambaNumHeads * mambaHeadDim, config.hiddenSize, bias: config.mambaProjBias)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim,
            kernelSize: config.convKernel, padding: 0,
            groups: convDim, bias: config.useConvBias
        )
        _norm.wrappedValue = RMSNorm(dimensions: mambaNumHeads * mambaHeadDim, eps: config.normEps)

        self.aLog = MLXArray.zeros([mambaNumHeads])
        self.D = MLXArray.ones([mambaNumHeads])
        self.dtBias = MLXArray.zeros([mambaNumHeads])

        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: VMLXMambaCache? = nil) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)

        // Project input: [B, S, hiddenSize] → [B, S, inProjSize]
        let projected = inProj(x)

        // Split into z (gate), xBC (conv input), dt (time step)
        let zSize = mambaNumHeads * mambaHeadDim
        let xBCSize = convDim  // x + B + C
        let dtSize = mambaNumHeads

        let z = projected[0..., 0..., ..<zSize]
        let xBC = projected[0..., 0..., zSize..<(zSize + xBCSize)]
        let dt = projected[0..., 0..., (zSize + xBCSize)..<(zSize + xBCSize + dtSize)]

        // Conv1d with causal state
        let convState: MLXArray
        if let cached = cache?[0] {
            convState = cached
        } else {
            convState = MLXArray.zeros([B, config.convKernel - 1, convDim], dtype: x.dtype)
        }

        let convInput = concatenated([convState, xBC], axis: 1)
        if let cache {
            cache[0] = convInput[0..., (-(config.convKernel - 1))...]
        }

        let convOut = silu(conv1d(convInput))
        // convOut: [B, S, convDim]

        // Split conv output into x, B_mat, C_mat
        let xSize = mambaNumHeads * mambaHeadDim
        let bcSize = nGroups * ssmStateSize
        let xPart = convOut[0..., 0..., ..<xSize]
            .reshaped(B, S, mambaNumHeads, mambaHeadDim)
        let bMat = convOut[0..., 0..., xSize..<(xSize + bcSize)]
            .reshaped(B, S, nGroups, ssmStateSize)
        let cMat = convOut[0..., 0..., (xSize + bcSize)...]
            .reshaped(B, S, nGroups, ssmStateSize)

        // Time step: softplus(dt + dt_bias)
        let dtSoft = log(1.0 + exp(dt + dtBias))
            .reshaped(B, S, mambaNumHeads, 1)

        // A: negative exp of A_log (decay)
        let A = -exp(aLog).reshaped(1, 1, mambaNumHeads, 1)

        // Discrete A: exp(A * dt)
        let dA = exp(A * dtSoft)  // [B, S, heads, 1]

        // Expand B/C for grouped heads: nGroups → mambaNumHeads
        let headsPerGroup = mambaNumHeads / nGroups
        let bExpanded: MLXArray
        let cExpanded: MLXArray
        if headsPerGroup > 1 {
            // Repeat each group headsPerGroup times along axis 2
            // [B, S, nGroups, stateSize] → [B, S, mambaNumHeads, stateSize]
            bExpanded = MLX.repeated(bMat, count: headsPerGroup, axis: 2)
            cExpanded = MLX.repeated(cMat, count: headsPerGroup, axis: 2)
        } else {
            bExpanded = bMat
            cExpanded = cMat
        }

        // SSM scan (sequential per timestep)
        var ssmState: MLXArray
        if let cached = cache?[1] {
            ssmState = cached  // [B, heads, headDim, stateSize]
        } else {
            ssmState = MLXArray.zeros([B, mambaNumHeads, mambaHeadDim, ssmStateSize], dtype: x.dtype)
        }

        var outputs: [MLXArray] = []
        for t in 0..<S {
            // Slice timestep t: [B, 1, dim] → [B, dim]
            let xt = xPart[0..., t..<(t+1), 0..., 0...].squeezed(axis: 1)    // [B, heads, headDim]
            let bt = bExpanded[0..., t..<(t+1), 0..., 0...].squeezed(axis: 1) // [B, heads, stateSize]
            let ct = cExpanded[0..., t..<(t+1), 0..., 0...].squeezed(axis: 1) // [B, heads, stateSize]
            let dAt = dA[0..., t..<(t+1), 0..., 0...].squeezed(axis: 1)       // [B, heads, 1]

            // state[b,h,d,n] = dA[b,h,1] * state[b,h,d,n] + x[b,h,d] * B[b,h,n]
            // outer product: [B,h,d,1] * [B,h,1,n] → [B,h,d,n]
            let xOuter = xt.expandedDimensions(axis: -1)  // [B, h, d, 1]
            let bOuter = bt.expandedDimensions(axis: -2)   // [B, h, 1, n]
            ssmState = dAt.expandedDimensions(axis: -1) * ssmState + xOuter * bOuter

            // y[b,h,d] = sum_n(state[b,h,d,n] * C[b,h,n]) + D[h] * x[b,h,d]
            let cOuter = ct.expandedDimensions(axis: -2)   // [B, h, 1, n]
            let y = (ssmState * cOuter).sum(axis: -1) + D.reshaped(1, mambaNumHeads, 1) * xt
            outputs.append(y.expandedDimensions(axis: 1))  // [B, 1, heads, headDim]
        }

        if let cache {
            cache[1] = ssmState
        }

        let yAll = concatenated(outputs, axis: 1)  // [B, S, heads, headDim]

        // Gate with z and normalize
        let zReshaped = z.reshaped(B, S, mambaNumHeads, mambaHeadDim)
        let gated = yAll * silu(zReshaped)

        // Output projection
        let normed = norm(gated.reshaped(B, S, -1))
        return outProj(normed)
    }
}

// MARK: - GQA Attention

final class NemotronHAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ config: NemotronHConfiguration) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: false,
            base: config.ropeTheta
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: VMLXKVCache? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)

        var q = qProj(x).reshaped(B, S, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, S, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, S, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)

        if let cache = cache as? VMLXKVCacheSimple {
            (k, v) = cache.update(keys: k, values: v)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, S, -1)

        return oProj(out)
    }
}

// MARK: - MoE MLP

final class NemotronHMoE: Module {
    let numExpertsPerTok: Int
    let routedScalingFactor: Float
    let hasLatentProj: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMlp: NemotronHSwitchMLP
    @ModuleInfo(key: "shared_experts") var sharedExperts: NemotronHSharedExpert
    @ModuleInfo(key: "e_score_correction_bias") var eCorrBias: MLXArray

    // Latent projections (Nemotron-3-Super 120B only, not Cascade 30B)
    @ModuleInfo(key: "fc1_latent_proj") var fc1LatentProj: Linear?
    @ModuleInfo(key: "fc2_latent_proj") var fc2LatentProj: Linear?

    init(_ config: NemotronHConfiguration) {
        self.numExpertsPerTok = config.numExpertsPerTok
        self.routedScalingFactor = config.routedScalingFactor
        // Latent projections exist on 120B Super (512 experts) but not 30B Cascade (128 experts).
        // Detected by expert count — 512+ experts use latent bottleneck.
        self.hasLatentProj = config.nRoutedExperts >= 256

        let latentDim = config.hiddenSize / 4
        let gateDim = hasLatentProj ? latentDim : config.hiddenSize
        let expertInputDim = hasLatentProj ? latentDim : config.hiddenSize
        let expertOutputDim = hasLatentProj ? latentDim : config.hiddenSize

        _gate.wrappedValue = Linear(gateDim, config.nRoutedExperts, bias: false)
        _eCorrBias.wrappedValue = MLXArray.zeros([config.nRoutedExperts])
        _switchMlp.wrappedValue = NemotronHSwitchMLP(config, inputDim: expertInputDim, outputDim: expertOutputDim)
        _sharedExperts.wrappedValue = NemotronHSharedExpert(config)

        if hasLatentProj {
            _fc1LatentProj.wrappedValue = Linear(config.hiddenSize, latentDim, bias: false)
            _fc2LatentProj.wrappedValue = Linear(latentDim, config.hiddenSize, bias: false)
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let origShape = x.shape
        var flat = x.reshaped(-1, origShape.last!)

        // Latent projection (120B Super variant)
        let latentInput: MLXArray
        if let proj = fc1LatentProj {
            latentInput = proj(flat)
        } else {
            latentInput = flat
        }

        // Gate routing in float32 for precision
        var scores = gate(latentInput.asType(.float32))
        scores = scores + eCorrBias

        let k = numExpertsPerTok
        let allIndices = argPartition(scores, kth: scores.dim(-1) - k, axis: -1)
        let topIndices = allIndices[0..., (-k)...]
        // Gather top-k scores using the selected indices
        let topScores = takeAlong(scores, topIndices, axis: -1)
        let weights = softmax(topScores, axis: -1, precise: true)

        // Routed experts
        let expertOut = switchMlp(latentInput, indices: topIndices)
        let scaledWeights = weights * routedScalingFactor
        var routedOut = (expertOut * scaledWeights.expandedDimensions(axis: -1)).sum(axis: 1)

        // Latent back-projection (120B Super variant)
        if let proj = fc2LatentProj {
            routedOut = proj(routedOut)
        }

        // Shared expert (operates on original hidden dim)
        let sharedOut = sharedExperts(flat)

        return (routedOut + sharedOut).reshaped(origShape)
    }
}

/// Routed expert MLP (ReLU² activation, no gate_proj).
/// Weight keys: fc1 (up_proj renamed by sanitize) and fc2 (down_proj renamed).
final class NemotronHSwitchMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: VMLXSwitchLinear   // up_proj
    @ModuleInfo(key: "fc2") var fc2: VMLXSwitchLinear   // down_proj

    init(_ config: NemotronHConfiguration, inputDim: Int? = nil, outputDim: Int? = nil) {
        let inDim = inputDim ?? config.hiddenSize
        let outDim = outputDim ?? config.hiddenSize
        _fc1.wrappedValue = VMLXSwitchLinear(
            inputDims: inDim,
            outputDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts, bias: false
        )
        _fc2.wrappedValue = VMLXSwitchLinear(
            inputDims: config.moeIntermediateSize,
            outputDims: outDim,
            numExperts: config.nRoutedExperts, bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = indices.size >= 64
        var idx = indices
        var inverseOrder = MLXArray()
        if doSort {
            (x, idx, inverseOrder) = vmlxGatherSort(x: x, indices: indices)
        }
        let up = fc1(x, idx, sortedIndices: doSort)
        let activated = relu(up) * relu(up)  // ReLU²
        x = fc2(activated, idx, sortedIndices: doSort)
        if doSort {
            x = vmlxScatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }
        return MLX.squeezed(x, axis: -2)
    }
}

/// Shared expert MLP (ReLU²).
final class NemotronHSharedExpert: Module {
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: NemotronHConfiguration) {
        _upProj.wrappedValue = Linear(config.hiddenSize, config.moeSharedExpertIntermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.moeSharedExpertIntermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // ReLU² activation (matches config.mlpHiddenAct = "relu2")
        let up = upProj(x)
        let activated = relu(up) * relu(up)
        return downProj(activated)
    }
}

// MARK: - Decoder Block

final class NemotronHBlock: Module {
    let layerType: Character  // M, E, *

    @ModuleInfo(key: "mixer") var mixer: Module
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: NemotronHConfiguration, layerIdx: Int) {
        let pattern = config.hybridOverridePattern
        self.layerType = layerIdx < pattern.count
            ? pattern[pattern.index(pattern.startIndex, offsetBy: layerIdx)]
            : "E"

        switch layerType {
        case "M":
            _mixer.wrappedValue = NemotronHMamba2Mixer(config)
        case "*":
            _mixer.wrappedValue = NemotronHAttention(config)
        default:  // "E"
            _mixer.wrappedValue = NemotronHMoE(config)
        }

        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any VMLXKVCache)? = nil
    ) -> MLXArray {
        let normed = norm(x)
        let out: MLXArray
        switch layerType {
        case "M":
            out = (mixer as! NemotronHMamba2Mixer)(normed, cache: cache as? VMLXMambaCache)
        case "*":
            out = (mixer as! NemotronHAttention)(normed, mask: mask, cache: cache)
        default:
            out = (mixer as! NemotronHMoE)(normed)
        }
        return x + out
    }
}

// MARK: - Full Model

public class NemotronHModel: Module {
    public let vocabularySize: Int
    let config: NemotronHConfiguration

    @ModuleInfo(key: "backbone.embeddings") var embeddings: Embedding
    @ModuleInfo(key: "backbone.layers") var layers: [NemotronHBlock]
    @ModuleInfo(key: "backbone.norm_f") var normF: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: NemotronHConfiguration) {
        self.vocabularySize = config.vocabSize
        self.config = config

        _embeddings.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { NemotronHBlock(config, layerIdx: $0) }
        _normF.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }

        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [VMLXKVCache]? = nil) -> MLXArray {
        var h = embeddings(inputs)

        let firstAttnIdx = config.hybridOverridePattern.firstIndex(of: "*")
            .map { config.hybridOverridePattern.distance(from: config.hybridOverridePattern.startIndex, to: $0) }
        let attnCache = firstAttnIdx.flatMap { cache?[$0] as? VMLXKVCacheSimple }
        let mask = vmlxCreateAttentionMask(h: h, cache: attnCache)

        let hop = config.hybridOverridePattern
        for (i, layer) in layers.enumerated() {
            let lt = i < hop.count ? String(hop[hop.index(hop.startIndex, offsetBy: i)]) : "?"
            if i < 5 {
                let line = "[NemotronH] Layer \(i)/\(layers.count) type=\(lt) h=\(h.shape)\n"
                if let fh = FileHandle(forWritingAtPath: "/tmp/vmlx_debug.log") {
                    fh.seekToEndOfFile(); fh.write(line.data(using: .utf8)!); fh.closeFile()
                }
            }
            h = layer(h, mask: mask, cache: cache?[i])
            MLX.eval(h)
            if i < 5 {
                let line = "[NemotronH] Layer \(i) DONE h=\(h.shape)\n"
                if let fh = FileHandle(forWritingAtPath: "/tmp/vmlx_debug.log") {
                    fh.seekToEndOfFile(); fh.write(line.data(using: .utf8)!); fh.closeFile()
                }
            }
        }

        h = normF(h)

        if let lmHead {
            return lmHead(h)
        } else {
            return matmul(h, embeddings.weight.transposed())
        }
    }

    public func newCache() -> [VMLXKVCache] {
        let pattern = config.hybridOverridePattern
        return (0..<config.numHiddenLayers).map { i in
            let lt = i < pattern.count
                ? pattern[pattern.index(pattern.startIndex, offsetBy: i)]
                : Character("E")
            switch lt {
            case "M":
                return VMLXMambaCache()
            case "*":
                return VMLXKVCacheSimple()
            default:
                // MoE-MLP layers don't need cache — use a placeholder
                return VMLXArraysCache(size: 0)
            }
        }
    }
}

// MARK: - Sanitizable

extension NemotronHModel: VMLXSanitizable {
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights: [String: MLXArray] = [:]
        for (key, value) in weights {
            var newKey = key
            // Rename switch_mlp.up_proj → switch_mlp.fc1, down_proj → fc2
            if newKey.contains("switch_mlp.up_proj") {
                newKey = newKey.replacingOccurrences(of: "switch_mlp.up_proj", with: "switch_mlp.fc1")
            } else if newKey.contains("switch_mlp.down_proj") {
                newKey = newKey.replacingOccurrences(of: "switch_mlp.down_proj", with: "switch_mlp.fc2")
            }
            // Move gate.e_score_correction_bias → e_score_correction_bias
            // (gate is a Linear, can't hold child params)
            if newKey.contains(".mixer.gate.e_score_correction_bias") {
                newKey = newKey.replacingOccurrences(of: ".mixer.gate.e_score_correction_bias",
                                                     with: ".mixer.e_score_correction_bias")
            }
            // Skip vision weights and multi-token prediction training artifacts
            if newKey.hasPrefix("vision_") || newKey.contains(".vision.") { continue }
            if newKey.hasPrefix("mtp.") || newKey.contains(".mtp.") { continue }
            newWeights[newKey] = value
        }
        return newWeights
    }
}
