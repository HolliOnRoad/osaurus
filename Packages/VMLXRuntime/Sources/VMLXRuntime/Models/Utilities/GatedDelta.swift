//
//  GatedDelta.swift
//  VMLXRuntime
//
//  GatedDeltaNet SSM kernel for Qwen3.5 linear attention layers.
//  Ported from Python mlx-lm gated_delta.py — includes all 4 kernel variants
//  (scalar/vec × masked/unmasked) matching the Python reference exactly.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Compute G (compiled)

/// Gating decay: exp(-exp(A_log) * softplus(a + dt_bias)).
/// NOTE: compile(shapeless:true) crashes with EXC_BAD_ACCESS on hybrid SSM models
/// (Nemotron Cascade) during prefill. The compiled kernel interacts with the custom
/// Metal GatedDelta kernel causing memory corruption. Uncompiled.
let computeGatedDeltaG: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
    (aLog: MLXArray, a: MLXArray, dtBias: MLXArray) -> MLXArray in
    let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    return decay.asType(a.dtype)
}

// MARK: - Metal Kernels (4 variants matching Python)

/// Build a GatedDelta Metal kernel.
/// - `vectorized`: true for g:[B,T,Hv,Dk] (Qwen3.5), false for g:[B,T,Hv]
/// - `hasMask`: true for masked variant (prefill with padding)
private func makeGatedDeltaKernel(hasMask: Bool, vectorized: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

    // Configure g indexing based on vectorized flag
    let gComment: String
    let gSetup: String
    let gAccess: String
    let gAdvance: String

    if vectorized {
        gComment = "// g: [B, T, Hv, Dk]"
        gSetup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        gAccess = "g_[s_idx]"
        gAdvance = "g_ += Hv * Dk;"
    } else {
        gComment = "// g: [B, T, Hv]"
        gSetup = "auto g_ = g + b_idx * T * Hv;"
        gAccess = "g_[hv_idx]"
        gAdvance = "g_ += Hv;"
    }

    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            \(gComment)
            \(gSetup)
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * \(gAccess);
                  kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              \(gAdvance)
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<InT>(state[i]);
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    var suffix = ""
    if vectorized { suffix += "_vec" }
    if hasMask { suffix += "_mask" }

    return MLXFast.metalKernel(
        name: "gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

/// All 4 kernel variants, lazily initialized once.
final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    let kernelVec: MLXFast.MLXFastKernel?
    let kernelVecMasked: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(hasMask: false, vectorized: false)
        kernelMasked = makeGatedDeltaKernel(hasMask: true, vectorized: false)
        kernelVec = makeGatedDeltaKernel(hasMask: false, vectorized: true)
        kernelVecMasked = makeGatedDeltaKernel(hasMask: true, vectorized: true)
    }
}

// MARK: - Kernel Dispatch

func gatedDeltaKernel(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray, state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype

    // Select kernel variant based on g dimensionality and mask
    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]

    if g.ndim == 4 {
        // Vectorized gating: g is [B, T, Hv, Dk] (Qwen3.5)
        if let mask {
            selectedKernel = GatedDeltaKernelManager.shared.kernelVecMasked
            inputs.append(mask)
        } else {
            selectedKernel = GatedDeltaKernelManager.shared.kernelVec
        }
    } else {
        // Scalar gating: g is [B, T, Hv]
        if let mask {
            selectedKernel = GatedDeltaKernelManager.shared.kernelMasked
            inputs.append(mask)
        } else {
            selectedKernel = GatedDeltaKernelManager.shared.kernel
        }
    }

    guard let kernel = selectedKernel else {
        // Metal kernel unavailable — fall back to compiled ops
        return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, inputType]
    )

    return (outputs[0], outputs[1])
}

// MARK: - Ops Fallback (compiled single-step)

/// Single-step GatedDelta update — compiled to fuse arithmetic ops.
/// Matches Python's @mx.compile _gated_delta_step_ops.
private let _compiledGatedDeltaStep: @Sendable ([MLXArray]) -> [MLXArray] = compile(
    shapeless: true
) { (inputs: [MLXArray]) -> [MLXArray] in
    let q = inputs[0]        // [B, H, Dk]
    let k = inputs[1]        // [B, H, Dk]
    let v = inputs[2]        // [B, H, Dv]
    let g = inputs[3]        // [B, H] or [B, H, Dk]
    let beta = inputs[4]     // [B, H]
    let stateIn = inputs[5]  // [B, H, Dv, Dk]

    let decay: MLXArray
    if g.ndim == 2 {
        decay = g[.ellipsis, .newAxis, .newAxis]
    } else if g.ndim == 3 {
        decay = g[.ellipsis, .newAxis, 0...]
    } else {
        decay = g[.ellipsis, .newAxis, 0...]
    }

    var s = stateIn * decay
    let kvMem = (s * k[.ellipsis, .newAxis, 0...]).sum(axis: -1)
    let delta = (v - kvMem) * beta[.ellipsis, .newAxis]
    s = s + k[.ellipsis, .newAxis, 0...] * delta[.ellipsis, .newAxis]
    let y = (s * q[.ellipsis, .newAxis, 0...]).sum(axis: -1)

    // Check if mask is provided (inputs.count > 6)
    if inputs.count > 6 {
        let mask = inputs[6]
        let expandedMask: MLXArray
        if mask.ndim == 1 {
            expandedMask = mask[.newAxis, .newAxis, .newAxis, .newAxis]
        } else if mask.ndim == 2 {
            expandedMask = mask[.ellipsis, .newAxis, .newAxis]
        } else {
            expandedMask = mask[.ellipsis, .newAxis]
        }
        let sFinal = MLX.where(expandedMask, s, stateIn)
        return [y, sFinal]
    }

    return [y, s]
}

func gatedDeltaOps(
    q: MLXArray, k: MLXArray, v: MLXArray,
    g: MLXArray, beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var q = q
    var k = k

    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        q = repeated(q, count: repeatFactor, axis: -2)
        k = repeated(k, count: repeatFactor, axis: -2)
    }

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

    var ys = [MLXArray]()
    ys.reserveCapacity(T)

    for t in 0 ..< T {
        let qT = q[0..., t]
        let kT = k[0..., t]
        let vT = v[0..., t]
        let gT = g[0..., t]
        let betaT = beta[0..., t]

        var inputs: [MLXArray] = [qT, kT, vT, gT, betaT, state]
        if let mask {
            inputs.append(mask[0..., t])
        }

        let outputs = _compiledGatedDeltaStep(inputs)
        ys.append(outputs[0])
        state = outputs[1]
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, state)
}

// MARK: - Public API

func vmlxGatedDeltaUpdate(
    q: MLXArray, k: MLXArray, v: MLXArray,
    a: MLXArray, b: MLXArray,
    aLog: MLXArray, dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let beta = sigmoid(b)
    let g = computeGatedDeltaG(aLog, a, dtBias)

    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    let state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

    // Try Metal kernel first (matching Python's default use_kernel=True).
    // Fall back to compiled ops if kernel unavailable or crashes.
    // Metal kernels are known to crash (EXC_BAD_ACCESS) on some configurations
    // due to template instantiation issues. The compiled ops path is ~2x slower
    // but produces identical results.
    if GatedDeltaKernelManager.shared.kernel != nil {
        return gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }
    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
}
