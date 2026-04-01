//
//  SSM.swift
//  VMLXRuntime
//
//  SSD (Structured State Space Duality) operations for Mamba2 models.
//  Ported from mlx-lm's SSM.swift / ssm.py.
//
//  Two paths:
//  - ssmAttn: parallel SSD using surrogate attention matrix (for seq_len > 1)
//  - ssmUpdateKernel: single-step Metal kernel (for seq_len == 1, generation)
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Time Step Computation

@_optimize(speed)
public func vmlxComputeDt(_ dt: MLXArray, _ dtBias: MLXArray, _ timeStepLimit: (Float, Float)) -> MLXArray {
    let dt = softplus(dt + dtBias)
    return MLX.clip(dt, min: timeStepLimit.0, max: timeStepLimit.1)
}

// MARK: - Metal Kernel (single-step decode)

private func makeSSMKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        auto n = thread_position_in_grid.z;
        auto h_idx = n % H;
        auto g_idx = n / G;
        constexpr int n_per_t = Ds / 32;

        auto x = X + n * Dh;
        out += n * Dh;
        auto i_state = state_in + n * Dh * Ds;
        auto o_state = state_out + n * Dh * Ds;

        auto C_ = C + g_idx * Ds;
        auto B_ = B + g_idx * Ds;

        auto ds_idx = thread_position_in_threadgroup.x;
        auto d_idx = thread_position_in_grid.y;

        auto dt_ = static_cast<float>(dt[n]);
        auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
        auto dA = fast::exp(A * dt_);

        float acc = 0.0;
        auto x_ = static_cast<float>(x[d_idx]);

        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * ds_idx + i;
            auto idx = d_idx * Ds + s_idx;
            auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
            auto state = dA * i_state[idx] + dB_by_x;
            o_state[idx] = static_cast<T>(state);
            acc += state * C_[s_idx];
        }
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
        }
    """
    return MLXFast.metalKernel(
        name: "vmlx_ssm_kernel",
        inputNames: ["X", "A_log", "B", "C", "D", "dt", "state_in"],
        outputNames: ["out", "state_out"],
        source: source
    )
}

final class VMLXSSMKernelManager: Sendable {
    static let shared = VMLXSSMKernelManager()
    let kernel: MLXFast.MLXFastKernel?
    private init() { kernel = makeSSMKernel() }
}

/// Single-step SSM update via Metal kernel (seq_len == 1, generation).
public func vmlxSSMUpdateKernel(
    hiddenStates: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray,
    timeStepLimit: (Float, Float)
) -> (MLXArray, MLXArray) {
    let (n, _, h, d) = hiddenStates.shape4
    let inputType = hiddenStates.dtype
    let (hb, ds) = (B.dim(-2), B.dim(-1))
    let dt = vmlxComputeDt(dt, dtBias, timeStepLimit)

    guard let kernel = VMLXSSMKernelManager.shared.kernel else {
        fatalError("SSM Metal kernel not available")
    }

    let outputs = kernel(
        [hiddenStates, ALog, B, C, D, dt, state],
        template: [
            ("T", inputType),
            ("Dh", d),
            ("Ds", ds),
            ("H", h),
            ("G", h / hb),
        ],
        grid: (32, d, h * n),
        threadGroup: (32, 8, 1),
        outputShapes: [[n, 1, h, d], state.shape],
        outputDTypes: [inputType, inputType]
    )
    return (outputs[0], outputs[1])
}

// MARK: - SSD Parallel Scan (seq_len > 1, prefill)

/// Segment sum: compute pairwise cumulative sums for causal structure.
public func vmlxSegsum(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let l = x.dim(-1)
    var x = x
    if let mask {
        x = x * MLX.expandedDimensions(mask, axis: 1)
    }
    x = MLX.repeated(x[.ellipsis, .newAxis], count: l, axis: -1)
    x = MLX.tril(x, k: -1)
    var xSegsum = MLX.cumsum(x, axis: -2)
    if let mask {
        xSegsum = which(
            MLX.expandedDimensions(mask, axis: 1)[.ellipsis, .newAxis, 0...]
                * MLX.expandedDimensions(mask, axis: 1)[.ellipsis, .newAxis],
            xSegsum,
            MLXArray(-Float.infinity)
        )
    }
    return xSegsum
}

/// SSD-SSM forward pass using surrogate attention matrix.
/// Processes in chunks of `step` tokens for memory efficiency.
public func vmlxSSMAttn(
    x: MLXArray,        // [batch, seq, heads, head_dim]
    ALog: MLXArray,     // [heads]
    B: MLXArray,        // [batch, seq, groups, state_size]
    C: MLXArray,        // [batch, seq, groups, state_size]
    D: MLXArray,        // [heads]
    dt: MLXArray,       // [batch, seq, heads]
    dtBias: MLXArray,   // [heads]
    state: MLXArray? = nil,   // [batch, groups, head_dim, state_size]
    timeStepLimit: (Float, Float) = (0.001, 0.1),
    mask: MLXArray? = nil,
    step: Int = 256
) -> (MLXArray, MLXArray) {
    let (b, l, h, dh) = x.shape4
    let g = B.dim(2)
    let d = B.dim(3)

    let dt = vmlxComputeDt(dt, dtBias, timeStepLimit)
    let repeats = h / g
    let A = -MLX.exp(ALog)
    let dtA = dt * A.reshaped(1, 1, -1)
    let dtx = dt.reshaped(b, l, h, 1) * x

    func _step(
        dtx: MLXArray, dtA: MLXArray,
        B: MLXArray, C: MLXArray,
        state: MLXArray?, mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let s = dtx.dim(1)
        let Bt = MLX.transposed(B, axes: [0, 2, 3, 1])

        var CB = MLX.swappedAxes(C, 1, 2).matmul(Bt)
        CB = MLX.repeated(CB, count: repeats, axis: 1)

        let decay = MLX.exp(vmlxSegsum(dtA.swappedAxes(1, 2), mask: mask))
        let surrogateAttention = MLX.tril(CB * decay, k: 0)

        var y = surrogateAttention.matmul(dtx.swappedAxes(1, 2))
        y = MLX.swappedAxes(y, 1, 2)

        var decayLast = decay[0..., 0..., (-1)..., 0...].transposed(0, 3, 1, 2)
        let Brep = MLX.repeated(Bt, count: h / g, axis: 1).swappedAxes(2, 3)
        let dtxdecay = (dtx * decayLast).swappedAxes(1, 2).swappedAxes(2, 3)
        var nextState = dtxdecay.matmul(Brep)

        if let state {
            let expDtACumsum = MLX.exp(MLX.cumsum(dtA, axis: -2))
            nextState = nextState + expDtACumsum[0..., -1, 0..., .newAxis, .newAxis] * state
            let stateR = state.reshaped(b, 1, g, repeats, dh, d)
            let Cr = C.reshaped(b, s, g, 1, d, 1)
            let yPrev = stateR.matmul(Cr).squeezed(axis: -1).flattened(start: 2, end: 3)
            y = y + expDtACumsum[.ellipsis, .newAxis] * yPrev
        }

        return (y, nextState)
    }

    var currentState = state
    var ys: [MLXArray] = []
    var pos = 0
    while pos < l {
        let end = min(pos + step, l)
        let (y, ns) = _step(
            dtx: dtx[0..., pos..<end],
            dtA: dtA[0..., pos..<end],
            B: B[0..., pos..<end],
            C: C[0..., pos..<end],
            state: currentState,
            mask: mask
        )
        currentState = ns
        ys.append(y)
        pos = end
    }

    let yFull = concatenated(ys, axis: 1) + x * D.reshaped(1, 1, h, 1)
    return (yFull, currentState!)
}

// MARK: - Unified SSM Update

/// Dispatches to Metal kernel for single-token decode, SSD parallel scan for prefill.
public func vmlxSSMUpdate(
    hiddenStates: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    timeStepLimit: (Float, Float) = (0.001, 0.1),
    mask: MLXArray? = nil,
    step: Int = 256
) -> (MLXArray, MLXArray) {
    let seqLen = hiddenStates.dim(1)

    if seqLen == 1,
       let state,
       VMLXSSMKernelManager.shared.kernel != nil
    {
        return vmlxSSMUpdateKernel(
            hiddenStates: hiddenStates,
            ALog: ALog, B: B, C: C, D: D,
            dt: dt, dtBias: dtBias,
            state: state,
            timeStepLimit: timeStepLimit
        )
    }

    return vmlxSSMAttn(
        x: hiddenStates,
        ALog: ALog, B: B, C: C, D: D,
        dt: dt, dtBias: dtBias,
        state: state,
        timeStepLimit: timeStepLimit,
        mask: mask,
        step: step
    )
}
