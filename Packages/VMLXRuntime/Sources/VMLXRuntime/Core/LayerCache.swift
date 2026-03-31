import MLX

// MARK: - Attention Layer (KV Cache)

/// A single attention layer's key-value cache state.
/// Positional — can be sliced/truncated to shorter token counts.
public struct KVCacheLayer: @unchecked Sendable {
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
/// Path-dependent — CANNOT be truncated. Includes all prior tokens' contributions.
public struct SSMStateLayer: @unchecked Sendable {
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
/// TurboQuant-compressed layers use `.compressedAttention` for 3-bit storage.
public enum LayerCacheEntry: @unchecked Sendable {
    case attention(KVCacheLayer)
    case ssm(SSMStateLayer)
    /// TurboQuant-compressed attention layer (Phase 5).
    /// Stores 3-bit encoded keys/values + token offset.
    /// On cache fetch, decode to float and load into VMLXKVCacheSimple.
    case compressedAttention(EncodedKeys, EncodedValues, Int)

    public var isAttention: Bool {
        switch self {
        case .attention, .compressedAttention: return true
        default: return false
        }
    }

    public var isSSM: Bool {
        if case .ssm = self { return true }
        return false
    }

    public var isCompressed: Bool {
        if case .compressedAttention = self { return true }
        return false
    }

    public var canTruncate: Bool {
        switch self {
        case .attention: return true
        case .compressedAttention: return false  // Compressed data can't be partially truncated
        case .ssm(let s): return s.canTruncate
        }
    }

    public var estimatedBytes: Int {
        switch self {
        case .attention(let kv): return kv.estimatedBytes
        case .ssm(let ssm): return ssm.estimatedBytes
        case .compressedAttention(let ek, let ev, _):
            return ek.estimatedBytes + ev.estimatedBytes
        }
    }

    /// Truncate if possible. Returns nil for SSM and compressed layers.
    public func truncated(to tokenCount: Int) -> LayerCacheEntry? {
        switch self {
        case .attention(let kv):
            return .attention(kv.truncated(to: tokenCount))
        case .ssm, .compressedAttention:
            return nil  // Cannot truncate cumulative SSM or compressed data
        }
    }
}
