import Foundation
import MLX

/// Phase of the TurboQuant KV cache lifecycle.
public enum TQPhase: Sendable {
    case fill       // Prefill: accumulating float16 KV (no compression)
    case compressed // Post-prefill: KV compressed to 3-bit, stays in GPU
}

/// TurboQuant KV cache for a single attention layer.
/// Two-phase operation: fill with float16 during prefill, compress after.
/// During decode, data stays compressed in GPU (zero decompression overhead).
public final class TurboQuantKVCache: @unchecked Sendable {

    /// Current phase.
    public private(set) var phase: TQPhase = .fill

    /// Float16 KV data (used during fill phase, cleared after compression).
    private var _floatKeys: MLXArray?
    private var _floatValues: MLXArray?

    /// Compressed KV data (populated after compression).
    public private(set) var compressedKeys: EncodedKeys?
    public private(set) var compressedValues: EncodedValues?

    /// Decoded buffer: float arrays decoded once after compress().
    /// Subsequent getKeys()/getValues() reads from these buffers — no re-decode.
    /// New tokens during decode append to the float window, not the buffer.
    /// This is the critical Phase 2 fix from Python VMLX TurboQuant.
    private var _decodedKeyBuffer: MLXArray?
    private var _decodedValueBuffer: MLXArray?

    /// Float window: new tokens appended during decode (after compression).
    /// getKeys()/getValues() returns concat(decodedBuffer, floatWindow).
    private var _floatWindowKeys: MLXArray?
    private var _floatWindowValues: MLXArray?

    /// Cache offset (number of tokens stored).
    public private(set) var offset: Int = 0

    /// Configuration for this layer's compression.
    public let config: TurboQuantConfig

    /// Layer index (for per-layer bit width resolution).
    public let layerIndex: Int

    /// Total number of layers in the model (for resolving negative critical layer indices).
    public let totalLayers: Int

    /// Bit widths for this specific layer (nil if SSM layer — shouldn't be wrapped in TQ).
    public let keyBits: Int?
    public let valueBits: Int?

    public init(config: TurboQuantConfig, layerIndex: Int, totalLayers: Int) {
        self.config = config
        self.layerIndex = layerIndex
        self.totalLayers = totalLayers
        self.keyBits = config.keyBits(forLayer: layerIndex, totalLayers: totalLayers)
        self.valueBits = config.valueBits(forLayer: layerIndex, totalLayers: totalLayers)
    }

    // MARK: - Fill Phase

    /// Append new KV data during prefill. Only valid in fill phase.
    public func appendFloat(keys: MLXArray, values: MLXArray) {
        precondition(phase == .fill, "Cannot append float data in compressed phase")

        if let existing = _floatKeys {
            // Concatenate along sequence dimension (dim 2 in [batch, heads, seq, dim])
            _floatKeys = concatenated([existing, keys], axis: 2)
            _floatValues = concatenated([_floatValues!, values], axis: 2)
        } else {
            _floatKeys = keys
            _floatValues = values
        }

        offset += keys.shape[2]  // Sequence dimension
    }

    // MARK: - Compression

    /// Compress float KV to TurboQuant format. Transitions from fill to compressed phase.
    /// Call this after prefill completes.
    ///
    /// Uses TurboQuantEncoder to perform random projection quantization:
    /// vectors are projected through a deterministic codebook, and the best
    /// codebook index + vector norm are stored for reconstruction.
    ///
    /// After encoding, decoded buffers are populated so subsequent getKeys()/getValues()
    /// reads from float buffers (no re-decode per step). This is the critical Phase 2 fix
    /// from Python VMLX TurboQuant — without it, post-compress generation is O(n*d^2) per step.
    public func compress() {
        precondition(phase == .fill, "Already compressed")
        guard let keys = _floatKeys, let values = _floatValues else { return }

        // Encode to compressed format
        compressedKeys = TurboQuantEncoder.encodeKeys(
            keys: keys, bits: keyBits ?? 3, seed: config.seed
        )
        compressedValues = TurboQuantEncoder.encodeValues(
            values: values, bits: valueBits ?? 3, seed: config.seed
        )

        // Decode once into persistent float buffers.
        // These buffers serve as the read-only compressed region for attention.
        // New tokens during decode will append to the float window instead.
        _decodedKeyBuffer = TurboQuantEncoder.decodeKeys(compressedKeys!, seed: config.seed)
        _decodedValueBuffer = TurboQuantEncoder.decodeValues(compressedValues!, seed: config.seed)

        // Initialize empty float windows for decode-phase tokens
        _floatWindowKeys = nil
        _floatWindowValues = nil

        phase = .compressed

        // Release original float data — decoded buffers + compressed are authoritative now
        _floatKeys = nil
        _floatValues = nil
    }

    // MARK: - Recompression

    /// Append new tokens during decode phase. Does NOT re-encode — just grows the float window.
    /// The float window is concatenated with the decoded buffer in getKeys()/getValues().
    /// This keeps decode at O(1) per step instead of O(n*d^2) re-encoding.
    public func appendDecodeTokens(newKeys: MLXArray, newValues: MLXArray) {
        guard phase == .compressed else {
            // In fill phase, just append to float data
            appendFloat(keys: newKeys, values: newValues)
            return
        }

        offset += newKeys.dim(2)

        // Append to float window (not compressed region)
        if let existingKeys = _floatWindowKeys {
            _floatWindowKeys = concatenated([existingKeys, newKeys], axis: 2)
            _floatWindowValues = concatenated([_floatWindowValues!, newValues], axis: 2)
        } else {
            _floatWindowKeys = newKeys
            _floatWindowValues = newValues
        }
    }

    // MARK: - Access

    /// Get keys for attention computation.
    /// In fill phase: returns raw float keys.
    /// In compressed phase: returns concat(decodedBuffer, floatWindow) — O(1), no re-decode.
    public func getKeys() -> MLXArray? {
        if phase == .compressed {
            guard let buffer = _decodedKeyBuffer else { return nil }
            if let window = _floatWindowKeys {
                return concatenated([buffer, window], axis: 2)
            }
            return buffer
        }
        return _floatKeys
    }

    /// Get values for attention computation.
    /// In fill phase: returns raw float values.
    /// In compressed phase: returns concat(decodedBuffer, floatWindow) — O(1), no re-decode.
    public func getValues() -> MLXArray? {
        if phase == .compressed {
            guard let buffer = _decodedValueBuffer else { return nil }
            if let window = _floatWindowValues {
                return concatenated([buffer, window], axis: 2)
            }
            return buffer
        }
        return _floatValues
    }

    /// Whether this cache has any data.
    public var isEmpty: Bool {
        offset == 0
    }

    /// Estimated memory usage in bytes.
    public var estimatedBytes: Int {
        var total = 0
        if let k = _floatKeys { total += k.nbytes }
        if let v = _floatValues { total += v.nbytes }
        if let ck = compressedKeys { total += ck.estimatedBytes }
        if let cv = compressedValues { total += cv.estimatedBytes }
        return total
    }

    /// Convert to a KVCacheLayer (for interop with HybridCache).
    public func toKVCacheLayer() -> KVCacheLayer? {
        guard let keys = getKeys(), let values = getValues() else { return nil }
        return KVCacheLayer(keys: keys, values: values, offset: offset)
    }

    /// Create from an existing KVCacheLayer (wrap existing float data for future compression).
    public static func fromKVCacheLayer(
        _ layer: KVCacheLayer,
        config: TurboQuantConfig,
        layerIndex: Int,
        totalLayers: Int
    ) -> TurboQuantKVCache {
        let tqCache = TurboQuantKVCache(config: config, layerIndex: layerIndex, totalLayers: totalLayers)
        tqCache._floatKeys = layer.keys
        tqCache._floatValues = layer.values
        tqCache.offset = layer.offset
        return tqCache
    }
}
