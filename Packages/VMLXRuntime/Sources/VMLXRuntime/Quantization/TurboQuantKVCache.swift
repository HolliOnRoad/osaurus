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
    /// The actual codebook quantization is delegated to TurboQuantEncoder (Task 3.4).
    /// For now, this stores the float data and marks the phase transition.
    /// Once TurboQuantEncoder is implemented, this method will call it.
    public func compress() {
        precondition(phase == .fill, "Already compressed")
        guard _floatKeys != nil, _floatValues != nil else { return }

        // TODO: Actual compression via TurboQuantEncoder
        // For now, transition phase and keep float data accessible via getKeys/getValues
        // compressedKeys = TurboQuantEncoder.encode(keys: keys, bits: keyBits ?? 3, seed: config.seed)
        // compressedValues = TurboQuantEncoder.encode(values: values, bits: valueBits ?? 3, seed: config.seed)

        phase = .compressed

        // In production: _floatKeys and _floatValues would be released after compression
        // For now, keep them for getKeys/getValues to work until TQ encoder is implemented
    }

    // MARK: - Recompression

    /// Recompress after new tokens are appended during decode.
    /// Only compresses the delta (new tokens since last compression).
    public func recompress(newKeys: MLXArray, newValues: MLXArray) {
        guard phase == .compressed else {
            // In fill phase, just append
            appendFloat(keys: newKeys, values: newValues)
            return
        }

        offset += newKeys.shape[2]

        // TODO: Incremental compression of the delta
        // For now, append to float storage
        if let existing = _floatKeys {
            _floatKeys = concatenated([existing, newKeys], axis: 2)
            _floatValues = concatenated([_floatValues!, newValues], axis: 2)
        } else {
            _floatKeys = newKeys
            _floatValues = newValues
        }
    }

    // MARK: - Access

    /// Get keys for attention computation.
    /// In compressed phase, this will decompress on-the-fly (once encoder implemented).
    /// Currently returns float data.
    public func getKeys() -> MLXArray? {
        // TODO: In compressed phase, decompress from compressedKeys
        return _floatKeys
    }

    /// Get values for attention computation.
    public func getValues() -> MLXArray? {
        // TODO: In compressed phase, decompress from compressedValues
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
