import Foundation
import MLX

/// TurboQuant-compressed value cache for a single attention layer.
/// Simpler than keys — no QJL residual correction needed.
public struct EncodedValues: @unchecked Sendable {
    /// Packed codebook indices (uint32).
    public let indicesPacked: MLXArray

    /// Per-vector norms (float16).
    public let vectorNorms: MLXArray

    /// Original tensor shape before compression.
    public let shape: [Int]

    /// Bits per codebook index.
    public let indexBits: Int

    /// Codebook seed used during encoding. Required for correct decoding.
    /// Note: values use seed+1 internally (different codebook from keys).
    public let seed: Int

    /// Float16 sink tokens preserved at full precision.
    /// Shape: [batch, heads, sinkCount, head_dim] or nil.
    public let sinkData: MLXArray?

    public var sinkCount: Int { sinkData?.dim(2) ?? 0 }

    public init(
        indicesPacked: MLXArray,
        vectorNorms: MLXArray,
        shape: [Int],
        indexBits: Int,
        seed: Int = 42,
        sinkData: MLXArray? = nil
    ) {
        self.indicesPacked = indicesPacked
        self.vectorNorms = vectorNorms
        self.shape = shape
        self.indexBits = indexBits
        self.seed = seed
        self.sinkData = sinkData
    }

    /// Estimated memory in bytes (compressed + sink data).
    public var estimatedBytes: Int {
        var total = indicesPacked.nbytes + vectorNorms.nbytes
        if let sink = sinkData { total += sink.nbytes }
        return total
    }

    /// Number of encoded vectors.
    public var vectorCount: Int {
        vectorNorms.size
    }

    /// Compression ratio vs float16 original.
    public var compressionRatio: Float {
        guard shape.count == 4 else { return 1.0 }
        let originalBytes = shape.reduce(1, *) * 2
        guard estimatedBytes > 0 else { return Float.infinity }
        return Float(originalBytes) / Float(estimatedBytes)
    }
}
