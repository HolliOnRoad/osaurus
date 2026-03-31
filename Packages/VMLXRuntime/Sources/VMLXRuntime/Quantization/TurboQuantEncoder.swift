import Foundation
import MLX

/// TurboQuant encoder — compresses float KV cache to codebook indices
/// using random projection quantization (QJL — Quantized Johnson-Lindenstrauss).
///
/// Algorithm overview:
/// 1. Generate a random projection matrix (codebook) from a deterministic seed
/// 2. Normalize each vector and project through the codebook
/// 3. Select the best codebook entry per vector (argmax of absolute projections)
/// 4. Store the index, sign, vector norm for reconstruction
/// 5. On decode, look up the codebook column and scale by the stored norm
public struct TurboQuantEncoder: Sendable {

    // MARK: - Key Encoding

    /// Encode float16 keys to compressed format.
    /// Default number of sink tokens to preserve at full precision.
    public static let defaultSinkTokens = 4

    /// - Parameters:
    ///   - keys: Float16 key tensor, shape [batch, heads, tokens, dim]
    ///   - bits: Codebook index bits (3-8). Determines codebook size = 2^bits.
    ///   - seed: Random seed for reproducible codebook generation
    ///   - sinkTokens: Number of leading tokens to keep at full precision (default 4).
    ///     These "attention sinks" (BOS/system prompt) are stored as float alongside
    ///     the compressed remainder. Set to 0 to compress everything.
    /// - Returns: EncodedKeys with packed indices, QJL signs, norms, and sink data
    public static func encodeKeys(
        keys: MLXArray,
        bits: Int = 3,
        seed: Int = 42,
        sinkTokens: Int = defaultSinkTokens
    ) -> EncodedKeys {
        let shape = keys.shape  // [batch, heads, tokens, head_dim]
        let seqLen = shape[2]

        // Extract sink tokens (first N) as float — preserved at full precision
        let sinkData: MLXArray?
        let compressKeys: MLXArray
        if sinkTokens > 0 && seqLen > sinkTokens {
            sinkData = keys[.ellipsis, 0..<sinkTokens, 0...]
            compressKeys = keys[.ellipsis, sinkTokens..., 0...]
        } else {
            sinkData = nil
            compressKeys = keys
        }

        let compressShape = compressKeys.shape
        let headDim = compressShape[compressShape.count - 1]
        let numVectors = compressShape.dropLast().reduce(1, *)

        // Work in float32 for numerical stability (compress only non-sink tokens)
        let flat = compressKeys.asType(.float32).reshaped([numVectors, headDim])

        // Compute per-vector L2 norms: [numVectors]
        let norms = (flat * flat).sum(axis: -1).sqrt()

        // Normalize to unit length: [numVectors, headDim]
        let epsilon = MLXArray(Float(1e-8))
        let safeDenom = (norms + epsilon).expandedDimensions(axis: -1)
        let normalized = flat / safeDenom

        // Generate random projection matrix (codebook) from seed
        // Shape: [headDim, numCodewords] where numCodewords = 2^bits
        let numCodewords = 1 << bits
        let rngKey = MLXRandom.key(UInt64(seed))
        let projection = MLXRandom.normal([headDim, numCodewords], key: rngKey)

        // Project normalized vectors through codebook: [numVectors, numCodewords]
        let projections = matmul(normalized, projection)

        // For each vector, find the codebook entry with largest absolute projection
        // This is the "best match" index: [numVectors]
        let bestIndices = abs(projections).argMax(axis: -1)  // [numVectors]

        // Pack indices as uint32
        let packedIndices = bestIndices.asType(.uint32)

        // QJL sign bits: store the sign of the projection at the selected index
        // We gather the projection value at each vector's best index
        // Use a simple approach: for each vector i, sign of projections[i, bestIndices[i]]
        // We can compute this via: sign = (projection_at_best > 0) ? 1 : 0
        //
        // Efficient approach: take the argmax projections and check sign
        // projections shape is [numVectors, numCodewords], bestIndices is [numVectors]
        // We need a diagonal gather. Use element-wise: gather projection values at best indices.
        let flatProjections = projections.reshaped([-1])  // [numVectors * numCodewords]
        let offsets = MLXArray(Array(0..<numVectors)).asType(.int32) * MLXArray(Int32(numCodewords))
        let bestIdxInt = bestIndices.asType(.int32)
        let gatherIndices = offsets + bestIdxInt
        let selectedProjections = take(flatProjections, gatherIndices)  // [numVectors]

        // Sign bits: 1 if positive, 0 if negative
        let signBits = (selectedProjections .> MLXArray(Float(0.0))).asType(.uint32)

        // Residual norms: approximate the quantization error
        // |original - reconstructed| / |original|
        // For a rough estimate, use the magnitude of the non-selected projection energy
        let bestMagnitudes = abs(selectedProjections)
        let totalEnergy = (projections * projections).sum(axis: -1).sqrt()
        let residualNorms = ((totalEnergy - bestMagnitudes) * MLXArray(Float(0.1))).asType(.float16)

        return EncodedKeys(
            indicesPacked: packedIndices.reshaped([-1]),
            qjlPacked: signBits.reshaped([-1]),
            residualNorms: residualNorms.reshaped([-1]),
            vectorNorms: norms.asType(.float16).reshaped([-1]),
            shape: compressShape,  // Shape of compressed portion (excludes sink tokens)
            indexBits: bits,
            seed: seed,
            sinkData: sinkData
        )
    }

    // MARK: - Value Encoding

    /// Encode float16 values to compressed format.
    /// Simpler than keys -- no QJL residual correction needed.
    public static func encodeValues(
        values: MLXArray,
        bits: Int = 3,
        seed: Int = 42,
        sinkTokens: Int = defaultSinkTokens
    ) -> EncodedValues {
        let shape = values.shape  // [batch, heads, tokens, head_dim]
        let seqLen = shape[2]

        // Extract sink tokens as float
        let sinkData: MLXArray?
        let compressValues: MLXArray
        if sinkTokens > 0 && seqLen > sinkTokens {
            sinkData = values[.ellipsis, 0..<sinkTokens, 0...]
            compressValues = values[.ellipsis, sinkTokens..., 0...]
        } else {
            sinkData = nil
            compressValues = values
        }

        let compressShape = compressValues.shape
        let headDim = compressShape[compressShape.count - 1]
        let numVectors = compressShape.dropLast().reduce(1, *)

        // Work in float32
        let flat = compressValues.asType(.float32).reshaped([numVectors, headDim])

        // Per-vector norms
        let norms = (flat * flat).sum(axis: -1).sqrt()

        // Normalize
        let epsilon = MLXArray(Float(1e-8))
        let safeDenom = (norms + epsilon).expandedDimensions(axis: -1)
        let normalized = flat / safeDenom

        // Generate codebook from seed (use seed + 1 to get different codebook than keys)
        let numCodewords = 1 << bits
        let rngKey = MLXRandom.key(UInt64(seed + 1))
        let projection = MLXRandom.normal([headDim, numCodewords], key: rngKey)

        // Project and find best codebook entry
        let projections = matmul(normalized, projection)
        let bestIndices = abs(projections).argMax(axis: -1)
        let packedIndices = bestIndices.asType(.uint32)

        return EncodedValues(
            indicesPacked: packedIndices.reshaped([-1]),
            vectorNorms: norms.asType(.float16).reshaped([-1]),
            shape: compressShape,
            indexBits: bits,
            seed: seed,
            sinkData: sinkData
        )
    }

    // MARK: - Key Decoding

    /// Decode compressed keys back to float16 for attention computation.
    /// Reconstructs vectors by looking up codebook columns and scaling by stored norms.
    public static func decodeKeys(_ encoded: EncodedKeys, seed: Int = 42) -> MLXArray {
        let headDim = encoded.shape.last ?? 128
        let numCodewords = 1 << encoded.indexBits

        // Regenerate the same codebook from the same seed
        let rngKey = MLXRandom.key(UInt64(seed))
        let projection = MLXRandom.normal([headDim, numCodewords], key: rngKey)

        // projection is [headDim, numCodewords]
        // Transpose to [numCodewords, headDim] so we can index rows by codebook index
        let codebook = projection.transposed()  // [numCodewords, headDim]

        // Look up the codebook vector for each stored index
        let indices = encoded.indicesPacked.asType(.int32)
        let codebookVectors = take(codebook, indices, axis: 0)  // [numVectors, headDim]

        // Apply QJL sign correction: flip vectors whose projection was negative
        // signBits: 1 = positive (keep), 0 = negative (negate)
        // Convert: sign = 2 * signBits - 1 => +1 or -1
        let signs = encoded.qjlPacked.asType(.float32) * MLXArray(Float(2.0)) - MLXArray(Float(1.0))
        let signedVectors = codebookVectors * signs.expandedDimensions(axis: -1)

        // Scale by stored vector norms
        let norms = encoded.vectorNorms.asType(.float32)
        let scaled = signedVectors * norms.expandedDimensions(axis: -1)

        // Reshape back to compressed shape and convert to float16
        var decoded = scaled.reshaped(encoded.shape).asType(.float16)

        // Prepend sink tokens if present
        if let sink = encoded.sinkData {
            decoded = concatenated([sink, decoded], axis: 2)
        }

        return decoded
    }

    // MARK: - Value Decoding

    /// Decode compressed values back to float16.
    public static func decodeValues(_ encoded: EncodedValues, seed: Int = 42) -> MLXArray {
        let headDim = encoded.shape.last ?? 128
        let numCodewords = 1 << encoded.indexBits

        // Regenerate codebook (seed + 1 to match encodeValues)
        let rngKey = MLXRandom.key(UInt64(seed + 1))
        let projection = MLXRandom.normal([headDim, numCodewords], key: rngKey)

        let codebook = projection.transposed()  // [numCodewords, headDim]

        // Look up codebook vectors
        let indices = encoded.indicesPacked.asType(.int32)
        let codebookVectors = take(codebook, indices, axis: 0)  // [numVectors, headDim]

        // Scale by stored norms
        let norms = encoded.vectorNorms.asType(.float32)
        let scaled = codebookVectors * norms.expandedDimensions(axis: -1)

        var decoded = scaled.reshaped(encoded.shape).asType(.float16)

        // Prepend sink tokens if present
        if let sink = encoded.sinkData {
            decoded = concatenated([sink, decoded], axis: 2)
        }

        return decoded
    }
}
