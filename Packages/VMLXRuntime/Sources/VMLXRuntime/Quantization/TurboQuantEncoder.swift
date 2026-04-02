import Foundation
import MLX
import MLXRandom

/// TurboQuant encoder/decoder — per-coordinate scalar quantization with QJL correction.
///
/// Algorithm (matching Python VMLX `jang_tools.turboquant.pipeline`):
///   Keys (b bits = (b-1) MSE + 1 QJL):
///     1. Normalize to unit sphere, store vector norms
///     2. Randomized Hadamard rotation (spreads energy across coordinates)
///     3. Per-coordinate scalar quantize via Lloyd-Max codebook (b-1 bits)
///     4. QJL 1-bit correction on residual (unbiased inner products)
///     5. Pack indices + signs + norms
///   Values (b bits, MSE only):
///     1. Normalize, store norms
///     2. Hadamard rotate
///     3. Per-coordinate scalar quantize (b bits)
///     4. Pack indices + norms
///
/// Reference: TurboQuant (arXiv:2504.19874)
public struct TurboQuantEncoder: Sendable {

    /// Default number of sink tokens preserved at full precision.
    public static let defaultSinkTokens = 4

    // MARK: - Precomputed State

    /// Precomputed encoder state for a given (dim, keyBits, valueBits, seed).
    /// Create once per model layer configuration, reuse across encode/decode calls.
    public struct EncoderState: @unchecked Sendable {
        public let dim: Int
        public let keyBits: Int
        public let valueBits: Int
        public let seed: Int

        /// Hadamard rotation signs (deterministic from seed)
        public let rotationSigns: MLXArray

        /// Lloyd-Max codebook for keys (b-1 bits, 2^(b-1) centroids)
        public let keyCodebook: [Float]
        public let keyIndexBits: Int

        /// Lloyd-Max codebook for values (b bits, 2^b centroids)
        public let valueCodebook: [Float]
        public let valueIndexBits: Int

        /// QJL projection matrix S (dim × dim, for key residual correction)
        public let qjlS: MLXArray

        public init(dim: Int, keyBits: Int = 3, valueBits: Int = 3, seed: Int = 42) {
            self.dim = dim
            self.keyBits = keyBits
            self.valueBits = valueBits
            self.seed = seed

            self.rotationSigns = TQHadamard.generateRandomSigns(dim: dim, seed: seed)

            let kMseBits = max(keyBits - 1, 1)
            self.keyCodebook = TQCodebook.computeCodebook(dim: dim, bits: kMseBits)
            self.keyIndexBits = kMseBits

            self.valueCodebook = TQCodebook.computeCodebook(dim: dim, bits: valueBits)
            self.valueIndexBits = valueBits

            self.qjlS = TQQJL.generateProjection(dim: dim, seed: seed + 1000)
        }
    }

    // MARK: - Encode Keys

    /// Compress float keys to TurboQuant format.
    ///
    /// - Parameters:
    ///   - keys: Float16/32 key tensor, shape [batch, heads, tokens, head_dim]
    ///   - state: Precomputed encoder state (codebooks, rotation signs, QJL matrix)
    ///   - sinkTokens: Number of leading tokens to preserve at full precision (default 4)
    /// - Returns: EncodedKeys with packed per-coordinate indices, QJL signs, and norms
    public static func encodeKeys(
        _ keys: MLXArray,
        state: EncoderState,
        sinkTokens: Int = defaultSinkTokens
    ) -> EncodedKeys {
        let origShape = keys.shape  // [batch, heads, tokens, head_dim]
        let seqLen = origShape[2]
        let dim = origShape[origShape.count - 1]

        // Extract sink tokens at full precision
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

        // Step 1: Normalize to unit sphere
        let vectorNorms = (compressKeys * compressKeys).sum(axis: -1, keepDims: true).sqrt()
        let keysUnit = compressKeys / (vectorNorms + 1e-8)

        // Step 2: Randomized Hadamard rotation
        let keysRotated = TQHadamard.hadamardRotate(keysUnit, signs: state.rotationSigns)

        // Step 3: Per-coordinate MSE quantization (b-1 bits)
        let flatRotated = keysRotated.asType(.float32).reshaped([-1, dim])
        let mseIndices = TQCodebook.quantizeScalar(flatRotated, codebook: state.keyCodebook)
        let mseDequant = TQCodebook.dequantizeScalar(mseIndices, codebook: state.keyCodebook)

        // Step 4: QJL 1-bit correction on residual
        let residual = flatRotated - mseDequant
        let projected = matmul(residual, state.qjlS.transposed())
        let qjlSigns = which(projected .>= 0, MLXArray(Float(1.0)), MLXArray(Float(-1.0)))
        let residualNorms = (residual * residual).sum(axis: -1, keepDims: true).sqrt()

        // Step 5: Pack
        let packedIndices = TQBitPack.packBits(mseIndices.reshaped(-1), bits: state.keyIndexBits)
        let packedQJL = TQBitPack.packSigns(qjlSigns.reshaped(-1))

        return EncodedKeys(
            indicesPacked: packedIndices,
            qjlPacked: packedQJL,
            residualNorms: residualNorms
                .reshaped(Array(compressShape.dropLast()) + [1]).asType(.float16),
            vectorNorms: vectorNorms.asType(.float16),
            shape: compressShape,
            indexBits: state.keyIndexBits,
            seed: state.seed,
            sinkData: sinkData
        )
    }

    // MARK: - Decode Keys

    /// Decompress keys from TurboQuant format.
    public static func decodeKeys(_ encoded: EncodedKeys, state: EncoderState) -> MLXArray {
        let origShape = encoded.shape
        let dim = origShape[origShape.count - 1]
        let nElements = origShape.reduce(1, *)

        // Step 1: Unpack
        let flatIndices = TQBitPack.unpackBits(
            encoded.indicesPacked, bits: encoded.indexBits, nElements: nElements
        ).reshaped([-1, dim])
        let flatQJL = TQBitPack.unpackSigns(
            encoded.qjlPacked, nElements: nElements
        ).reshaped([-1, dim])
        let flatResNorms = encoded.residualNorms.asType(.float32).reshaped([-1, 1])
        let flatVecNorms = encoded.vectorNorms.asType(.float32).reshaped([-1, 1])

        // Step 2: Codebook lookup (per-coordinate)
        let mseDequant = TQCodebook.dequantizeScalar(flatIndices, codebook: state.keyCodebook)

        // Step 3: QJL correction
        let qjlScale = Float(Foundation.sqrt(Double.pi / 2.0)) / Float(dim)
        let qjlDequant = MLXArray(qjlScale) * flatResNorms * matmul(flatQJL, state.qjlS)

        // Step 4: Combine MSE + QJL, inverse Hadamard
        let reconstructedRotated = (mseDequant + qjlDequant).reshaped(origShape)
        let reconstructedUnit = TQHadamard.hadamardInverse(
            reconstructedRotated, signs: state.rotationSigns)

        // Step 5: Scale by stored norms
        var decoded = (reconstructedUnit * flatVecNorms.reshaped(
            Array(origShape.dropLast()) + [1])).asType(.float16)

        // Prepend sink tokens if present
        if let sink = encoded.sinkData {
            decoded = concatenated([sink, decoded], axis: 2)
        }

        return decoded
    }

    // MARK: - Encode Values

    /// Compress float values to TurboQuant format (MSE only, no QJL).
    public static func encodeValues(
        _ values: MLXArray,
        state: EncoderState,
        sinkTokens: Int = defaultSinkTokens
    ) -> EncodedValues {
        let origShape = values.shape
        let seqLen = origShape[2]
        let dim = origShape[origShape.count - 1]

        // Extract sink tokens
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

        // Step 1: Normalize
        let vectorNorms = (compressValues * compressValues).sum(axis: -1, keepDims: true).sqrt()
        let valuesUnit = compressValues / (vectorNorms + 1e-8)

        // Step 2: Hadamard rotate
        let valuesRotated = TQHadamard.hadamardRotate(valuesUnit, signs: state.rotationSigns)

        // Step 3: Per-coordinate MSE quantization (b bits)
        let flatRotated = valuesRotated.asType(.float32).reshaped([-1, dim])
        let mseIndices = TQCodebook.quantizeScalar(flatRotated, codebook: state.valueCodebook)

        // Step 4: Pack
        let packedIndices = TQBitPack.packBits(mseIndices.reshaped(-1), bits: state.valueIndexBits)

        return EncodedValues(
            indicesPacked: packedIndices,
            vectorNorms: vectorNorms.asType(.float16),
            shape: compressShape,
            indexBits: state.valueIndexBits,
            seed: state.seed,
            sinkData: sinkData
        )
    }

    // MARK: - Decode Values

    /// Decompress values from TurboQuant format.
    public static func decodeValues(_ encoded: EncodedValues, state: EncoderState) -> MLXArray {
        let origShape = encoded.shape
        let dim = origShape[origShape.count - 1]
        let nElements = origShape.reduce(1, *)

        // Step 1: Unpack
        let flatIndices = TQBitPack.unpackBits(
            encoded.indicesPacked, bits: encoded.indexBits, nElements: nElements
        ).reshaped([-1, dim])
        let flatVecNorms = encoded.vectorNorms.asType(.float32).reshaped([-1, 1])

        // Step 2: Codebook lookup
        let mseDequant = TQCodebook.dequantizeScalar(flatIndices, codebook: state.valueCodebook)

        // Step 3: Inverse Hadamard
        let reconstructedRotated = mseDequant.reshaped(origShape)
        let reconstructedUnit = TQHadamard.hadamardInverse(
            reconstructedRotated, signs: state.rotationSigns)

        // Step 4: Scale by norms
        var decoded = (reconstructedUnit * flatVecNorms.reshaped(
            Array(origShape.dropLast()) + [1])).asType(.float16)

        // Prepend sink tokens
        if let sink = encoded.sinkData {
            decoded = concatenated([sink, decoded], axis: 2)
        }

        return decoded
    }

    // MARK: - Legacy API (backwards compatible)

    /// Encode keys using seed-based state creation (convenience).
    public static func encodeKeys(
        keys: MLXArray,
        bits: Int = 3,
        seed: Int = 42,
        sinkTokens: Int = defaultSinkTokens
    ) -> EncodedKeys {
        let dim = keys.dim(keys.ndim - 1)
        let state = EncoderState(dim: dim, keyBits: bits, seed: seed)
        return encodeKeys(keys, state: state, sinkTokens: sinkTokens)
    }

    /// Encode values using seed-based state creation (convenience).
    public static func encodeValues(
        values: MLXArray,
        bits: Int = 3,
        seed: Int = 42,
        sinkTokens: Int = defaultSinkTokens
    ) -> EncodedValues {
        let dim = values.dim(values.ndim - 1)
        let state = EncoderState(dim: dim, valueBits: bits, seed: seed)
        return encodeValues(values, state: state, sinkTokens: sinkTokens)
    }

    /// Decode keys using seed-based state creation (convenience).
    public static func decodeKeys(_ encoded: EncodedKeys, seed: Int = 42) -> MLXArray {
        let dim = encoded.shape.last ?? 128
        let keyBits = encoded.indexBits + 1  // indexBits = keyBits - 1
        let state = EncoderState(dim: dim, keyBits: keyBits, seed: encoded.seed)
        return decodeKeys(encoded, state: state)
    }

    /// Decode values using seed-based state creation (convenience).
    public static func decodeValues(_ encoded: EncodedValues, seed: Int = 42) -> MLXArray {
        let dim = encoded.shape.last ?? 128
        let state = EncoderState(dim: dim, valueBits: encoded.indexBits, seed: encoded.seed)
        return decodeValues(encoded, state: state)
    }
}

/// Helpers for preserving TurboQuant-compressed KV through cache store/fetch paths.
enum TurboQuantLayerCache {

    static func totalTokenCount(for encoded: EncodedKeys) -> Int {
        encoded.sinkCount + (encoded.shape.count > 2 ? encoded.shape[2] : 0)
    }

    static func encodeAttentionLayer(
        keys: MLXArray,
        values: MLXArray,
        config: TurboQuantConfig,
        layerIndex: Int,
        totalLayers: Int,
        sinkTokens: Int? = nil
    ) -> LayerCacheEntry? {
        guard let keyBits = config.keyBits(forLayer: layerIndex, totalLayers: totalLayers),
              let valueBits = config.valueBits(forLayer: layerIndex, totalLayers: totalLayers),
              keys.ndim == 4,
              values.ndim == 4 else {
            return nil
        }

        let keyDim = keys.dim(keys.ndim - 1)
        let valueDim = values.dim(values.ndim - 1)
        guard keyDim == valueDim else {
            // MLA-style asymmetric KV dimensions still need a dedicated path.
            return nil
        }

        let state = TurboQuantEncoder.EncoderState(
            dim: keyDim,
            keyBits: keyBits,
            valueBits: valueBits,
            seed: config.seed
        )
        let preservedSinkTokens = max(0, sinkTokens ?? TurboQuantEncoder.defaultSinkTokens)
        let encodedKeys = TurboQuantEncoder.encodeKeys(
            keys,
            state: state,
            sinkTokens: preservedSinkTokens
        )
        let encodedValues = TurboQuantEncoder.encodeValues(
            values,
            state: state,
            sinkTokens: preservedSinkTokens
        )
        return .compressedAttention(encodedKeys, encodedValues, keys.dim(2))
    }

    static func sliceCompressedAttention(
        _ encodedKeys: EncodedKeys,
        _ encodedValues: EncodedValues,
        range: Range<Int>
    ) -> LayerCacheEntry? {
        guard let slicedKeys = slice(encodedKeys, range: range),
              let slicedValues = slice(encodedValues, range: range) else {
            return nil
        }
        return .compressedAttention(slicedKeys, slicedValues, range.count)
    }

    static func mergeCompressedAttention(_ segments: [LayerCacheEntry]) -> LayerCacheEntry? {
        let encodedSegments = segments.compactMap { entry -> (EncodedKeys, EncodedValues, Int)? in
            guard case .compressedAttention(let encodedKeys, let encodedValues, let offset) = entry else {
                return nil
            }
            return (encodedKeys, encodedValues, offset)
        }

        guard encodedSegments.count == segments.count,
              let first = encodedSegments.first else {
            return nil
        }

        let firstKeys = first.0
        let firstValues = first.1
        guard firstKeys.shape.count == 4, firstValues.shape.count == 4 else {
            return nil
        }

        let batch = firstKeys.shape[0]
        let heads = firstKeys.shape[1]
        let dim = firstKeys.shape[3]
        let valueDim = firstValues.shape[3]

        var keyIndexVectors: [MLXArray] = []
        var keyQJLVectors: [MLXArray] = []
        var keyResiduals: [MLXArray] = []
        var keyNorms: [MLXArray] = []
        var valueIndexVectors: [MLXArray] = []
        var valueNorms: [MLXArray] = []
        var totalCompressedTokens = 0
        var totalOffset = 0

        for (segmentIndex, segment) in encodedSegments.enumerated() {
            let encodedKeys = segment.0
            let encodedValues = segment.1
            let offset = segment.2

            guard encodedKeys.shape.count == 4,
                  encodedValues.shape.count == 4,
                  encodedKeys.shape[0] == batch,
                  encodedKeys.shape[1] == heads,
                  encodedKeys.shape[3] == dim,
                  encodedValues.shape[0] == batch,
                  encodedValues.shape[1] == heads,
                  encodedValues.shape[3] == valueDim,
                  encodedKeys.indexBits == firstKeys.indexBits,
                  encodedValues.indexBits == firstValues.indexBits,
                  encodedKeys.seed == firstKeys.seed,
                  encodedValues.seed == firstValues.seed else {
                return nil
            }

            if segmentIndex > 0 && (encodedKeys.sinkCount > 0 || encodedValues.sinkCount > 0) {
                return nil
            }

            let compressedTokenCount = encodedKeys.shape[2]
            let keyElementCount = encodedKeys.shape.reduce(1, *)
            let valueElementCount = encodedValues.shape.reduce(1, *)

            if compressedTokenCount > 0 {
                keyIndexVectors.append(
                    TQBitPack.unpackBits(
                        encodedKeys.indicesPacked,
                        bits: encodedKeys.indexBits,
                        nElements: keyElementCount
                    )
                )
                keyQJLVectors.append(
                    TQBitPack.unpackSigns(
                        encodedKeys.qjlPacked,
                        nElements: keyElementCount
                    )
                )
                keyResiduals.append(encodedKeys.residualNorms)
                keyNorms.append(encodedKeys.vectorNorms)
                valueIndexVectors.append(
                    TQBitPack.unpackBits(
                        encodedValues.indicesPacked,
                        bits: encodedValues.indexBits,
                        nElements: valueElementCount
                    )
                )
                valueNorms.append(encodedValues.vectorNorms)
                totalCompressedTokens += compressedTokenCount
            }

            totalOffset += offset
        }

        guard totalCompressedTokens > 0 else {
            return nil
        }

        let mergedKeys = EncodedKeys(
            indicesPacked: TQBitPack.packBits(
                keyIndexVectors.count == 1 ? keyIndexVectors[0] : concatenated(keyIndexVectors, axis: 0),
                bits: firstKeys.indexBits
            ),
            qjlPacked: TQBitPack.packSigns(
                keyQJLVectors.count == 1 ? keyQJLVectors[0] : concatenated(keyQJLVectors, axis: 0)
            ),
            residualNorms: keyResiduals.count == 1 ? keyResiduals[0] : concatenated(keyResiduals, axis: 2),
            vectorNorms: keyNorms.count == 1 ? keyNorms[0] : concatenated(keyNorms, axis: 2),
            shape: [batch, heads, totalCompressedTokens, dim],
            indexBits: firstKeys.indexBits,
            seed: firstKeys.seed,
            sinkData: firstKeys.sinkData
        )
        let mergedValues = EncodedValues(
            indicesPacked: TQBitPack.packBits(
                valueIndexVectors.count == 1 ? valueIndexVectors[0] : concatenated(valueIndexVectors, axis: 0),
                bits: firstValues.indexBits
            ),
            vectorNorms: valueNorms.count == 1 ? valueNorms[0] : concatenated(valueNorms, axis: 2),
            shape: [batch, heads, totalCompressedTokens, valueDim],
            indexBits: firstValues.indexBits,
            seed: firstValues.seed,
            sinkData: firstValues.sinkData
        )

        return .compressedAttention(mergedKeys, mergedValues, totalOffset)
    }

    private static func slice(_ encoded: EncodedKeys, range: Range<Int>) -> EncodedKeys? {
        guard encoded.shape.count == 4 else {
            return nil
        }

        let totalTokens = totalTokenCount(for: encoded)
        guard !range.isEmpty,
              range.lowerBound >= 0,
              range.upperBound <= totalTokens else {
            return nil
        }

        let sinkStart = min(range.lowerBound, encoded.sinkCount)
        let sinkEnd = min(range.upperBound, encoded.sinkCount)
        let compressedStart = max(range.lowerBound - encoded.sinkCount, 0)
        let compressedEnd = max(range.upperBound - encoded.sinkCount, 0)
        let compressedCount = compressedEnd - compressedStart

        guard compressedCount > 0 else {
            return nil
        }

        let sinkSlice: MLXArray?
        if let sinkData = encoded.sinkData, sinkEnd > sinkStart {
            sinkSlice = sinkData[.ellipsis, sinkStart..<sinkEnd, 0...]
        } else {
            sinkSlice = nil
        }

        let keyElementCount = encoded.shape.reduce(1, *)
        let unpackedIndices = TQBitPack.unpackBits(
            encoded.indicesPacked,
            bits: encoded.indexBits,
            nElements: keyElementCount
        ).reshaped(encoded.shape)
        let unpackedQJL = TQBitPack.unpackSigns(
            encoded.qjlPacked,
            nElements: keyElementCount
        ).reshaped(encoded.shape)

        let slicedShape = [encoded.shape[0], encoded.shape[1], compressedCount, encoded.shape[3]]
        let slicedIndices = unpackedIndices[.ellipsis, compressedStart..<compressedEnd, 0...]
        let slicedQJL = unpackedQJL[.ellipsis, compressedStart..<compressedEnd, 0...]
        let slicedResiduals = encoded.residualNorms[.ellipsis, compressedStart..<compressedEnd, 0...]
        let slicedNorms = encoded.vectorNorms[.ellipsis, compressedStart..<compressedEnd, 0...]

        return EncodedKeys(
            indicesPacked: TQBitPack.packBits(slicedIndices.reshaped([-1]), bits: encoded.indexBits),
            qjlPacked: TQBitPack.packSigns(slicedQJL.reshaped([-1])),
            residualNorms: slicedResiduals,
            vectorNorms: slicedNorms,
            shape: slicedShape,
            indexBits: encoded.indexBits,
            seed: encoded.seed,
            sinkData: sinkSlice
        )
    }

    private static func slice(_ encoded: EncodedValues, range: Range<Int>) -> EncodedValues? {
        guard encoded.shape.count == 4 else {
            return nil
        }

        let totalTokens = encoded.sinkCount + encoded.shape[2]
        guard !range.isEmpty,
              range.lowerBound >= 0,
              range.upperBound <= totalTokens else {
            return nil
        }

        let sinkStart = min(range.lowerBound, encoded.sinkCount)
        let sinkEnd = min(range.upperBound, encoded.sinkCount)
        let compressedStart = max(range.lowerBound - encoded.sinkCount, 0)
        let compressedEnd = max(range.upperBound - encoded.sinkCount, 0)
        let compressedCount = compressedEnd - compressedStart

        guard compressedCount > 0 else {
            return nil
        }

        let sinkSlice: MLXArray?
        if let sinkData = encoded.sinkData, sinkEnd > sinkStart {
            sinkSlice = sinkData[.ellipsis, sinkStart..<sinkEnd, 0...]
        } else {
            sinkSlice = nil
        }

        let valueElementCount = encoded.shape.reduce(1, *)
        let unpackedIndices = TQBitPack.unpackBits(
            encoded.indicesPacked,
            bits: encoded.indexBits,
            nElements: valueElementCount
        ).reshaped(encoded.shape)

        let slicedShape = [encoded.shape[0], encoded.shape[1], compressedCount, encoded.shape[3]]
        let slicedIndices = unpackedIndices[.ellipsis, compressedStart..<compressedEnd, 0...]
        let slicedNorms = encoded.vectorNorms[.ellipsis, compressedStart..<compressedEnd, 0...]

        return EncodedValues(
            indicesPacked: TQBitPack.packBits(slicedIndices.reshaped([-1]), bits: encoded.indexBits),
            vectorNorms: slicedNorms,
            shape: slicedShape,
            indexBits: encoded.indexBits,
            seed: encoded.seed,
            sinkData: sinkSlice
        )
    }
}
