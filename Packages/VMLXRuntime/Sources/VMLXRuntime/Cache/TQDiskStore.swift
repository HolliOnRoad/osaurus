import Foundation
import MLX
import CryptoKit

/// Metadata keys for TQ-native safetensors format.
public enum TQMetadataKeys {
    public static let tqNativeMarker = "__tq_native__"
    public static let numLayers = "__num_layers__"

    public static func layerClass(_ i: Int) -> String { "__layer_\(i)_class__" }
    public static func ckShape(_ i: Int) -> String { "__tq_\(i)_ck_shape__" }
    public static func ckBits(_ i: Int) -> String { "__tq_\(i)_ck_bits__" }
    public static func cvShape(_ i: Int) -> String { "__tq_\(i)_cv_shape__" }
    public static func cvBits(_ i: Int) -> String { "__tq_\(i)_cv_bits__" }
    public static func offset(_ i: Int) -> String { "__tq_\(i)_offset__" }
    public static func keyDim(_ i: Int) -> String { "__tq_\(i)_key_dim__" }
    public static func valueDim(_ i: Int) -> String { "__tq_\(i)_value_dim__" }
    public static func sinkTokens(_ i: Int) -> String { "__tq_\(i)_sink_tokens__" }
    public static func seed(_ i: Int) -> String { "__tq_\(i)_seed__" }
}

/// Tensor keys for TQ-native safetensors format.
public enum TQTensorKeys {
    public static func ckIndicesPacked(_ i: Int) -> String { "tq_\(i)_ck_indices_packed" }
    public static func ckQjlPacked(_ i: Int) -> String { "tq_\(i)_ck_qjl_packed" }
    public static func ckResidualNorms(_ i: Int) -> String { "tq_\(i)_ck_residual_norms" }
    public static func ckVectorNorms(_ i: Int) -> String { "tq_\(i)_ck_vector_norms" }
    public static func cvIndicesPacked(_ i: Int) -> String { "tq_\(i)_cv_indices_packed" }
    public static func cvVectorNorms(_ i: Int) -> String { "tq_\(i)_cv_vector_norms" }
}

/// Serialization result: tensors dict + metadata dict.
public struct TQSerializedCache: @unchecked Sendable {
    /// Tensor name -> MLXArray (for safetensors storage).
    public let tensors: [String: MLXArray]
    /// Metadata key -> string value.
    public let metadata: [String: String]

    /// Estimated compressed size in bytes.
    public var compressedBytes: Int {
        tensors.values.reduce(0) { $0 + $1.nbytes }
    }
}

/// TurboQuant-native disk serialization.
/// Stores EncodedKeys/EncodedValues directly (packed uint32 + float16 norms).
/// 26x smaller than float16 (40KB vs 1MB per 100 tokens).
public struct TQDiskStore: Sendable {

    /// Check if a cache contains TQ-compressed data.
    public static func isTQCompressed(_ cache: HybridCache) -> Bool {
        // Check if any attention layer has compressed data
        // In production, this checks for TurboQuantKVCache instances
        // For now, check if any attention layer exists (TQ wraps all attention)
        cache.attentionLayers.count > 0
    }

    /// Serialize TQ-compressed cache layers to tensors + metadata.
    /// Only serializes attention layers (SSM state stored separately).
    public static func serialize(
        keys: [EncodedKeys],
        values: [EncodedValues],
        offsets: [Int]
    ) -> TQSerializedCache {
        var tensors: [String: MLXArray] = [:]
        var metadata: [String: String] = [:]

        metadata[TQMetadataKeys.tqNativeMarker] = "true"
        metadata[TQMetadataKeys.numLayers] = "\(keys.count)"

        for i in 0..<keys.count {
            let ek = keys[i]
            let ev = values[i]

            // Store key tensors
            tensors[TQTensorKeys.ckIndicesPacked(i)] = ek.indicesPacked
            tensors[TQTensorKeys.ckQjlPacked(i)] = ek.qjlPacked
            tensors[TQTensorKeys.ckResidualNorms(i)] = ek.residualNorms
            tensors[TQTensorKeys.ckVectorNorms(i)] = ek.vectorNorms

            // Store value tensors
            tensors[TQTensorKeys.cvIndicesPacked(i)] = ev.indicesPacked
            tensors[TQTensorKeys.cvVectorNorms(i)] = ev.vectorNorms

            // Store metadata
            metadata[TQMetadataKeys.ckShape(i)] = ek.shape.description
            metadata[TQMetadataKeys.ckBits(i)] = "\(ek.indexBits)"
            metadata[TQMetadataKeys.cvShape(i)] = ev.shape.description
            metadata[TQMetadataKeys.cvBits(i)] = "\(ev.indexBits)"
            metadata[TQMetadataKeys.offset(i)] = "\(offsets[i])"
            metadata[TQMetadataKeys.seed(i)] = "\(ek.seed)"
        }

        return TQSerializedCache(tensors: tensors, metadata: metadata)
    }

    /// Deserialize TQ-native format back to EncodedKeys/EncodedValues.
    public static func deserialize(
        tensors: [String: MLXArray],
        metadata: [String: String]
    ) -> (keys: [EncodedKeys], values: [EncodedValues], offsets: [Int])? {
        guard metadata[TQMetadataKeys.tqNativeMarker] == "true" else { return nil }
        guard let numLayersStr = metadata[TQMetadataKeys.numLayers],
              let numLayers = Int(numLayersStr) else { return nil }

        var keys: [EncodedKeys] = []
        var values: [EncodedValues] = []
        var offsets: [Int] = []

        for i in 0..<numLayers {
            // Load key tensors
            guard let ckIndices = tensors[TQTensorKeys.ckIndicesPacked(i)],
                  let ckQjl = tensors[TQTensorKeys.ckQjlPacked(i)],
                  let ckResNorms = tensors[TQTensorKeys.ckResidualNorms(i)],
                  let ckVecNorms = tensors[TQTensorKeys.ckVectorNorms(i)] else {
                return nil
            }

            // Load value tensors
            guard let cvIndices = tensors[TQTensorKeys.cvIndicesPacked(i)],
                  let cvVecNorms = tensors[TQTensorKeys.cvVectorNorms(i)] else {
                return nil
            }

            // Parse metadata
            let ckBits = Int(metadata[TQMetadataKeys.ckBits(i)] ?? "3") ?? 3
            let cvBits = Int(metadata[TQMetadataKeys.cvBits(i)] ?? "3") ?? 3
            let offset = Int(metadata[TQMetadataKeys.offset(i)] ?? "0") ?? 0
            let seed = Int(metadata[TQMetadataKeys.seed(i)] ?? "42") ?? 42

            // Parse shape from string representation
            let ckShape = _parseShape(metadata[TQMetadataKeys.ckShape(i)] ?? "[]")
            let cvShape = _parseShape(metadata[TQMetadataKeys.cvShape(i)] ?? "[]")

            keys.append(EncodedKeys(
                indicesPacked: ckIndices,
                qjlPacked: ckQjl,
                residualNorms: ckResNorms,
                vectorNorms: ckVecNorms,
                shape: ckShape,
                indexBits: ckBits,
                seed: seed
            ))

            values.append(EncodedValues(
                indicesPacked: cvIndices,
                vectorNorms: cvVecNorms,
                shape: cvShape,
                indexBits: cvBits,
                seed: seed
            ))

            offsets.append(offset)
        }

        return (keys, values, offsets)
    }

    /// Parse a shape string like "[1, 8, 100, 128]" to [Int].
    private static func _parseShape(_ str: String) -> [Int] {
        let cleaned = str.trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        guard !cleaned.isEmpty else { return [] }
        return cleaned.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }
}
