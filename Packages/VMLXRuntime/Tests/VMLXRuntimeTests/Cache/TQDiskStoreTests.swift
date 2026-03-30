import Testing
import Foundation
import MLX
@testable import VMLXRuntime

@Suite("TQDiskStore")
struct TQDiskStoreTests {

    @Test("Metadata key generation")
    func metadataKeys() {
        #expect(TQMetadataKeys.ckShape(0) == "__tq_0_ck_shape__")
        #expect(TQMetadataKeys.ckBits(5) == "__tq_5_ck_bits__")
        #expect(TQMetadataKeys.offset(3) == "__tq_3_offset__")
        #expect(TQMetadataKeys.tqNativeMarker == "__tq_native__")
    }

    @Test("Tensor key generation")
    func tensorKeys() {
        #expect(TQTensorKeys.ckIndicesPacked(0) == "tq_0_ck_indices_packed")
        #expect(TQTensorKeys.cvVectorNorms(2) == "tq_2_cv_vector_norms")
    }

    @Test("Serialize and deserialize roundtrip")
    func serializeDeserialize() {
        let keys = [EncodedKeys(
            indicesPacked: MLXArray.zeros([100], dtype: .uint32),
            qjlPacked: MLXArray.zeros([50], dtype: .uint32),
            residualNorms: MLXArray.zeros([200]),
            vectorNorms: MLXArray.zeros([200]),
            shape: [1, 8, 100, 128],
            indexBits: 3
        )]
        let values = [EncodedValues(
            indicesPacked: MLXArray.zeros([100], dtype: .uint32),
            vectorNorms: MLXArray.zeros([200]),
            shape: [1, 8, 100, 128],
            indexBits: 3
        )]

        let serialized = TQDiskStore.serialize(keys: keys, values: values, offsets: [100])
        #expect(serialized.metadata[TQMetadataKeys.tqNativeMarker] == "true")
        #expect(serialized.metadata[TQMetadataKeys.numLayers] == "1")
        #expect(serialized.tensors.count == 6)  // 4 key + 2 value tensors

        let deserialized = TQDiskStore.deserialize(
            tensors: serialized.tensors,
            metadata: serialized.metadata
        )
        #expect(deserialized != nil)
        #expect(deserialized?.keys.count == 1)
        #expect(deserialized?.values.count == 1)
        #expect(deserialized?.offsets == [100])
        #expect(deserialized?.keys[0].indexBits == 3)
    }

    @Test("Deserialize non-TQ returns nil")
    func deserializeNonTQ() {
        let result = TQDiskStore.deserialize(tensors: [:], metadata: [:])
        #expect(result == nil)
    }

    @Test("Compressed bytes estimate")
    func compressedBytes() {
        let keys = [EncodedKeys(
            indicesPacked: MLXArray.zeros([100], dtype: .uint32),
            qjlPacked: MLXArray.zeros([50], dtype: .uint32),
            residualNorms: MLXArray.zeros([200]),
            vectorNorms: MLXArray.zeros([200]),
            shape: [1, 8, 100, 128],
            indexBits: 3
        )]
        let values = [EncodedValues(
            indicesPacked: MLXArray.zeros([100], dtype: .uint32),
            vectorNorms: MLXArray.zeros([200]),
            shape: [1, 8, 100, 128],
            indexBits: 3
        )]

        let serialized = TQDiskStore.serialize(keys: keys, values: values, offsets: [100])
        #expect(serialized.compressedBytes > 0)
    }

    @Test("TQ encoder stub returns valid structure")
    func encoderStub() {
        let keys = MLXArray.zeros([1, 8, 100, 128])
        let encoded = TurboQuantEncoder.encodeKeys(keys: keys, bits: 3)
        #expect(encoded.indexBits == 3)
        #expect(encoded.shape == [1, 8, 100, 128])
    }

    @Test("TQ decoder stub returns correct shape")
    func decoderStub() {
        let encoded = EncodedKeys(
            indicesPacked: MLXArray.zeros([1], dtype: .uint32),
            qjlPacked: MLXArray.zeros([1], dtype: .uint32),
            residualNorms: MLXArray.zeros([1]),
            vectorNorms: MLXArray.zeros([1]),
            shape: [1, 8, 50, 128],
            indexBits: 3
        )
        let decoded = TurboQuantEncoder.decodeKeys(encoded)
        #expect(decoded.shape == [1, 8, 50, 128])
    }
}
