import Testing
import MLX
@testable import VMLXRuntime

@Suite("TurboQuantLayerCache")
struct TurboQuantLayerCacheTests {

    private func makeKV(tokenCount: Int) -> KVCacheLayer {
        KVCacheLayer(
            keys: MLXArray.ones([1, 2, tokenCount, 8]),
            values: MLXArray.ones([1, 2, tokenCount, 8]) * MLXArray(Float(2.0)),
            offset: tokenCount
        )
    }

    @Test("slice and merge preserve compressed token coverage")
    func sliceAndMergeCompressedAttention() {
        let kv = makeKV(tokenCount: 12)
        let config = TurboQuantConfig(defaultKeyBits: 3, defaultValueBits: 3, seed: 42)

        guard let fullEntry = TurboQuantLayerCache.encodeAttentionLayer(
            keys: kv.keys,
            values: kv.values,
            config: config,
            layerIndex: 0,
            totalLayers: 1
        ) else {
            Issue.record("Expected TurboQuant encoding to succeed")
            return
        }

        guard case .compressedAttention(let fullKeys, let fullValues, let fullOffset) = fullEntry else {
            Issue.record("Expected compressed attention entry")
            return
        }
        #expect(fullOffset == 12)

        let firstSlice = TurboQuantLayerCache.sliceCompressedAttention(
            fullKeys,
            fullValues,
            range: 0..<6
        )
        let secondSlice = TurboQuantLayerCache.sliceCompressedAttention(
            fullKeys,
            fullValues,
            range: 6..<12
        )

        guard let firstSlice,
              let secondSlice,
              let mergedEntry = TurboQuantLayerCache.mergeCompressedAttention([firstSlice, secondSlice]),
              case .compressedAttention(let mergedKeys, let mergedValues, let mergedOffset) = mergedEntry else {
            Issue.record("Expected slice + merge to preserve compressed entries")
            return
        }

        #expect(mergedOffset == 12)

        let state = TurboQuantEncoder.EncoderState(
            dim: fullKeys.shape.last ?? 8,
            keyBits: fullKeys.indexBits + 1,
            valueBits: fullValues.indexBits,
            seed: fullKeys.seed
        )
        let originalDecodedKeys = TurboQuantEncoder.decodeKeys(fullKeys, state: state)
        let mergedDecodedKeys = TurboQuantEncoder.decodeKeys(mergedKeys, state: state)
        let originalDecodedValues = TurboQuantEncoder.decodeValues(fullValues, state: state)
        let mergedDecodedValues = TurboQuantEncoder.decodeValues(mergedValues, state: state)

        #expect(mergedDecodedKeys.shape == originalDecodedKeys.shape)
        #expect(mergedDecodedValues.shape == originalDecodedValues.shape)
        #expect(abs(mergedDecodedKeys.sum().item(Float.self) - originalDecodedKeys.sum().item(Float.self)) < 0.001)
        #expect(abs(mergedDecodedValues.sum().item(Float.self) - originalDecodedValues.sum().item(Float.self)) < 0.001)
    }

    @Test("paged cache fetch preserves compressed entries when slices include compressed tail")
    func pagedCachePreservesCompressedEntries() {
        let config = CacheCoordinatorConfig(
            usePagedCache: true,
            useMemoryAwareCache: false,
            enableDiskCache: false,
            pagedBlockSize: 6,
            maxCacheBlocks: 32
        )
        let coordinator = CacheCoordinator(config: config)
        let kv = makeKV(tokenCount: 10)
        let tqConfig = TurboQuantConfig(defaultKeyBits: 3, defaultValueBits: 3, seed: 42)

        guard let compressedEntry = TurboQuantLayerCache.encodeAttentionLayer(
            keys: kv.keys,
            values: kv.values,
            config: tqConfig,
            layerIndex: 0,
            totalLayers: 1
        ) else {
            Issue.record("Expected TurboQuant encoding to succeed")
            return
        }

        let cache = HybridCache(layers: [compressedEntry])
        coordinator.store(tokens: Array(1...10), cache: cache)

        let result = coordinator.fetch(tokens: Array(1...10))
        guard case .hit(let cached, let remaining, let detail, _) = result else {
            Issue.record("Expected paged cache hit")
            return
        }

        #expect(detail == .paged)
        #expect(remaining.isEmpty)
        #expect(cached.layerCount == 1)
        if case .compressedAttention = cached.layers[0] {
            // expected
        } else {
            Issue.record("Expected paged cache to preserve compressed attention entry")
        }
    }

    @Test("Nemotron H and Cascade patterns only encode attention KV layers")
    func nemotronCascadePatternSkipsNonKVLayers() {
        let kv = makeKV(tokenCount: 8)
        let config = TurboQuantConfig(
            defaultKeyBits: 3,
            defaultValueBits: 3,
            layerPattern: parseHybridPattern("MEMEM*")
        )

        #expect(
            TurboQuantLayerCache.encodeAttentionLayer(
                keys: kv.keys,
                values: kv.values,
                config: config,
                layerIndex: 0,
                totalLayers: 6
            ) == nil
        )
        #expect(
            TurboQuantLayerCache.encodeAttentionLayer(
                keys: kv.keys,
                values: kv.values,
                config: config,
                layerIndex: 1,
                totalLayers: 6
            ) == nil
        )
        #expect(
            TurboQuantLayerCache.encodeAttentionLayer(
                keys: kv.keys,
                values: kv.values,
                config: config,
                layerIndex: 5,
                totalLayers: 6
            ) != nil
        )
    }

    @Test("Qwen 3.5 pattern only encodes full-attention layers")
    func qwenPatternSkipsLinearAttentionLayers() {
        let kv = makeKV(tokenCount: 8)
        let config = TurboQuantConfig(
            defaultKeyBits: 3,
            defaultValueBits: 3,
            layerPattern: [.ssm, .ssm, .ssm, .attention, .ssm, .ssm, .ssm, .attention]
        )

        #expect(
            TurboQuantLayerCache.encodeAttentionLayer(
                keys: kv.keys,
                values: kv.values,
                config: config,
                layerIndex: 0,
                totalLayers: 8
            ) == nil
        )
        #expect(
            TurboQuantLayerCache.encodeAttentionLayer(
                keys: kv.keys,
                values: kv.values,
                config: config,
                layerIndex: 3,
                totalLayers: 8
            ) != nil
        )
        #expect(
            TurboQuantLayerCache.encodeAttentionLayer(
                keys: kv.keys,
                values: kv.values,
                config: config,
                layerIndex: 4,
                totalLayers: 8
            ) == nil
        )
    }
}
