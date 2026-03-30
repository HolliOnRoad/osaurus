import Testing
import MLX
@testable import VMLXRuntime

@Suite("TurboQuantKVCache")
struct TurboQuantKVCacheTests {

    @Test("Initial state is fill phase")
    func initialState() {
        let config = TurboQuantConfig()
        let cache = TurboQuantKVCache(config: config, layerIndex: 5, totalLayers: 32)
        #expect(cache.phase == .fill)
        #expect(cache.isEmpty)
        #expect(cache.offset == 0)
    }

    @Test("Append float keys and values")
    func appendFloat() {
        let config = TurboQuantConfig()
        let cache = TurboQuantKVCache(config: config, layerIndex: 5, totalLayers: 32)

        let keys = MLXArray.zeros([1, 8, 10, 128])   // batch=1, heads=8, tokens=10, dim=128
        let values = MLXArray.zeros([1, 8, 10, 128])
        cache.appendFloat(keys: keys, values: values)

        #expect(cache.offset == 10)
        #expect(!cache.isEmpty)
        #expect(cache.phase == .fill)
    }

    @Test("Compress transitions to compressed phase")
    func compress() {
        let config = TurboQuantConfig()
        let cache = TurboQuantKVCache(config: config, layerIndex: 5, totalLayers: 32)

        let keys = MLXArray.zeros([1, 8, 10, 128])
        let values = MLXArray.zeros([1, 8, 10, 128])
        cache.appendFloat(keys: keys, values: values)

        cache.compress()
        #expect(cache.phase == .compressed)
    }

    @Test("Get keys/values returns data")
    func getKeysValues() {
        let config = TurboQuantConfig()
        let cache = TurboQuantKVCache(config: config, layerIndex: 5, totalLayers: 32)

        let keys = MLXArray.zeros([1, 8, 10, 128])
        let values = MLXArray.zeros([1, 8, 10, 128])
        cache.appendFloat(keys: keys, values: values)

        #expect(cache.getKeys() != nil)
        #expect(cache.getValues() != nil)
    }

    @Test("Layer-specific bit widths")
    func layerBitWidths() {
        let config = TurboQuantConfig(
            defaultKeyBits: 3,
            criticalLayers: [0, 1, -1],
            criticalKeyBits: 4
        )

        // Critical layer (first layer)
        let cache0 = TurboQuantKVCache(config: config, layerIndex: 0, totalLayers: 32)
        #expect(cache0.keyBits == 4)

        // Standard layer
        let cache5 = TurboQuantKVCache(config: config, layerIndex: 5, totalLayers: 32)
        #expect(cache5.keyBits == 3)

        // Critical layer (last layer)
        let cache31 = TurboQuantKVCache(config: config, layerIndex: 31, totalLayers: 32)
        #expect(cache31.keyBits == 4)
    }

    @Test("SSM layer gets nil bits")
    func ssmLayerNilBits() {
        let config = TurboQuantConfig(
            layerPattern: [.ssm, .attention, .ssm, .attention]
        )
        let cache = TurboQuantKVCache(config: config, layerIndex: 0, totalLayers: 4)
        #expect(cache.keyBits == nil)  // SSM layer

        let cache1 = TurboQuantKVCache(config: config, layerIndex: 1, totalLayers: 4)
        #expect(cache1.keyBits != nil)  // Attention layer
    }

    @Test("FromKVCacheLayer creates wrapper")
    func fromKVCacheLayer() {
        let layer = KVCacheLayer(
            keys: MLXArray.zeros([1, 8, 50, 128]),
            values: MLXArray.zeros([1, 8, 50, 128]),
            offset: 50
        )
        let config = TurboQuantConfig()
        let tq = TurboQuantKVCache.fromKVCacheLayer(layer, config: config, layerIndex: 3, totalLayers: 32)

        #expect(tq.offset == 50)
        #expect(tq.phase == .fill)
        #expect(tq.getKeys() != nil)
    }

    @Test("ToKVCacheLayer converts back")
    func toKVCacheLayer() {
        let config = TurboQuantConfig()
        let cache = TurboQuantKVCache(config: config, layerIndex: 5, totalLayers: 32)
        cache.appendFloat(
            keys: MLXArray.zeros([1, 8, 10, 128]),
            values: MLXArray.zeros([1, 8, 10, 128])
        )

        let layer = cache.toKVCacheLayer()
        #expect(layer != nil)
        #expect(layer?.offset == 10)
    }

    @Test("Estimated bytes non-zero with data")
    func estimatedBytes() {
        let config = TurboQuantConfig()
        let cache = TurboQuantKVCache(config: config, layerIndex: 5, totalLayers: 32)
        cache.appendFloat(
            keys: MLXArray.zeros([1, 8, 10, 128]),
            values: MLXArray.zeros([1, 8, 10, 128])
        )
        #expect(cache.estimatedBytes > 0)
    }
}
