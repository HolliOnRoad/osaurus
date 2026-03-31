import MLX

// MARK: - Layer Type

/// Describes the type of a model layer for pattern-based cache construction.
public enum LayerType: Sendable, Equatable {
    case attention
    case ssm
    case expert
}

// MARK: - Hybrid Cache

/// A complete multi-layer cache that may contain both attention (KV) and SSM layers.
/// This is the primary cache container for hybrid architectures like Jamba, Nemotron-H,
/// and Qwen3.5-A3B that interleave attention and SSM layers.
public struct HybridCache: @unchecked Sendable {
    public var layers: [LayerCacheEntry]

    public init(layers: [LayerCacheEntry]) {
        self.layers = layers
    }

    // MARK: - Introspection

    public var layerCount: Int { layers.count }

    /// True if the cache contains both attention and SSM layers.
    public var isHybrid: Bool {
        let hasAttention = layers.contains { $0.isAttention }
        let hasSSM = layers.contains { $0.isSSM }
        return hasAttention && hasSSM
    }

    /// True if every layer is attention (traditional transformer).
    public var isPureAttention: Bool {
        !layers.isEmpty && layers.allSatisfy { $0.isAttention }
    }

    /// True if every layer is SSM (pure Mamba-style).
    public var isPureSSM: Bool {
        !layers.isEmpty && layers.allSatisfy { $0.isSSM }
    }

    /// Whether the entire cache can be safely truncated.
    /// Only possible when there are no SSM layers (which are path-dependent).
    public var canTruncate: Bool {
        layers.allSatisfy { $0.canTruncate }
    }

    /// Indices of attention layers in the cache.
    public var attentionLayerIndices: [Int] {
        layers.indices.filter { layers[$0].isAttention }
    }

    /// Indices of SSM layers in the cache.
    public var ssmLayerIndices: [Int] {
        layers.indices.filter { layers[$0].isSSM }
    }

    /// Total estimated memory usage across all layers.
    public var estimatedBytes: Int {
        layers.reduce(0) { $0 + $1.estimatedBytes }
    }

    /// All attention layer entries (KVCacheLayer values).
    /// Compressed attention layers are NOT included — use decompressedAttentionLayers instead.
    public var attentionLayers: [KVCacheLayer] {
        layers.compactMap { entry in
            if case .attention(let kv) = entry { return kv }
            return nil
        }
    }

    /// All SSM layer entries (SSMStateLayer values).
    public var ssmLayers: [SSMStateLayer] {
        layers.compactMap { entry in
            if case .ssm(let ssm) = entry { return ssm }
            return nil
        }
    }

    // MARK: - Truncation

    /// Truncate all attention layers to keep only the first `tokenCount` tokens.
    /// Returns nil if any SSM layer is present (safety gate — SSM state is path-dependent
    /// and cannot be un-done to match a shorter prefix).
    public func truncated(to tokenCount: Int) -> HybridCache? {
        guard canTruncate else { return nil }

        let truncatedLayers = layers.compactMap { $0.truncated(to: tokenCount) }
        guard truncatedLayers.count == layers.count else { return nil }

        return HybridCache(layers: truncatedLayers)
    }

    // MARK: - Materialization

    /// Force evaluation of all lazy MLXArrays in the cache.
    /// Call this after building or modifying the cache to ensure all computations
    /// are materialized before passing to the next inference step.
    public func materialized() {
        for layer in layers {
            switch layer {
            case .attention(let kv):
                kv.keys.eval()
                kv.values.eval()
            case .ssm(let ssm):
                for s in ssm.state {
                    s.eval()
                }
            case .compressedAttention(let ek, let ev, _):
                // Eval compressed arrays so they're materialized for storage
                ek.indicesPacked.eval()
                ek.qjlPacked.eval()
                ek.residualNorms.eval()
                ek.vectorNorms.eval()
                ek.sinkData?.eval()
                ev.indicesPacked.eval()
                ev.vectorNorms.eval()
                ev.sinkData?.eval()
            }
        }
    }

    // MARK: - Factory

    /// Build a HybridCache from a layer type pattern.
    ///
    /// - Parameters:
    ///   - pattern: Array of `LayerType` values describing each layer's type.
    ///   - kvFactory: Closure that creates a `KVCacheLayer` for attention layers.
    ///   - ssmFactory: Closure that creates an `SSMStateLayer` for SSM layers.
    /// - Returns: A fully populated `HybridCache`.
    ///
    /// Expert layers are currently treated as attention layers (most expert-mixture
    /// architectures use attention within each expert).
    public static func fromPattern(
        _ pattern: [LayerType],
        kvFactory: () -> KVCacheLayer,
        ssmFactory: () -> SSMStateLayer
    ) -> HybridCache {
        let layers = pattern.map { layerType -> LayerCacheEntry in
            switch layerType {
            case .attention, .expert:
                return .attention(kvFactory())
            case .ssm:
                return .ssm(ssmFactory())
            }
        }
        return HybridCache(layers: layers)
    }
}

// MARK: - Pattern Parsing

/// Parse a hybrid architecture pattern string into an array of `LayerType`.
///
/// Pattern characters:
/// - `M` — SSM (Mamba) layer
/// - `*` — Attention layer
/// - `E` — Expert layer
///
/// Example: `"MMM*"` → `[.ssm, .ssm, .ssm, .attention]`
/// This matches patterns used in papers describing hybrid architectures
/// (e.g., Jamba's "every 4th layer is attention" pattern).
public func parseHybridPattern(_ pattern: String) -> [LayerType] {
    pattern.map { char in
        switch char {
        case "M", "m":
            return .ssm
        case "*":
            return .attention
        case "E", "e":
            return .expert
        default:
            return .attention  // default to attention for unknown characters
        }
    }
}
