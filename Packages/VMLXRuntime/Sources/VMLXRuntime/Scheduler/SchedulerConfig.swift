import Foundation

/// Scheduling policy for request ordering.
public enum SchedulingPolicy: String, Sendable {
    case fcfs = "fcfs"       // First-come, first-served
    case priority = "priority"  // Priority-based
}

/// Configuration for the continuous batching scheduler.
/// Controls all knobs: batch sizes, cache behavior, KV quantization, disk paths.
public struct SchedulerConfig: Sendable {

    // MARK: - Batch Sizing

    /// Maximum concurrent sequences in a batch.
    public var maxNumSeqs: Int

    /// Maximum total tokens across all sequences in a batch.
    public var maxNumBatchedTokens: Int

    /// Scheduling policy.
    public var policy: SchedulingPolicy

    /// Number of requests to prefill simultaneously.
    public var prefillBatchSize: Int

    /// Number of requests to decode simultaneously.
    public var completionBatchSize: Int

    /// Tokens per prefill step (chunked prefill for long prompts).
    public var prefillStepSize: Int

    // MARK: - Prefix Cache

    /// Enable any prefix caching.
    public var enablePrefixCache: Bool

    /// Legacy max entries for simple prefix cache.
    public var prefixCacheSize: Int

    // MARK: - Memory-Aware Cache

    /// Use memory-aware prefix cache (recommended over simple prefix cache).
    public var useMemoryAwareCache: Bool

    /// Explicit memory limit in MB (nil = auto-detect from system RAM).
    public var cacheMemoryMB: Int?

    /// Fraction of available RAM for cache (default 0.30).
    public var cacheMemoryPercent: Float

    /// Cache entry TTL in minutes (0 = disabled).
    public var cacheTTLMinutes: Float

    // MARK: - Paged Cache

    /// Use paged (block-based) cache instead of simple prefix cache.
    public var usePagedCache: Bool

    /// Tokens per cache block.
    public var pagedCacheBlockSize: Int

    /// Maximum number of cache blocks in pool.
    public var maxCacheBlocks: Int

    // MARK: - KV Quantization

    /// KV cache quantization level: "none", "q4", "q8".
    public var kvCacheQuantization: String

    /// Group size for KV quantization.
    public var kvCacheGroupSize: Int

    // MARK: - Disk Cache (L2)

    /// Enable L2 disk cache for safetensors persistence.
    public var enableDiskCache: Bool

    /// Directory for disk cache files.
    public var diskCacheDir: URL?

    /// Maximum disk cache size in GB.
    public var diskCacheMaxGB: Float

    // MARK: - Block Disk Cache (L2)

    /// Enable block-level disk cache.
    public var enableBlockDiskCache: Bool

    /// Directory for block disk cache.
    public var blockDiskCacheDir: URL?

    /// Maximum block disk cache size in GB.
    public var blockDiskCacheMaxGB: Float

    // MARK: - Model Path

    /// Path to the loaded model (for disk cache scoping).
    public var modelPath: String?

    // MARK: - TurboQuant

    /// Enable TurboQuant 3-bit KV cache compression.
    public var enableTurboQuant: Bool

    // MARK: - Init

    public init(
        maxNumSeqs: Int = 256,
        maxNumBatchedTokens: Int = 8192,
        policy: SchedulingPolicy = .fcfs,
        prefillBatchSize: Int = 8,
        completionBatchSize: Int = 32,
        prefillStepSize: Int = 2048,
        enablePrefixCache: Bool = true,
        prefixCacheSize: Int = 100,
        useMemoryAwareCache: Bool = true,
        cacheMemoryMB: Int? = nil,
        cacheMemoryPercent: Float = 0.30,
        cacheTTLMinutes: Float = 0,
        usePagedCache: Bool = false,
        pagedCacheBlockSize: Int = 64,
        maxCacheBlocks: Int = 1000,
        kvCacheQuantization: String = "none",
        kvCacheGroupSize: Int = 64,
        enableDiskCache: Bool = false,
        diskCacheDir: URL? = nil,
        diskCacheMaxGB: Float = 10.0,
        enableBlockDiskCache: Bool = false,
        blockDiskCacheDir: URL? = nil,
        blockDiskCacheMaxGB: Float = 10.0,
        modelPath: String? = nil,
        enableTurboQuant: Bool = false
    ) {
        self.maxNumSeqs = maxNumSeqs
        self.maxNumBatchedTokens = maxNumBatchedTokens
        self.policy = policy
        self.prefillBatchSize = prefillBatchSize
        self.completionBatchSize = completionBatchSize
        self.prefillStepSize = prefillStepSize
        self.enablePrefixCache = enablePrefixCache
        self.prefixCacheSize = prefixCacheSize
        self.useMemoryAwareCache = useMemoryAwareCache
        self.cacheMemoryMB = cacheMemoryMB
        self.cacheMemoryPercent = cacheMemoryPercent
        self.cacheTTLMinutes = cacheTTLMinutes
        self.usePagedCache = usePagedCache
        self.pagedCacheBlockSize = pagedCacheBlockSize
        self.maxCacheBlocks = maxCacheBlocks
        self.kvCacheQuantization = kvCacheQuantization
        self.kvCacheGroupSize = kvCacheGroupSize
        self.enableDiskCache = enableDiskCache
        self.diskCacheDir = diskCacheDir
        self.diskCacheMaxGB = diskCacheMaxGB
        self.enableBlockDiskCache = enableBlockDiskCache
        self.blockDiskCacheDir = blockDiskCacheDir
        self.blockDiskCacheMaxGB = blockDiskCacheMaxGB
        self.modelPath = modelPath
        self.enableTurboQuant = enableTurboQuant
    }

    /// Build a CacheCoordinatorConfig from this scheduler config.
    public func toCacheCoordinatorConfig() -> CacheCoordinatorConfig {
        CacheCoordinatorConfig(
            enablePrefixCache: enablePrefixCache,
            usePagedCache: usePagedCache,
            useMemoryAwareCache: useMemoryAwareCache,
            enableDiskCache: enableDiskCache,
            pagedBlockSize: pagedCacheBlockSize,
            maxCacheBlocks: maxCacheBlocks,
            cacheMemoryPercent: cacheMemoryPercent,
            diskCacheMaxGB: diskCacheMaxGB,
            diskCacheDir: diskCacheDir
        )
    }

    /// Auto-configure based on system RAM.
    public static func autoDetect() -> SchedulerConfig {
        let ramGB = Float(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)

        var config = SchedulerConfig()

        // Scale prefill step by RAM
        if ramGB <= 24 {
            config.prefillStepSize = 1024
        } else if ramGB <= 64 {
            config.prefillStepSize = 2048
        } else {
            config.prefillStepSize = 4096
        }

        // Scale max concurrent by RAM
        if ramGB <= 24 {
            config.maxNumSeqs = 8
            config.completionBatchSize = 8
        } else if ramGB <= 48 {
            config.maxNumSeqs = 16
            config.completionBatchSize = 16
        } else {
            config.maxNumSeqs = 32
            config.completionBatchSize = 32
        }

        return config
    }
}
