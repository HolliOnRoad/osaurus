//
//  ServerConfiguration.swift
//  osaurus
//
//  Created by Terence on 8/17/25.
//

import Foundation

/// Appearance mode setting for the app
public enum AppearanceMode: String, Codable, CaseIterable, Sendable {
    case system = "system"
    case light = "light"
    case dark = "dark"

    public var displayName: String {
        switch self {
        case .system: return "System"
        case .light: return "Light"
        case .dark: return "Dark"
        }
    }
}

/// Configuration settings for the server
public struct ServerConfiguration: Codable, Equatable, Sendable {
    /// Server port (1-65535) — the gateway port for SwiftNIO
    public var port: Int

    /// Expose the server to the local network (0.0.0.0) or keep it on localhost (127.0.0.1)
    public var exposeToNetwork: Bool

    /// Start Osaurus automatically at login
    public var startAtLogin: Bool

    /// Hide the dock icon (run as accessory app)
    public var hideDockIcon: Bool

    /// Appearance mode (system, light, or dark)
    public var appearanceMode: AppearanceMode

    /// Number of threads for the event loop group
    public let numberOfThreads: Int

    /// Server backlog size
    public let backlog: Int32

    /// Default top-p sampling for generation (can be overridden per request)
    public var genTopP: Float

    /// Default max tokens for generation
    public var maxTokens: Int

    /// List of allowed origins for CORS. Empty disables CORS. Use "*" to allow any origin.
    public var allowedOrigins: [String]

    /// Memory management policy for loaded models
    public var modelEvictionPolicy: ModelEvictionPolicy

    // MARK: - vmlx Engine Settings

    /// Enable continuous batching (required for prefix cache, paged cache, multi-user)
    public var continuousBatching: Bool
    /// Maximum concurrent sequences in batched mode
    public var maxNumSeqs: Int
    /// Token stream interval (1 = every token)
    public var streamInterval: Int

    /// Enable prefix KV cache (reuse system prompt across turns)
    public var enablePrefixCache: Bool
    /// Max prefix cache entries (legacy count mode)
    public var prefixCacheSize: Int
    /// Fraction of RAM for prefix cache (0.0-1.0)
    public var cacheMemoryPercent: Float
    /// Fixed MB budget for cache (nil = auto from percent)
    public var cacheMemoryMB: Int?
    /// Cache entry TTL in minutes (0 = no expiry)
    public var cacheTTLMinutes: Float

    /// Use paged (block-based) KV cache
    public var usePagedCache: Bool
    /// Tokens per cache block
    public var pagedCacheBlockSize: Int
    /// Max cache blocks in memory
    public var maxCacheBlocks: Int

    /// Enable L2 disk cache for prompt KV states
    public var enableDiskCache: Bool
    /// Max disk cache size in GB
    public var diskCacheMaxGB: Float
    /// Enable block-level disk cache (L2 for paged cache)
    public var enableBlockDiskCache: Bool
    /// Max block disk cache size in GB
    public var blockDiskCacheMaxGB: Float

    /// KV cache quantization: "none", "q4", "q8"
    public var kvCacheQuantization: String
    /// KV quantization group size
    public var kvCacheGroupSize: Int

    /// Tool call parser: "auto", "none", "qwen", "llama", "mistral", "hermes", "deepseek", etc.
    public var toolCallParser: String
    /// Reasoning parser: "auto", "none", "qwen3", "deepseek_r1", "think", "mistral", "gemma4", etc.
    public var reasoningParser: String

    /// Enable JIT compilation (mx.compile on forward pass)
    public var enableJIT: Bool

    /// Idle sleep mode: "none", "soft", "deep" (derived from enableSoftSleep/enableDeepSleep)
    public var idleSleepMode: String
    /// Minutes of inactivity before sleep triggers
    public var idleSleepMinutes: Int
    /// Enable soft sleep (clear caches after timeout)
    public var enableSoftSleep: Bool
    /// Minutes before soft sleep triggers
    public var softSleepMinutes: Int
    /// Enable deep sleep (unload model after timeout)
    public var enableDeepSleep: Bool
    /// Minutes before deep sleep triggers
    public var deepSleepMinutes: Int

    /// Default enable_thinking for reasoning models: "true", "false", or nil (auto)
    public var defaultEnableThinking: String?
    /// Default temperature (nil = engine default 0.7)
    public var defaultTemperature: Float?
    /// Default top-p (nil = model default)
    public var defaultTopP: Float?

    /// Draft model path for speculative decoding
    public var speculativeModel: String?
    /// Draft tokens per speculative step
    public var numDraftTokens: Int
    /// Enable Prompt Lookup Decoding
    public var enablePLD: Bool

    private enum CodingKeys: String, CodingKey {
        case port, exposeToNetwork, startAtLogin, hideDockIcon, appearanceMode
        case numberOfThreads, backlog, genTopP, maxTokens, allowedOrigins, modelEvictionPolicy
        case continuousBatching, maxNumSeqs, streamInterval
        case enablePrefixCache, prefixCacheSize, cacheMemoryPercent, cacheMemoryMB, cacheTTLMinutes
        case usePagedCache, pagedCacheBlockSize, maxCacheBlocks
        case enableDiskCache, diskCacheMaxGB, enableBlockDiskCache, blockDiskCacheMaxGB
        case kvCacheQuantization, kvCacheGroupSize
        case toolCallParser, reasoningParser
        case enableJIT, idleSleepMode, idleSleepMinutes
        case enableSoftSleep, softSleepMinutes, enableDeepSleep, deepSleepMinutes
        case defaultEnableThinking, defaultTemperature, defaultTopP
        case speculativeModel, numDraftTokens, enablePLD
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let d = ServerConfiguration.default
        self.port = try container.decodeIfPresent(Int.self, forKey: .port) ?? d.port
        self.exposeToNetwork = try container.decodeIfPresent(Bool.self, forKey: .exposeToNetwork) ?? d.exposeToNetwork
        self.startAtLogin = try container.decodeIfPresent(Bool.self, forKey: .startAtLogin) ?? d.startAtLogin
        self.hideDockIcon = try container.decodeIfPresent(Bool.self, forKey: .hideDockIcon) ?? d.hideDockIcon
        self.appearanceMode = try container.decodeIfPresent(AppearanceMode.self, forKey: .appearanceMode) ?? d.appearanceMode
        self.numberOfThreads = try container.decodeIfPresent(Int.self, forKey: .numberOfThreads) ?? d.numberOfThreads
        self.backlog = try container.decodeIfPresent(Int32.self, forKey: .backlog) ?? d.backlog
        self.genTopP = try container.decodeIfPresent(Float.self, forKey: .genTopP) ?? d.genTopP
        self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? d.maxTokens
        self.allowedOrigins = try container.decodeIfPresent([String].self, forKey: .allowedOrigins) ?? d.allowedOrigins
        self.modelEvictionPolicy = try container.decodeIfPresent(ModelEvictionPolicy.self, forKey: .modelEvictionPolicy) ?? d.modelEvictionPolicy
        // vmlx engine
        self.continuousBatching = try container.decodeIfPresent(Bool.self, forKey: .continuousBatching) ?? d.continuousBatching
        self.maxNumSeqs = try container.decodeIfPresent(Int.self, forKey: .maxNumSeqs) ?? d.maxNumSeqs
        self.streamInterval = try container.decodeIfPresent(Int.self, forKey: .streamInterval) ?? d.streamInterval
        self.enablePrefixCache = try container.decodeIfPresent(Bool.self, forKey: .enablePrefixCache) ?? d.enablePrefixCache
        self.prefixCacheSize = try container.decodeIfPresent(Int.self, forKey: .prefixCacheSize) ?? d.prefixCacheSize
        self.cacheMemoryPercent = try container.decodeIfPresent(Float.self, forKey: .cacheMemoryPercent) ?? d.cacheMemoryPercent
        self.cacheMemoryMB = try container.decodeIfPresent(Int.self, forKey: .cacheMemoryMB)
        self.cacheTTLMinutes = try container.decodeIfPresent(Float.self, forKey: .cacheTTLMinutes) ?? d.cacheTTLMinutes
        self.usePagedCache = try container.decodeIfPresent(Bool.self, forKey: .usePagedCache) ?? d.usePagedCache
        self.pagedCacheBlockSize = try container.decodeIfPresent(Int.self, forKey: .pagedCacheBlockSize) ?? d.pagedCacheBlockSize
        self.maxCacheBlocks = try container.decodeIfPresent(Int.self, forKey: .maxCacheBlocks) ?? d.maxCacheBlocks
        self.enableDiskCache = try container.decodeIfPresent(Bool.self, forKey: .enableDiskCache) ?? d.enableDiskCache
        self.diskCacheMaxGB = try container.decodeIfPresent(Float.self, forKey: .diskCacheMaxGB) ?? d.diskCacheMaxGB
        self.enableBlockDiskCache = try container.decodeIfPresent(Bool.self, forKey: .enableBlockDiskCache) ?? d.enableBlockDiskCache
        self.blockDiskCacheMaxGB = try container.decodeIfPresent(Float.self, forKey: .blockDiskCacheMaxGB) ?? d.blockDiskCacheMaxGB
        self.kvCacheQuantization = try container.decodeIfPresent(String.self, forKey: .kvCacheQuantization) ?? d.kvCacheQuantization
        self.kvCacheGroupSize = try container.decodeIfPresent(Int.self, forKey: .kvCacheGroupSize) ?? d.kvCacheGroupSize
        self.toolCallParser = try container.decodeIfPresent(String.self, forKey: .toolCallParser) ?? d.toolCallParser
        self.reasoningParser = try container.decodeIfPresent(String.self, forKey: .reasoningParser) ?? d.reasoningParser
        self.enableJIT = try container.decodeIfPresent(Bool.self, forKey: .enableJIT) ?? d.enableJIT
        self.idleSleepMode = try container.decodeIfPresent(String.self, forKey: .idleSleepMode) ?? d.idleSleepMode
        self.idleSleepMinutes = try container.decodeIfPresent(Int.self, forKey: .idleSleepMinutes) ?? d.idleSleepMinutes
        self.enableSoftSleep = try container.decodeIfPresent(Bool.self, forKey: .enableSoftSleep) ?? d.enableSoftSleep
        self.softSleepMinutes = try container.decodeIfPresent(Int.self, forKey: .softSleepMinutes) ?? d.softSleepMinutes
        self.enableDeepSleep = try container.decodeIfPresent(Bool.self, forKey: .enableDeepSleep) ?? d.enableDeepSleep
        self.deepSleepMinutes = try container.decodeIfPresent(Int.self, forKey: .deepSleepMinutes) ?? d.deepSleepMinutes
        self.defaultEnableThinking = try container.decodeIfPresent(String.self, forKey: .defaultEnableThinking)
        self.defaultTemperature = try container.decodeIfPresent(Float.self, forKey: .defaultTemperature)
        self.defaultTopP = try container.decodeIfPresent(Float.self, forKey: .defaultTopP)
        self.speculativeModel = try container.decodeIfPresent(String.self, forKey: .speculativeModel)
        self.numDraftTokens = try container.decodeIfPresent(Int.self, forKey: .numDraftTokens) ?? d.numDraftTokens
        self.enablePLD = try container.decodeIfPresent(Bool.self, forKey: .enablePLD) ?? d.enablePLD
    }

    /// Default configuration
    public static var `default`: ServerConfiguration {
        ServerConfiguration(
            port: 1337,
            exposeToNetwork: false,
            startAtLogin: false,
            hideDockIcon: false,
            appearanceMode: .system,
            numberOfThreads: ProcessInfo.processInfo.activeProcessorCount,
            backlog: 256,
            genTopP: 1.0,
            maxTokens: 32768,
            allowedOrigins: [],
            modelEvictionPolicy: .strictSingleModel,
            continuousBatching: true,
            maxNumSeqs: 256,
            streamInterval: 1,
            enablePrefixCache: true,
            prefixCacheSize: 100,
            cacheMemoryPercent: 0.30,
            cacheMemoryMB: nil,
            cacheTTLMinutes: 0,
            usePagedCache: true,
            pagedCacheBlockSize: 64,
            maxCacheBlocks: 1000,
            enableDiskCache: true,
            diskCacheMaxGB: 10.0,
            enableBlockDiskCache: true,
            blockDiskCacheMaxGB: 10.0,
            kvCacheQuantization: "none",
            kvCacheGroupSize: 64,
            toolCallParser: "auto",
            reasoningParser: "auto",
            enableJIT: true,
            idleSleepMode: "deep",
            idleSleepMinutes: 30,
            enableSoftSleep: true,
            softSleepMinutes: 10,
            enableDeepSleep: true,
            deepSleepMinutes: 30,
            defaultEnableThinking: nil,
            defaultTemperature: nil,
            defaultTopP: nil,
            speculativeModel: nil,
            numDraftTokens: 3,
            enablePLD: false
        )
    }

    // Memberwise init
    public init(
        port: Int, exposeToNetwork: Bool, startAtLogin: Bool,
        hideDockIcon: Bool = false, appearanceMode: AppearanceMode = .system,
        numberOfThreads: Int, backlog: Int32,
        genTopP: Float, maxTokens: Int = 32768,
        allowedOrigins: [String] = [],
        modelEvictionPolicy: ModelEvictionPolicy = .strictSingleModel,
        continuousBatching: Bool = true,
        maxNumSeqs: Int = 256, streamInterval: Int = 1,
        enablePrefixCache: Bool = true, prefixCacheSize: Int = 100,
        cacheMemoryPercent: Float = 0.30, cacheMemoryMB: Int? = nil, cacheTTLMinutes: Float = 0,
        usePagedCache: Bool = true, pagedCacheBlockSize: Int = 64, maxCacheBlocks: Int = 1000,
        enableDiskCache: Bool = true, diskCacheMaxGB: Float = 10.0,
        enableBlockDiskCache: Bool = true, blockDiskCacheMaxGB: Float = 10.0,
        kvCacheQuantization: String = "none", kvCacheGroupSize: Int = 64,
        toolCallParser: String = "auto", reasoningParser: String = "auto",
        enableJIT: Bool = true,
        idleSleepMode: String = "deep", idleSleepMinutes: Int = 30,
        enableSoftSleep: Bool = true, softSleepMinutes: Int = 10,
        enableDeepSleep: Bool = true, deepSleepMinutes: Int = 30,
        defaultEnableThinking: String? = nil,
        defaultTemperature: Float? = nil, defaultTopP: Float? = nil,
        speculativeModel: String? = nil, numDraftTokens: Int = 3, enablePLD: Bool = false
    ) {
        self.port = port
        self.exposeToNetwork = exposeToNetwork
        self.startAtLogin = startAtLogin
        self.hideDockIcon = hideDockIcon
        self.appearanceMode = appearanceMode
        self.numberOfThreads = numberOfThreads
        self.backlog = backlog
        self.genTopP = genTopP
        self.maxTokens = maxTokens
        self.allowedOrigins = allowedOrigins
        self.modelEvictionPolicy = modelEvictionPolicy
        self.continuousBatching = continuousBatching
        self.maxNumSeqs = maxNumSeqs
        self.streamInterval = streamInterval
        self.enablePrefixCache = enablePrefixCache
        self.prefixCacheSize = prefixCacheSize
        self.cacheMemoryPercent = cacheMemoryPercent
        self.cacheMemoryMB = cacheMemoryMB
        self.cacheTTLMinutes = cacheTTLMinutes
        self.usePagedCache = usePagedCache
        self.pagedCacheBlockSize = pagedCacheBlockSize
        self.maxCacheBlocks = maxCacheBlocks
        self.enableDiskCache = enableDiskCache
        self.diskCacheMaxGB = diskCacheMaxGB
        self.enableBlockDiskCache = enableBlockDiskCache
        self.blockDiskCacheMaxGB = blockDiskCacheMaxGB
        self.kvCacheQuantization = kvCacheQuantization
        self.kvCacheGroupSize = kvCacheGroupSize
        self.toolCallParser = toolCallParser
        self.reasoningParser = reasoningParser
        self.enableJIT = enableJIT
        self.idleSleepMode = idleSleepMode
        self.idleSleepMinutes = idleSleepMinutes
        self.enableSoftSleep = enableSoftSleep
        self.softSleepMinutes = softSleepMinutes
        self.enableDeepSleep = enableDeepSleep
        self.deepSleepMinutes = deepSleepMinutes
        self.defaultEnableThinking = defaultEnableThinking
        self.defaultTemperature = defaultTemperature
        self.defaultTopP = defaultTopP
        self.speculativeModel = speculativeModel
        self.numDraftTokens = numDraftTokens
        self.enablePLD = enablePLD
    }

    /// Validates if the port is in valid range
    public var isValidPort: Bool {
        (1 ..< 65536).contains(port)
    }
}

/// Policy for managing model eviction from memory
public enum ModelEvictionPolicy: String, Codable, CaseIterable, Sendable {
    /// Strictly keep only one model loaded at a time (safest for memory)
    case strictSingleModel = "Strict (One Model)"
    /// Allow multiple models (best for high RAM systems or rapid switching)
    case manualMultiModel = "Flexible (Multi Model)"

    public var description: String {
        switch self {
        case .strictSingleModel:
            return "Automatically unloads other models. Recommended for standard use."
        case .manualMultiModel:
            return "Keeps models loaded until manually unloaded. Requires 32GB+ RAM."
        }
    }
}
