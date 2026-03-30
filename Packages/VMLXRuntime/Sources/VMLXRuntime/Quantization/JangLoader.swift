import Foundation

/// JANG config file name.
public let jangConfigFileName = "jang_config.json"

/// Legacy config file names to search for (fallback only).
public let jangConfigFileNames = [
    "jang_config.json",
    "jjqf_config.json",
    "jang_cfg.json",
    "mxq_config.json"
]

// MARK: - JANG Config Structs

/// Quantization settings from jang_config.json `quantization` block.
public struct JangQuantization: Sendable, Equatable {
    public let method: String            // "jang-importance"
    public let profile: String           // "JANG_2S", "JANG_4K", etc.
    public let targetBits: Float         // e.g., 2.5
    public let actualBits: Float         // e.g., 2.85
    public let blockSize: Int            // e.g., 64
    public let calibrationMethod: String // "weights"
    public let quantizationMethod: String // "mse"
    public let scoringMethod: String     // "weight-magnitude"
    public let bitWidthsUsed: [Int]      // [2, 4, 6]
    public let quantizationScheme: String // "asymmetric"
    public let quantizationBackend: String // "mx.quantize"

    public init(
        method: String = "jang-importance",
        profile: String = "JANG_2S",
        targetBits: Float = 2.5,
        actualBits: Float = 2.85,
        blockSize: Int = 64,
        calibrationMethod: String = "weights",
        quantizationMethod: String = "mse",
        scoringMethod: String = "weight-magnitude",
        bitWidthsUsed: [Int] = [2, 4, 6],
        quantizationScheme: String = "asymmetric",
        quantizationBackend: String = "mx.quantize"
    ) {
        self.method = method
        self.profile = profile
        self.targetBits = targetBits
        self.actualBits = actualBits
        self.blockSize = blockSize
        self.calibrationMethod = calibrationMethod
        self.quantizationMethod = quantizationMethod
        self.scoringMethod = scoringMethod
        self.bitWidthsUsed = bitWidthsUsed
        self.quantizationScheme = quantizationScheme
        self.quantizationBackend = quantizationBackend
    }
}

/// Source model info from jang_config.json `source_model` block.
public struct JangSourceModel: Sendable, Equatable {
    public let name: String       // "Qwen3.5-4B"
    public let dtype: String      // "bfloat16"
    public let parameters: String // "4025327616" (string in JSON)

    public init(name: String = "", dtype: String = "bfloat16", parameters: String = "0") {
        self.name = name
        self.dtype = dtype
        self.parameters = parameters
    }

    /// Parameter count as integer.
    public var parameterCount: Int { Int(parameters) ?? 0 }
}

/// Architecture info from jang_config.json `architecture` block.
public struct JangArchitecture: Sendable, Equatable {
    public let type: String       // "hybrid_ssm", "moe", "hybrid_moe_ssm"
    public let attention: String  // "none", "gqa", "mla"
    public let hasVision: Bool
    public let hasSSM: Bool
    public let hasMoE: Bool

    public init(
        type: String = "transformer",
        attention: String = "gqa",
        hasVision: Bool = false,
        hasSSM: Bool = false,
        hasMoE: Bool = false
    ) {
        self.type = type
        self.attention = attention
        self.hasVision = hasVision
        self.hasSSM = hasSSM
        self.hasMoE = hasMoE
    }
}

/// Runtime info from jang_config.json `runtime` block.
public struct JangRuntime: Sendable, Equatable {
    public let totalWeightBytes: Int
    public let totalWeightGB: Float

    public init(totalWeightBytes: Int = 0, totalWeightGB: Float = 0) {
        self.totalWeightBytes = totalWeightBytes
        self.totalWeightGB = totalWeightGB
    }
}

/// Parsed JANG model configuration from jang_config.json.
///
/// Matches the real on-disk format:
/// ```json
/// {
///   "quantization": { ... },
///   "source_model": { ... },
///   "architecture": { ... },
///   "runtime": { ... },
///   "format": "jang",
///   "format_version": "2.0"
/// }
/// ```
public struct JangConfig: Sendable {
    /// Format identifier: "jang".
    public let format: String

    /// Format version: "2.0" (current), "1.0" (legacy).
    public let formatVersion: String

    /// Whether this is v2 format (MLX-native safetensors).
    public var isV2: Bool { formatVersion.hasPrefix("2") }

    /// Quantization settings.
    public let quantization: JangQuantization

    /// Source model information.
    public let sourceModel: JangSourceModel

    /// Architecture description.
    public let architecture: JangArchitecture

    /// Runtime size information.
    public let runtime: JangRuntime

    public init(
        format: String = "jang",
        formatVersion: String = "2.0",
        quantization: JangQuantization = JangQuantization(),
        sourceModel: JangSourceModel = JangSourceModel(),
        architecture: JangArchitecture = JangArchitecture(),
        runtime: JangRuntime = JangRuntime()
    ) {
        self.format = format
        self.formatVersion = formatVersion
        self.quantization = quantization
        self.sourceModel = sourceModel
        self.architecture = architecture
        self.runtime = runtime
    }
}

// MARK: - JANG Loader

/// JANG model loader.
/// Auto-detects JANG models, parses jang_config.json, and provides
/// architecture/quantization introspection.
public struct JangLoader: Sendable {

    /// Check if a model directory contains a JANG model.
    public static func isJangModel(at path: URL) -> Bool {
        findConfigPath(at: path) != nil
    }

    /// Find the JANG config file in a model directory.
    /// Checks the primary name first, then legacy fallbacks.
    public static func findConfigPath(at modelPath: URL) -> URL? {
        for name in jangConfigFileNames {
            let configURL = modelPath.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: configURL.path) {
                return configURL
            }
        }
        return nil
    }

    /// Load and parse the JANG config from a model directory.
    public static func loadConfig(at modelPath: URL) throws -> JangConfig {
        guard let configURL = findConfigPath(at: modelPath) else {
            throw JangLoaderError.configNotFound(modelPath.path)
        }

        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw JangLoaderError.invalidConfig("Failed to parse JSON")
        }

        return try parseConfig(from: json)
    }

    /// Parse a JangConfig from a raw JSON dictionary.
    public static func parseConfig(from json: [String: Any]) throws -> JangConfig {
        let format = json["format"] as? String ?? "jang"
        let formatVersion = json["format_version"] as? String ?? "2.0"

        // Parse quantization block
        let quantization: JangQuantization
        if let qDict = json["quantization"] as? [String: Any] {
            quantization = JangQuantization(
                method: qDict["method"] as? String ?? "jang-importance",
                profile: qDict["profile"] as? String ?? "JANG_2S",
                targetBits: floatValue(qDict["target_bits"]) ?? 2.5,
                actualBits: floatValue(qDict["actual_bits"]) ?? 2.5,
                blockSize: qDict["block_size"] as? Int ?? 64,
                calibrationMethod: qDict["calibration_method"] as? String ?? "weights",
                quantizationMethod: qDict["quantization_method"] as? String ?? "mse",
                scoringMethod: qDict["scoring_method"] as? String ?? "weight-magnitude",
                bitWidthsUsed: qDict["bit_widths_used"] as? [Int] ?? [],
                quantizationScheme: qDict["quantization_scheme"] as? String ?? "asymmetric",
                quantizationBackend: qDict["quantization_backend"] as? String ?? "mx.quantize"
            )
        } else {
            quantization = JangQuantization()
        }

        // Parse source_model block
        let sourceModel: JangSourceModel
        if let smDict = json["source_model"] as? [String: Any] {
            // parameters can be string or int in JSON
            let params: String
            if let s = smDict["parameters"] as? String {
                params = s
            } else if let n = smDict["parameters"] as? Int {
                params = String(n)
            } else {
                params = "0"
            }
            sourceModel = JangSourceModel(
                name: smDict["name"] as? String ?? "",
                dtype: smDict["dtype"] as? String ?? "bfloat16",
                parameters: params
            )
        } else {
            sourceModel = JangSourceModel()
        }

        // Parse architecture block
        let architecture: JangArchitecture
        if let aDict = json["architecture"] as? [String: Any] {
            architecture = JangArchitecture(
                type: aDict["type"] as? String ?? "transformer",
                attention: aDict["attention"] as? String ?? "gqa",
                hasVision: aDict["has_vision"] as? Bool ?? false,
                hasSSM: aDict["has_ssm"] as? Bool ?? false,
                hasMoE: aDict["has_moe"] as? Bool ?? false
            )
        } else {
            architecture = JangArchitecture()
        }

        // Parse runtime block
        let runtime: JangRuntime
        if let rDict = json["runtime"] as? [String: Any] {
            runtime = JangRuntime(
                totalWeightBytes: rDict["total_weight_bytes"] as? Int ?? 0,
                totalWeightGB: floatValue(rDict["total_weight_gb"]) ?? 0
            )
        } else {
            runtime = JangRuntime()
        }

        return JangConfig(
            format: format,
            formatVersion: formatVersion,
            quantization: quantization,
            sourceModel: sourceModel,
            architecture: architecture,
            runtime: runtime
        )
    }

    // MARK: - Architecture Introspection

    /// Whether the model is hybrid (has SSM layers).
    public static func isHybridModel(config: JangConfig) -> Bool {
        config.architecture.hasSSM
    }

    /// Whether the model supports vision/multimodal input.
    public static func isVisionModel(config: JangConfig) -> Bool {
        config.architecture.hasVision
    }

    /// Whether the model uses MLA (Multi-head Latent Attention).
    public static func isMLA(config: JangConfig) -> Bool {
        config.architecture.attention == "mla"
    }

    /// Whether the model is a Mixture-of-Experts model.
    public static func isMoE(config: JangConfig) -> Bool {
        config.architecture.hasMoE
    }

    // MARK: - TurboQuant Integration

    /// Build a TurboQuantConfig from JANG quantization profile info.
    ///
    /// Maps JANG profiles to TQ settings:
    /// - Uses `quantization.block_size` as a hint for quality tier
    /// - Uses `quantization.bit_widths_used` to determine critical layer bits
    /// - Higher bit widths in the profile -> higher critical layer bits
    public static func buildTQConfig(from jangConfig: JangConfig) -> TurboQuantConfig? {
        let q = jangConfig.quantization
        let bitWidths = q.bitWidthsUsed

        // No bit widths means no quantization info to work with
        guard !bitWidths.isEmpty else { return nil }

        let maxBits = bitWidths.max() ?? 4
        let minBits = bitWidths.min() ?? 2

        // Map JANG quantization to TQ cache compression:
        // - Default KV bits: use the minimum bit width (aggressive compression for most layers)
        // - Critical KV bits: use the maximum bit width (preserve quality for critical layers)
        let defaultBits = max(minBits, 2)  // floor at 2
        let criticalBits = min(maxBits, 8) // cap at 8

        // Build layer pattern for hybrid models
        var layerPattern: [LayerType]?
        if jangConfig.architecture.hasSSM {
            // Hybrid models: architecture.type hints at layer composition
            // Actual per-layer pattern comes from config.json (hybridOverridePattern / layer_types)
            // which ModelDetector merges. Here we just note it's hybrid.
            layerPattern = nil // Caller should provide from config.json
        }

        return TurboQuantConfig(
            defaultKeyBits: defaultBits,
            defaultValueBits: defaultBits,
            criticalLayers: [0, 1, 2, -3, -2, -1],
            criticalKeyBits: criticalBits,
            criticalValueBits: criticalBits,
            layerPattern: layerPattern
        )
    }

    /// Build a TurboQuantConfig with an explicit layer pattern (from config.json).
    ///
    /// Use this when you have hybrid layer info from HuggingFace config.json
    /// (e.g., `hybrid_override_pattern` or `text_config.layer_types`).
    public static func buildTQConfig(
        from jangConfig: JangConfig,
        layerPattern: [LayerType]?,
        kvLoraRank: Int? = nil,
        qkNopeHeadDim: Int? = nil,
        qkRopeHeadDim: Int? = nil,
        vHeadDim: Int? = nil
    ) -> TurboQuantConfig? {
        guard var tqConfig = buildTQConfig(from: jangConfig) else { return nil }

        tqConfig.layerPattern = layerPattern

        // MLA dimensions (from config.json for DeepSeek/Mistral-style models)
        if let rank = kvLoraRank, rank > 0 {
            if let nope = qkNopeHeadDim, let rope = qkRopeHeadDim {
                tqConfig.mlaKeyDim = nope + rope
            }
            tqConfig.mlaValueDim = vHeadDim
        }

        return tqConfig
    }

    // MARK: - Helpers

    /// Extract a Float from JSON values that may be Int, Double, or Float.
    private static func floatValue(_ value: Any?) -> Float? {
        if let d = value as? Double { return Float(d) }
        if let f = value as? Float { return f }
        if let i = value as? Int { return Float(i) }
        return nil
    }
}

// MARK: - Errors

public enum JangLoaderError: Error, LocalizedError, Sendable {
    case configNotFound(String)
    case invalidConfig(String)
    case unsupportedVersion(String)
    case loadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .configNotFound(let path): return "JANG config not found at: \(path)"
        case .invalidConfig(let msg): return "Invalid JANG config: \(msg)"
        case .unsupportedVersion(let ver): return "Unsupported JANG version: \(ver)"
        case .loadFailed(let msg): return "JANG load failed: \(msg)"
        }
    }
}
