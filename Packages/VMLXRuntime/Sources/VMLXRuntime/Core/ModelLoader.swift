import Foundation
import MLX
import MLXNN
import Tokenizers
import Hub

/// A loaded model ready for inference.
///
/// Holds the native VMLXRuntime model (Qwen3.5, etc.) with its tokenizer,
/// config dictionary, and detected model metadata.
public final class LoadedModel: @unchecked Sendable {

    /// The native model (VMLXNativeModel conformant Module).
    /// Handles the forward pass, KV/SSM cache creation, and weight sanitization.
    public let nativeModel: (any VMLXNativeModel & Module)

    /// Model configuration from config.json (raw dictionary for VMLXRuntime introspection).
    public let config: [String: Any]

    /// Detected model properties (name, family, hybrid, JANG, etc.).
    public let detected: DetectedModel

    /// Model directory path.
    public let modelPath: URL

    /// Tokenizer for this model.
    public let tokenizer: any Tokenizer

    /// Vocabulary size.
    public var vocabSize: Int {
        nativeModel.vocabularySize
    }

    /// Number of layers.
    public var numLayers: Int {
        if let nl = config["num_hidden_layers"] as? Int { return nl }
        if let tc = config["text_config"] as? [String: Any],
           let nl = tc["num_hidden_layers"] as? Int { return nl }
        return 32
    }

    /// Hidden dimension.
    public var hiddenSize: Int {
        if let hs = config["hidden_size"] as? Int { return hs }
        if let tc = config["text_config"] as? [String: Any],
           let hs = tc["hidden_size"] as? Int { return hs }
        return 4096
    }

    /// Number of attention heads.
    public var numAttentionHeads: Int {
        if let nh = config["num_attention_heads"] as? Int { return nh }
        if let tc = config["text_config"] as? [String: Any],
           let nh = tc["num_attention_heads"] as? Int { return nh }
        return 32
    }

    /// Number of KV heads (for GQA).
    public var numKVHeads: Int {
        if let nk = config["num_key_value_heads"] as? Int { return nk }
        if let tc = config["text_config"] as? [String: Any],
           let nk = tc["num_key_value_heads"] as? Int { return nk }
        return numAttentionHeads
    }

    /// EOS token IDs.
    public var eosTokenIds: Set<Int> {
        var ids = Set<Int>()
        if let eosIds = config["eos_token_id"] as? [Int] {
            ids.formUnion(eosIds)
        } else if let eosId = config["eos_token_id"] as? Int {
            ids.insert(eosId)
        }
        return ids
    }

    public init(nativeModel: any VMLXNativeModel & Module,
                config: [String: Any], detected: DetectedModel,
                modelPath: URL, tokenizer: any Tokenizer) {
        self.nativeModel = nativeModel
        self.config = config
        self.detected = detected
        self.modelPath = modelPath
        self.tokenizer = tokenizer
    }
}

/// Loads models from disk using VMLXRuntime's native model registry.
///
/// Uses `VMLXModelRegistry.loadModel(from:)` which handles:
/// - Config parsing and model architecture detection
/// - Quantized weight loading (QuantizedLinear with scales/biases)
/// - Correct weight key path mapping for each architecture
/// - Tokenizer setup via swift-transformers
public struct ModelLoader: Sendable {

    /// Load a model from a directory path using VMLXRuntime's native model registry.
    ///
    /// 1. Uses `VMLXModelRegistry.loadModel(from:)` for weight loading and model construction
    /// 2. Runs `ModelDetector.detect(at:)` for JANG/hybrid/family metadata
    /// 3. Loads tokenizer via swift-transformers
    public static func load(from path: URL) async throws -> LoadedModel {
        func _log(_ msg: String) {
            let line = "[\(Date())] \(msg)\n"
            if let fh = FileHandle(forWritingAtPath: "/tmp/vmlx_debug.log") {
                fh.seekToEndOfFile(); fh.write(line.data(using: .utf8)!); fh.closeFile()
            }
        }
        Self._vmlxLog("[ModelLoader] Step 1: detect at \(path.lastPathComponent)")
        let detected = try ModelDetector.detect(at: path)
        Self._vmlxLog("[ModelLoader] Step 1 done: \(detected.name) hybrid=\(detected.isHybrid)")

        Self._vmlxLog("[ModelLoader] Step 2: load config")
        let config = try _loadConfig(at: path)
        Self._vmlxLog("[ModelLoader] Step 2 done: model_type=\(config["model_type"] ?? "?")")

        Self._vmlxLog("[ModelLoader] Step 3: load model + weights")
        let nativeModel: any VMLXNativeModel & Module
        do {
            let (m, _) = try VMLXModelRegistry.loadModel(from: path)
            nativeModel = m
            Self._vmlxLog("[ModelLoader] Step 3 done: vocab=\(nativeModel.vocabularySize)")
        } catch {
            Self._vmlxLog("[ModelLoader] Step 3 FAILED: \(error)")
            throw error
        }

        Self._vmlxLog("[ModelLoader] Step 4: load tokenizer")
        let tokenizer: any Tokenizer
        do {
            tokenizer = try await _loadTokenizer(at: path)
            Self._vmlxLog("[ModelLoader] Step 4 done")
        } catch {
            Self._vmlxLog("[ModelLoader] Step 4 FAILED: \(error)")
            throw error
        }

        return LoadedModel(
            nativeModel: nativeModel,
            config: config,
            detected: detected,
            modelPath: path,
            tokenizer: tokenizer
        )
    }

    /// Load a model from a HuggingFace model name (downloads if needed).
    public static func loadFromHub(modelName: String) async throws -> LoadedModel {
        let hub = HubApi()
        let repo = Hub.Repo(id: modelName)
        let modelDir = try await hub.snapshot(from: repo)
        return try await load(from: modelDir)
    }

    // MARK: - Private

    private static func _loadConfig(at path: URL) throws -> [String: Any] {
        let configURL = path.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw ModelLoaderError.configNotFound(path.path)
        }
        let data = try Data(contentsOf: configURL)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelLoaderError.invalidConfig("Failed to parse config.json")
        }
        return json
    }

    private static func _vmlxLog(_ msg: String) {
        let line = "[\(Date())] \(msg)\n"
        if let fh = FileHandle(forWritingAtPath: "/tmp/vmlx_debug.log") {
            fh.seekToEndOfFile(); fh.write(line.data(using: .utf8)!); fh.closeFile()
        }
    }

    private static func _loadTokenizer(at path: URL) async throws -> any Tokenizer {
        let tokenizerURL = path.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw ModelLoaderError.tokenizerNotFound(path.path)
        }

        do {
            return try await AutoTokenizer.from(modelFolder: path)
        } catch {
            _vmlxLog("[ModelLoader] AutoTokenizer failed: \(error). Trying direct JSON...")
            let tokData = try Data(contentsOf: tokenizerURL)
            let tokConfigURL = path.appendingPathComponent("tokenizer_config.json")
            let tokConfigData = FileManager.default.fileExists(atPath: tokConfigURL.path)
                ? try Data(contentsOf: tokConfigURL) : "{}".data(using: .utf8)!

            guard let configDict = try JSONSerialization.jsonObject(with: tokConfigData) as? [NSString: Any],
                  let dataDict = try JSONSerialization.jsonObject(with: tokData) as? [NSString: Any]
            else {
                _vmlxLog("[ModelLoader] JSON parse failed for tokenizer")
                throw ModelLoaderError.tokenizerNotFound("Failed to parse tokenizer JSON at \(path.path)")
            }

            _vmlxLog("[ModelLoader] Creating tokenizer from JSON dicts...")
            return try AutoTokenizer.from(
                tokenizerConfig: Config(configDict),
                tokenizerData: Config(dataDict)
            )
        }
    }
}

// MARK: - Errors

public enum ModelLoaderError: Error, LocalizedError, Sendable {
    case configNotFound(String)
    case invalidConfig(String)
    case tokenizerNotFound(String)
    case weightsNotFound(String)
    case invalidWeightIndex
    case shardNotFound(String)
    case unsupportedArchitecture(String)

    public var errorDescription: String? {
        switch self {
        case .configNotFound(let p): return "config.json not found at: \(p)"
        case .invalidConfig(let m): return "Invalid config: \(m)"
        case .tokenizerNotFound(let p): return "Tokenizer not found at: \(p)"
        case .weightsNotFound(let p): return "No safetensors weights at: \(p)"
        case .invalidWeightIndex: return "Invalid model.safetensors.index.json"
        case .shardNotFound(let f): return "Weight shard not found: \(f)"
        case .unsupportedArchitecture(let a): return "Unsupported architecture: \(a)"
        }
    }
}
