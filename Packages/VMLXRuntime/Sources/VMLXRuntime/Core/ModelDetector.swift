import Foundation

/// Comprehensive model detection result combining info from
/// jang_config.json, config.json, model.safetensors.index.json,
/// and filesystem indicators.
public struct DetectedModel: Sendable {
    /// Display name (e.g., "Qwen3.5-4B-JANG_2S").
    public let name: String

    /// Model family (e.g., "qwen3.5"), lowercased.
    public let family: String

    /// Source model name (e.g., "Qwen3.5-4B").
    public let sourceModel: String

    /// JANG quantization profile (e.g., "JANG_2S"), nil if not JANG.
    public let jangProfile: String?

    /// Whether this is a JANG-quantized model.
    public let isJang: Bool

    // MARK: - Architecture

    /// Architecture type from jang_config (e.g., "hybrid_ssm", "moe", "hybrid_moe_ssm").
    public let architectureType: String

    /// Attention mechanism (e.g., "none", "gqa", "mla").
    public let attentionType: String

    /// Whether the model supports vision/multimodal input.
    public let hasVision: Bool

    /// Whether the model has SSM (state-space model) layers.
    public let hasSSM: Bool

    /// Whether the model is a Mixture-of-Experts model.
    public let hasMoE: Bool

    /// Whether the model is hybrid (has both SSM + attention layers).
    public let isHybrid: Bool

    // MARK: - Quantization

    /// JANG quantization profile name.
    public let quantProfile: String?

    /// Target bits per weight.
    public let targetBits: Float?

    /// Actual achieved bits per weight.
    public let actualBits: Float?

    /// Quantization block size.
    public let blockSize: Int?

    /// Bit widths used in mixed quantization.
    public let bitWidthsUsed: [Int]?

    // MARK: - Model Config (from config.json)

    /// HuggingFace model_type (e.g., "qwen3_5", "minimax_m2", "nemotron_h").
    public let modelType: String?

    /// HuggingFace architectures list.
    public let hfArchitectures: [String]?

    /// Hybrid override pattern for Nemotron-style models (e.g., "MEMEM*...").
    public let hybridOverridePattern: String?

    /// Per-layer type list for Qwen3.5-style models.
    public let layerTypes: [String]?

    /// Maximum context window size.
    public let contextWindow: Int?

    /// Vocabulary size.
    public let vocabSize: Int?

    /// Number of transformer/hybrid layers.
    public let numLayers: Int?

    /// Number of MoE experts (if applicable).
    public let numExperts: Int?

    /// Number of experts used per token.
    public let numExpertsPerTok: Int?

    // MARK: - MLA (from config.json)

    /// KV LoRA rank for MLA models.
    public let kvLoraRank: Int?

    /// QK nope head dimension for MLA models.
    public let qkNopeHeadDim: Int?

    /// QK rope head dimension for MLA models.
    public let qkRopeHeadDim: Int?

    // MARK: - Vision (from config.json)

    /// Image token ID.
    public let imageTokenId: Int?

    /// Video token ID.
    public let videoTokenId: Int?

    /// Whether a preprocessor_config.json exists (vision indicator).
    public let hasPreprocessorConfig: Bool

    // MARK: - Runtime

    /// Total weight size in bytes.
    public let totalWeightBytes: Int?

    /// Total weight size in GB.
    public let totalWeightGB: Float?

    /// Number of weight file shards.
    public let numShards: Int?

    // MARK: - File paths

    /// Root model directory.
    public let modelPath: URL

    /// Weight file names (relative to modelPath).
    public let weightFiles: [String]

    public init(
        name: String,
        family: String,
        sourceModel: String,
        jangProfile: String? = nil,
        isJang: Bool = false,
        architectureType: String = "transformer",
        attentionType: String = "gqa",
        hasVision: Bool = false,
        hasSSM: Bool = false,
        hasMoE: Bool = false,
        isHybrid: Bool = false,
        quantProfile: String? = nil,
        targetBits: Float? = nil,
        actualBits: Float? = nil,
        blockSize: Int? = nil,
        bitWidthsUsed: [Int]? = nil,
        modelType: String? = nil,
        hfArchitectures: [String]? = nil,
        hybridOverridePattern: String? = nil,
        layerTypes: [String]? = nil,
        contextWindow: Int? = nil,
        vocabSize: Int? = nil,
        numLayers: Int? = nil,
        numExperts: Int? = nil,
        numExpertsPerTok: Int? = nil,
        kvLoraRank: Int? = nil,
        qkNopeHeadDim: Int? = nil,
        qkRopeHeadDim: Int? = nil,
        imageTokenId: Int? = nil,
        videoTokenId: Int? = nil,
        hasPreprocessorConfig: Bool = false,
        totalWeightBytes: Int? = nil,
        totalWeightGB: Float? = nil,
        numShards: Int? = nil,
        modelPath: URL,
        weightFiles: [String] = []
    ) {
        self.name = name
        self.family = family
        self.sourceModel = sourceModel
        self.jangProfile = jangProfile
        self.isJang = isJang
        self.architectureType = architectureType
        self.attentionType = attentionType
        self.hasVision = hasVision
        self.hasSSM = hasSSM
        self.hasMoE = hasMoE
        self.isHybrid = isHybrid
        self.quantProfile = quantProfile
        self.targetBits = targetBits
        self.actualBits = actualBits
        self.blockSize = blockSize
        self.bitWidthsUsed = bitWidthsUsed
        self.modelType = modelType
        self.hfArchitectures = hfArchitectures
        self.hybridOverridePattern = hybridOverridePattern
        self.layerTypes = layerTypes
        self.contextWindow = contextWindow
        self.vocabSize = vocabSize
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.kvLoraRank = kvLoraRank
        self.qkNopeHeadDim = qkNopeHeadDim
        self.qkRopeHeadDim = qkRopeHeadDim
        self.imageTokenId = imageTokenId
        self.videoTokenId = videoTokenId
        self.hasPreprocessorConfig = hasPreprocessorConfig
        self.totalWeightBytes = totalWeightBytes
        self.totalWeightGB = totalWeightGB
        self.numShards = numShards
        self.modelPath = modelPath
        self.weightFiles = weightFiles
    }
}

// MARK: - Model Detector

/// Detects model properties by reading jang_config.json, config.json,
/// model.safetensors.index.json, and filesystem indicators from a model directory.
public struct ModelDetector: Sendable {

    /// Detect a model from a directory path.
    ///
    /// Reads all available config files and merges their information:
    /// 1. `jang_config.json` - JANG quantization and architecture info
    /// 2. `config.json` - HuggingFace model config (model_type, architectures, hybrid info, MLA, vision)
    /// 3. `model.safetensors.index.json` - weight file info (shards, total size)
    /// 4. `preprocessor_config.json` - vision model indicator
    public static func detect(at path: URL) throws -> DetectedModel {
        let fm = FileManager.default

        // 1. Try jang_config.json
        var jangConfig: JangConfig?
        var isJang = false
        if let jangURL = JangLoader.findConfigPath(at: path) {
            let data = try Data(contentsOf: jangURL)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                jangConfig = try JangLoader.parseConfig(from: json)
                isJang = true
            }
        }

        // 2. Try config.json (HuggingFace model config)
        var hfConfig: [String: Any]?
        let configURL = path.appendingPathComponent("config.json")
        if fm.fileExists(atPath: configURL.path) {
            let data = try Data(contentsOf: configURL)
            hfConfig = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        }

        // 3. Try model.safetensors.index.json
        var weightIndex: [String: Any]?
        let indexURL = path.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: indexURL.path) {
            let data = try Data(contentsOf: indexURL)
            weightIndex = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        }

        // 4. Check for preprocessor_config.json
        let preprocessorURL = path.appendingPathComponent("preprocessor_config.json")
        let hasPreprocessorConfig = fm.fileExists(atPath: preprocessorURL.path)

        // --- Merge information from all sources ---

        // Source model name
        let sourceModel = jangConfig?.sourceModel.name ?? ""
        let jangProfile = jangConfig?.quantization.profile

        // Build display name
        let dirName = path.lastPathComponent
        let name: String
        if isJang, let profile = jangProfile, !sourceModel.isEmpty {
            name = "\(sourceModel)-\(profile)"
        } else if !sourceModel.isEmpty {
            name = sourceModel
        } else {
            // For HF cache: extract repo name from parent path
            // Path: ~/.cache/huggingface/hub/models--org--name/snapshots/hash/
            let parentPath = path.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent
            if parentPath.hasPrefix("models--") {
                name = parentPath
                    .replacingOccurrences(of: "models--", with: "")
                    .replacingOccurrences(of: "--", with: "/")
            } else {
                name = dirName
            }
        }

        // Family detection
        let family = detectFamily(
            sourceModel: sourceModel,
            modelType: hfConfig?["model_type"] as? String,
            dirName: dirName
        )

        // Architecture from jang_config
        let jangArch = jangConfig?.architecture
        let archType = jangArch?.type ?? "transformer"
        let attentionType = jangArch?.attention ?? "gqa"

        // Vision detection: jang_config OR config.json OR preprocessor_config.json
        let jangVision = jangArch?.hasVision ?? false
        let hfVision = hfConfig?["image_token_id"] != nil
            || hfConfig?["video_token_id"] != nil
            || hfConfig?["vision_config"] != nil
        let hasVision = jangVision || hfVision || hasPreprocessorConfig

        // SSM detection: check jang_config first, then fall back to config.json evidence.
        // Some JANG configs have has_ssm=false despite the model having Mamba2 layers
        // (e.g., Nemotron Cascade JANG reports has_ssm=false but hybrid_override_pattern
        // contains "M" for Mamba2 layers). Always cross-check config.json.
        let jangHasSSM = jangArch?.hasSSM ?? false
        let hfHasSSM: Bool = {
            if let hop = hfConfig?["hybrid_override_pattern"] as? String, hop.contains("M") {
                return true
            }
            let lt: [String]? = (hfConfig?["layer_types"] as? [String])
                ?? (hfConfig?["text_config"] as? [String: Any])?["layer_types"] as? [String]
            if let types = lt {
                let ssmTypes: Set<String> = ["linear_attention", "ssm", "mamba", "recurrent", "gated_delta"]
                return types.contains(where: { ssmTypes.contains($0.lowercased()) })
            }
            return false
        }()
        let hasSSM = jangHasSSM || hfHasSSM

        // MoE detection: jang_config OR config.json
        let jangHasMoE = jangArch?.hasMoE ?? false
        let hfHasMoE: Bool = {
            let experts = hfConfig?["num_local_experts"] as? Int
                ?? hfConfig?["num_experts"] as? Int
                ?? (hfConfig?["text_config"] as? [String: Any])?["num_experts"] as? Int
                ?? (hfConfig?["text_config"] as? [String: Any])?["num_local_experts"] as? Int
            return (experts ?? 0) > 1
        }()
        let hasMoE = jangHasMoE || hfHasMoE

        // Hybrid = has SSM (mixed SSM+attention or SSM-only architecture)
        let isHybrid = hasSSM

        // HF config fields
        let modelType = hfConfig?["model_type"] as? String
        let hfArchitectures = hfConfig?["architectures"] as? [String]

        // Hybrid pattern from config.json
        let hybridOverridePattern = hfConfig?["hybrid_override_pattern"] as? String

        // Layer types from config.json (may be nested under text_config)
        let layerTypes: [String]?
        if let lt = hfConfig?["layer_types"] as? [String] {
            layerTypes = lt
        } else if let textConfig = hfConfig?["text_config"] as? [String: Any],
                  let lt = textConfig["layer_types"] as? [String] {
            layerTypes = lt
        } else {
            layerTypes = nil
        }

        // Context window
        let contextWindow = hfConfig?["max_position_embeddings"] as? Int
            ?? hfConfig?["max_seq_len"] as? Int

        // Model dimensions
        let vocabSize = hfConfig?["vocab_size"] as? Int
        let numLayers = hfConfig?["num_hidden_layers"] as? Int

        // MoE fields
        let numExperts = hfConfig?["num_local_experts"] as? Int
            ?? hfConfig?["num_experts"] as? Int
        let numExpertsPerTok = hfConfig?["num_experts_per_tok"] as? Int

        // MLA fields
        let kvLoraRank = hfConfig?["kv_lora_rank"] as? Int
        let qkNopeHeadDim = hfConfig?["qk_nope_head_dim"] as? Int
        let qkRopeHeadDim = hfConfig?["qk_rope_head_dim"] as? Int

        // Vision IDs
        let imageTokenId = hfConfig?["image_token_id"] as? Int
        let videoTokenId = hfConfig?["video_token_id"] as? Int

        // Weight info from index
        var totalWeightBytes: Int?
        var totalWeightGB: Float?
        var numShards: Int?
        var weightFiles: [String] = []

        if let index = weightIndex {
            if let meta = index["metadata"] as? [String: Any] {
                if let sizeStr = meta["total_size"] as? String, let size = Int(sizeStr) {
                    totalWeightBytes = size
                    totalWeightGB = Float(size) / (1024.0 * 1024.0 * 1024.0)
                } else if let size = meta["total_size"] as? Int {
                    totalWeightBytes = size
                    totalWeightGB = Float(size) / (1024.0 * 1024.0 * 1024.0)
                }
            }
            if let weightMap = index["weight_map"] as? [String: String] {
                let shardSet = Set(weightMap.values)
                weightFiles = shardSet.sorted()
                numShards = shardSet.count
            }
        }

        // Override with jang_config runtime if available
        if let jangRT = jangConfig?.runtime, jangRT.totalWeightBytes > 0 {
            totalWeightBytes = jangRT.totalWeightBytes
            totalWeightGB = jangRT.totalWeightGB
        }

        // If no weight index, look for safetensors files directly
        if weightFiles.isEmpty {
            let contents = (try? fm.contentsOfDirectory(atPath: path.path)) ?? []
            weightFiles = contents
                .filter { $0.hasSuffix(".safetensors") }
                .sorted()
            if numShards == nil && !weightFiles.isEmpty {
                numShards = weightFiles.count
            }
        }

        return DetectedModel(
            name: name,
            family: family,
            sourceModel: sourceModel,
            jangProfile: jangProfile,
            isJang: isJang,
            architectureType: archType,
            attentionType: attentionType,
            hasVision: hasVision,
            hasSSM: hasSSM,
            hasMoE: hasMoE,
            isHybrid: isHybrid,
            quantProfile: jangProfile,
            targetBits: jangConfig?.quantization.targetBits,
            actualBits: jangConfig?.quantization.actualBits,
            blockSize: jangConfig?.quantization.blockSize,
            bitWidthsUsed: jangConfig?.quantization.bitWidthsUsed,
            modelType: modelType,
            hfArchitectures: hfArchitectures,
            hybridOverridePattern: hybridOverridePattern,
            layerTypes: layerTypes,
            contextWindow: contextWindow,
            vocabSize: vocabSize,
            numLayers: numLayers,
            numExperts: numExperts,
            numExpertsPerTok: numExpertsPerTok,
            kvLoraRank: kvLoraRank,
            qkNopeHeadDim: qkNopeHeadDim,
            qkRopeHeadDim: qkRopeHeadDim,
            imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            hasPreprocessorConfig: hasPreprocessorConfig,
            totalWeightBytes: totalWeightBytes,
            totalWeightGB: totalWeightGB,
            numShards: numShards,
            modelPath: path,
            weightFiles: weightFiles
        )
    }

    /// Scan well-known model directories for all available models.
    ///
    /// Checks:
    /// - `~/.cache/huggingface/hub/models--*/snapshots/*/`
    /// - `~/jang/models/*/`
    /// - `~/models/*/`
    /// - `~/.mlxstudio/models/*/`
    /// - `~/.osaurus/models/*/`
    /// User-configured additional model directories.
    /// Set from the app's settings to add custom scan paths.
    public nonisolated(unsafe) static var additionalDirectories: [URL] = []

    public static func scanAvailableModels() -> [DetectedModel] {
        let fm = FileManager.default
        let home = fm.homeDirectoryForCurrentUser

        var models: [DetectedModel] = []

        // HuggingFace cache: ~/.cache/huggingface/hub/models--*/snapshots/*/
        let hfHub = home.appendingPathComponent(".cache/huggingface/hub")
        if let modelDirs = try? fm.contentsOfDirectory(atPath: hfHub.path) {
            for modelDir in modelDirs where modelDir.hasPrefix("models--") {
                let snapshotsDir = hfHub
                    .appendingPathComponent(modelDir)
                    .appendingPathComponent("snapshots")
                if let snapshots = try? fm.contentsOfDirectory(atPath: snapshotsDir.path) {
                    // Use the most recent snapshot (last alphabetically, as they're hashes)
                    for snapshot in snapshots {
                        let snapPath = snapshotsDir.appendingPathComponent(snapshot)
                        if let model = try? detect(at: snapPath) {
                            models.append(model)
                        }
                    }
                }
            }
        }

        // Custom model directories (built-in + user-configured)
        var customDirs = [
            home.appendingPathComponent("jang/models"),
            home.appendingPathComponent("models"),
            home.appendingPathComponent("MLXModels"),
            home.appendingPathComponent(".mlxstudio/models"),
            home.appendingPathComponent(".osaurus/models"),
        ]
        customDirs.append(contentsOf: additionalDirectories)

        for baseDir in customDirs {
            guard let entries = try? fm.contentsOfDirectory(atPath: baseDir.path) else { continue }
            for entry in entries {
                let entryPath = baseDir.appendingPathComponent(entry)
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: entryPath.path, isDirectory: &isDir),
                      isDir.boolValue else { continue }

                // Try direct detection (single-level: ~/models/model-name/)
                if let model = try? detect(at: entryPath) {
                    models.append(model)
                } else {
                    // Try two-level org/repo structure (~/MLXModels/JANGQ-AI/model-name/)
                    if let subEntries = try? fm.contentsOfDirectory(atPath: entryPath.path) {
                        for subEntry in subEntries {
                            let subPath = entryPath.appendingPathComponent(subEntry)
                            var subIsDir: ObjCBool = false
                            guard fm.fileExists(atPath: subPath.path, isDirectory: &subIsDir),
                                  subIsDir.boolValue else { continue }
                            if let model = try? detect(at: subPath) {
                                models.append(model)
                            }
                        }
                    }
                }
            }
        }

        // Deduplicate: same model can appear in multiple directories or HF snapshots.
        // Keep first occurrence (priority order: HF cache, then custom dirs).
        var seen = Set<String>()
        var unique: [DetectedModel] = []
        for model in models {
            let key = model.name.lowercased()
            if !seen.contains(key) {
                seen.insert(key)
                unique.append(model)
            }
        }
        return unique
    }

    // MARK: - Private Helpers

    /// Detect model family from source model name, HF model_type, or directory name.
    private static func detectFamily(
        sourceModel: String,
        modelType: String?,
        dirName: String
    ) -> String {
        // Priority 1: HF model_type
        if let mt = modelType {
            let normalized = mt.lowercased()
                .replacingOccurrences(of: "_", with: "")
            // Map known model_type values to families
            if normalized.contains("qwen3") { return "qwen3.5" }
            if normalized.contains("qwen2") { return "qwen2.5" }
            if normalized.contains("qwen") { return "qwen" }
            if normalized.contains("llama") { return "llama" }
            if normalized.contains("mistral") { return "mistral" }
            if normalized.contains("deepseek") { return "deepseek" }
            if normalized.contains("nemotron") { return "nemotron" }
            if normalized.contains("gemma") { return "gemma" }
            if normalized.contains("phi") { return "phi" }
            if normalized.contains("minimax") || normalized.contains("m2") { return "minimax" }
            if normalized.contains("jamba") { return "jamba" }
            if normalized.contains("intern") { return "internvl" }
            return mt.lowercased()
        }

        // Priority 2: source model name from jang_config
        let candidates = [sourceModel, dirName]
        for candidate in candidates where !candidate.isEmpty {
            let lower = candidate.lowercased()
            if lower.contains("qwen3.5") || lower.contains("qwen3_5") { return "qwen3.5" }
            if lower.contains("qwen3") { return "qwen3" }
            if lower.contains("qwen2.5") { return "qwen2.5" }
            if lower.contains("qwen") { return "qwen" }
            if lower.contains("llama-4") || lower.contains("llama4") { return "llama4" }
            if lower.contains("llama") { return "llama" }
            if lower.contains("mistral") { return "mistral" }
            if lower.contains("deepseek") { return "deepseek" }
            if lower.contains("nemotron") { return "nemotron" }
            if lower.contains("gemma") { return "gemma" }
            if lower.contains("phi") { return "phi" }
            if lower.contains("minimax") { return "minimax" }
            if lower.contains("jamba") { return "jamba" }
            if lower.contains("intern") { return "internvl" }
        }

        return "unknown"
    }
}
