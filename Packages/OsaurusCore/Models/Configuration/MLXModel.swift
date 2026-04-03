//
//  MLXModel.swift
//  osaurus
//
//  Created by Terence on 8/17/25.
//

import Foundation

/// Represents an MLX-compatible LLM that can be downloaded and used
struct MLXModel: Identifiable, Codable {
    let id: String
    let name: String
    let description: String
    let downloadURL: String

    /// Whether this model should appear at the top of the suggested models list
    let isTopSuggestion: Bool

    /// Approximate download size in bytes (optional, for display purposes)
    let downloadSizeBytes: Int64?

    // When non-nil, pins the model to a specific directory (used by tests).
    // When nil, `localDirectory` resolves dynamically so that user-selected
    // storage path changes are always respected.
    private let rootDirectory: URL?

    init(
        id: String,
        name: String,
        description: String,
        downloadURL: String,
        isTopSuggestion: Bool = false,
        downloadSizeBytes: Int64? = nil,
        rootDirectory: URL? = nil
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.downloadURL = downloadURL
        self.isTopSuggestion = isTopSuggestion
        self.downloadSizeBytes = downloadSizeBytes
        self.rootDirectory = rootDirectory
    }

    /// Formatted download size string (e.g., "3.9 GB")
    var formattedDownloadSize: String? {
        guard let bytes = totalSizeEstimateBytes else { return nil }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.includesUnit = true
        return formatter.string(fromByteCount: bytes)
    }

    /// Best estimate of the total model size in bytes.
    /// Uses explicit downloadSizeBytes if available, otherwise estimates based on parameters/quantization.
    var totalSizeEstimateBytes: Int64? {
        if let bytes = downloadSizeBytes { return bytes }

        // Estimate based on params and quantization (without the runtime overhead multiplier)
        if let params = parameterCountBillions {
            return Int64(params * bytesPerParameter * 1024 * 1024 * 1024)
        }

        return nil
    }

    /// Local directory where this model should be stored.
    /// If the id is an absolute filesystem path, use it directly.
    /// Otherwise resolves against the models directory.
    var localDirectory: URL {
        // If id is an absolute path (from local model discovery), use it directly
        if id.hasPrefix("/") {
            return URL(fileURLWithPath: id, isDirectory: true)
        }
        let baseDir = rootDirectory ?? DirectoryPickerService.effectiveModelsDirectory()
        let components = id.split(separator: "/").map(String.init)
        return components.reduce(baseDir) { partial, component in
            partial.appendingPathComponent(component, isDirectory: true)
        }
    }

    /// Check if model is downloaded
    /// A model is considered complete if:
    /// - Core config exists: config.json
    /// - Tokenizer assets exist in ANY of the supported variants:
    ///   - tokenizer.json (HF consolidated JSON)
    ///   - BPE: merges.txt + (vocab.json OR vocab.txt)
    ///   - SentencePiece: tokenizer.model OR spiece.model
    /// - At least one *.safetensors file exists (weights)
    var isDownloaded: Bool {
        let fileManager = FileManager.default
        let directory = localDirectory

        func exists(_ name: String) -> Bool {
            fileManager.fileExists(atPath: directory.appendingPathComponent(name).path)
        }

        // Core config
        guard exists("config.json") else { return false }

        // Tokenizer variants (including tokenizer_config.json for JANG models)
        let hasTokenizerJSON = exists("tokenizer.json")
        let hasBPE = exists("merges.txt") && (exists("vocab.json") || exists("vocab.txt"))
        let hasSentencePiece = exists("tokenizer.model") || exists("spiece.model")
        let hasTokenizerConfig = exists("tokenizer_config.json")
        let hasTokenizerAssets = hasTokenizerJSON || hasBPE || hasSentencePiece || hasTokenizerConfig
        guard hasTokenizerAssets else { return false }

        // Weights
        if let items = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ) {
            let hasWeights = items.contains { $0.pathExtension == "safetensors" }
            return hasWeights
        }
        return false
    }

    /// Approximate download timestamp based on directory creation/modification time
    /// Newer downloads should have more recent dates.
    var downloadedAt: Date? {
        let directory = localDirectory
        let values = try? directory.resourceValues(forKeys: [
            .creationDateKey, .contentModificationDateKey,
        ])
        return values?.creationDate ?? values?.contentModificationDate
    }

    // MARK: - Metadata Extraction

    /// Extracts parameter count from model name/id (e.g., "1.7B", "7B", "30B", "235B")
    var parameterCount: String? {
        let text = id.lowercased()
        // Match patterns like: 1.7b, 7b, 30b, 235b, 270m, 120b, etc.
        // Also handles formats like "3n-E4B" (Gemma 3n) or "A22B" (MoE active params)
        let patterns = [
            #"(\d+\.?\d*)[bm](?:-|$|\s|[^a-z])"#,  // Standard: 7b, 1.7b, 270m
            #"(\d+\.?\d*)b-"#,  // With suffix: 7b-instruct
            #"-(\d+\.?\d*)[bm]-"#,  // Middle: llama-7b-instruct
            #"[- ](\d+\.?\d*)[bm]$"#,  // End: model-7b
            #"e(\d+)[bm]"#,  // Gemma style: E4B
            #"a(\d+)[bm]"#,  // MoE active: A22B
        ]

        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) {
                let range = NSRange(text.startIndex..., in: text)
                if let match = regex.firstMatch(in: text, options: [], range: range) {
                    if let numRange = Range(match.range(at: 1), in: text) {
                        let number = String(text[numRange])
                        // Determine unit (B or M)
                        let fullMatch = String(text[Range(match.range, in: text)!]).uppercased()
                        let unit = fullMatch.contains("M") ? "M" : "B"
                        return "\(number)\(unit)"
                    }
                }
            }
        }
        return nil
    }

    /// Extracts quantization level from model name/id (e.g., "4-bit", "8-bit", "fp16")
    var quantization: String? {
        let text = id.lowercased()

        // Check for bit patterns: 4bit, 4-bit, 8bit, 8-bit, etc.
        if let regex = try? NSRegularExpression(pattern: #"(\d+)-?bit"#, options: .caseInsensitive) {
            let range = NSRange(text.startIndex..., in: text)
            if let match = regex.firstMatch(in: text, options: [], range: range) {
                if let numRange = Range(match.range(at: 1), in: text) {
                    return "\(text[numRange])-bit"
                }
            }
        }

        // Check for precision formats: fp16, bf16, fp32
        if text.contains("fp16") { return "FP16" }
        if text.contains("bf16") { return "BF16" }
        if text.contains("fp32") { return "FP32" }

        return nil
    }

    /// Determines if this model is a Vision Language Model (VLM) based on name heuristics
    /// For downloaded models, use ModelManager.isVisionModel() for accurate detection
    var isLikelyVLM: Bool {
        let lowerId = id.lowercased()
        let vlmIndicators = [
            "-vl-", "-vl", "vl-",  // Qwen2-VL, Kimi-VL
            "llava",  // LLaVA family
            "pixtral",  // Mistral's vision model
            "paligemma",  // Google's VLM
            "idefics",  // HF's VLM
            "internvl",  // InternVL
            "cogvlm",  // CogVLM
            "minicpm-v",  // MiniCPM-V
            "phi3-v", "phi-3-v",  // Phi-3-Vision
            "florence",  // Florence
            "blip",  // BLIP family
            "instructblip",  // InstructBLIP
            "vision",  // Generic vision indicator
        ]
        return vlmIndicators.contains { lowerId.contains($0) }
    }

    /// Extracts the model family from the name/id (e.g., "Llama", "Qwen", "Gemma", "Phi")
    var family: String {
        let name = self.name.lowercased()

        // 1. Check for common families first (strong matches)
        let strongMatches = [
            "llama": "Llama",
            "qwen": "Qwen",
            "gemma": "Gemma",
            "phi": "Phi",
            "mistral": "Mistral",
            "mixtral": "Mixtral",
            "deepseek": "DeepSeek",
            "nemotron": "Nemotron",
            "command-r": "Command-R",
            "grok": "Grok",
            "yi": "Yi",
            "falcon": "Falcon",
            "internlm": "InternLM",
            "stablelm": "StableLM",
            "smollm": "SmolLM",
            "hermes": "Hermes",
            "liquid": "Liquid",
            "lfm": "Liquid",
            "starcoder": "StarCoder",
            "granite": "Granite",
            "exat": "Exat",
            "opcoder": "OpCoder",
            "opencoder": "OpenCoder",
        ]

        for (key, value) in strongMatches {
            if name.contains(key) { return value }
        }

        // 2. Fallback heuristic: clean up the name and take the first part
        // Remove common vendor prefixes
        var cleaned = self.name
        let prefixes = [
            "Meta-", "Google-", "Mistral-", "MistralAI-", "Microsoft-", "NousResearch-", "Qwen-", "DeepSeek-",
        ]
        for prefix in prefixes {
            if cleaned.hasPrefix(prefix) {
                cleaned = String(cleaned.dropFirst(prefix.count))
                break
            }
        }

        // Take first semantic part (before dash or dot)
        let parts = cleaned.components(separatedBy: CharacterSet(charactersIn: "-. "))
        if let first = parts.first, !first.isEmpty {
            // Filter out junk or purely numeric parts
            if first.rangeOfCharacter(from: .letters) != nil {
                return first.capitalized
            }
        }

        return "Other"
    }

    /// Model type enum for display purposes
    enum ModelType: String, CaseIterable, Identifiable {
        case llm = "LLM"
        case vlm = "VLM"

        var id: String { rawValue }
    }

    /// Returns the model type based on heuristics
    var modelType: ModelType {
        isLikelyVLM ? .vlm : .llm
    }

    // MARK: - Memory Estimation & Hardware Compatibility

    private static let bytesPerGB: Double = 1024 * 1024 * 1024
    private static let overheadMultiplier: Double = 1.2

    /// Numeric parameter count in billions (e.g. "7B" -> 7.0, "270M" -> 0.27)
    var parameterCountBillions: Double? {
        guard let params = parameterCount else { return nil }
        let text = params.uppercased()
        guard let num = Double(text.dropLast()) else { return nil }
        return text.hasSuffix("M") ? num / 1000.0 : num
    }

    /// Bytes per parameter based on the quantization extracted from the model name.
    private var bytesPerParameter: Double {
        guard let quant = quantization?.lowercased() else { return 0.5 }

        let bitWidths: [(String, Double)] = [
            ("2-bit", 0.25), ("3-bit", 0.375), ("4-bit", 0.5),
            ("5-bit", 0.625), ("6-bit", 0.75), ("8-bit", 1.0),
        ]
        for (label, bytes) in bitWidths {
            if quant.contains(label) { return bytes }
        }

        switch quant {
        case "fp16", "bf16": return 2.0
        case "fp32": return 4.0
        default: return 0.5
        }
    }

    /// Estimated memory required to run this model (in GB), including overhead
    /// for KV cache, activations, and runtime buffers.
    var estimatedMemoryGB: Double? {
        if let params = parameterCountBillions {
            return params * bytesPerParameter * 1e9 * Self.overheadMultiplier / Self.bytesPerGB
        }
        if let dlBytes = downloadSizeBytes {
            return Double(dlBytes) * Self.overheadMultiplier / Self.bytesPerGB
        }
        return nil
    }

    /// Formatted estimated memory string (e.g. "~3.5 GB")
    var formattedEstimatedMemory: String? {
        guard let gb = estimatedMemoryGB else { return nil }
        return gb < 1.0
            ? String(format: "~%.0f MB", gb * 1024)
            : String(format: "~%.1f GB", gb)
    }

    /// Assess whether this model can run on the given hardware.
    func compatibility(totalMemoryGB: Double) -> ModelCompatibility {
        guard let required = estimatedMemoryGB, totalMemoryGB > 0 else { return .unknown }
        let ratio = required / totalMemoryGB
        if ratio < 0.75 { return .compatible }
        if ratio < 0.95 { return .tight }
        return .tooLarge
    }
}

/// Hardware compatibility assessment for a model.
enum ModelCompatibility {
    case compatible
    case tight
    case tooLarge
    case unknown
}

/// Download state for tracking progress
enum DownloadState: Equatable {
    case notStarted
    case downloading(progress: Double)
    case completed
    case failed(error: String)
}
