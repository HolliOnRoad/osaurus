import Foundation

/// Tool call format used by a model family.
public enum ToolCallFormat: String, Sendable {
    case qwen = "qwen"
    case llama = "llama"
    case mistral = "mistral"
    case deepseek = "deepseek"
    case hermes = "hermes"
    case functionary = "functionary"
    case granite = "granite"
    case glm = "glm"
    case minimax = "minimax"
    case nemotron = "nemotron"
    case xlam = "xlam"
    case moonshot = "moonshot"
    case stepfun = "stepfun"
    case generic = "generic"
    case none = "none"
}

/// Reasoning/thinking format used by a model.
public enum ReasoningFormat: String, Sendable {
    case qwen3 = "qwen3"          // <think>...</think>
    case deepseekR1 = "deepseek_r1"
    case gptoss = "gptoss"
    case mistral = "mistral"
    case none = "none"
}

/// Configuration for a specific model family.
public struct ModelFamilyConfig: Sendable {
    /// Model family name (e.g., "qwen3", "llama4", "nemotron").
    public let family: String

    /// Tool call format for this model family.
    public let toolCallFormat: ToolCallFormat

    /// Reasoning format (if model supports thinking).
    public let reasoningFormat: ReasoningFormat

    /// Whether the model supports vision/multimodal input.
    public let supportsVision: Bool

    /// Whether this is a hybrid SSM model.
    public let isHybrid: Bool

    /// Hybrid layer pattern (e.g., "MMM*MMM*" for Nemotron-H). nil = all attention.
    public let hybridPattern: String?

    /// Default context window size.
    public let defaultContextWindow: Int

    /// Whether the chat template natively injects <think> when thinking is enabled.
    /// When true, the model's output starts INSIDE the think block — no <think> tag
    /// appears in the generated text. The engine must treat all text before </think>
    /// as reasoning content.
    public let thinkInTemplate: Bool

    /// Whether TurboQuant is recommended for this model.
    public let recommendTQ: Bool

    /// Stop tokens specific to this model family.
    public let defaultStopTokens: [String]

    public init(
        family: String,
        toolCallFormat: ToolCallFormat = .none,
        reasoningFormat: ReasoningFormat = .none,
        supportsVision: Bool = false,
        isHybrid: Bool = false,
        hybridPattern: String? = nil,
        thinkInTemplate: Bool = false,
        defaultContextWindow: Int = 8192,
        recommendTQ: Bool = true,
        defaultStopTokens: [String] = []
    ) {
        self.family = family
        self.toolCallFormat = toolCallFormat
        self.reasoningFormat = reasoningFormat
        self.supportsVision = supportsVision
        self.isHybrid = isHybrid
        self.hybridPattern = hybridPattern
        self.thinkInTemplate = thinkInTemplate
        self.defaultContextWindow = defaultContextWindow
        self.recommendTQ = recommendTQ
        self.defaultStopTokens = defaultStopTokens
    }
}

/// Registry of model family configurations.
/// Auto-detects model family from name and returns appropriate config.
public struct ModelConfigRegistry: Sendable {

    /// All registered model families, ordered by matching priority.
    public static let configs: [ModelFamilyConfig] = [
        // Qwen family
        ModelFamilyConfig(family: "qwen3", toolCallFormat: .qwen, reasoningFormat: .qwen3,
                         thinkInTemplate: true,
                         defaultContextWindow: 32768, defaultStopTokens: ["<|endoftext|>", "<|im_end|>"]),
        ModelFamilyConfig(family: "qwen2.5-vl", toolCallFormat: .qwen, supportsVision: true,
                         defaultContextWindow: 32768, defaultStopTokens: ["<|endoftext|>", "<|im_end|>"]),
        ModelFamilyConfig(family: "qwen2.5", toolCallFormat: .qwen,
                         defaultContextWindow: 32768, defaultStopTokens: ["<|endoftext|>", "<|im_end|>"]),
        ModelFamilyConfig(family: "qwq", toolCallFormat: .qwen, reasoningFormat: .qwen3,
                         thinkInTemplate: true,
                         defaultContextWindow: 32768, defaultStopTokens: ["<|endoftext|>", "<|im_end|>"]),

        // Llama family
        ModelFamilyConfig(family: "llama-4", toolCallFormat: .llama,
                         defaultContextWindow: 131072, defaultStopTokens: ["<|eot_id|>"]),
        ModelFamilyConfig(family: "llama-3.3", toolCallFormat: .llama,
                         defaultContextWindow: 131072, defaultStopTokens: ["<|eot_id|>"]),
        ModelFamilyConfig(family: "llama-3.2-vision", toolCallFormat: .llama, supportsVision: true,
                         defaultContextWindow: 131072, defaultStopTokens: ["<|eot_id|>"]),
        ModelFamilyConfig(family: "llama-3", toolCallFormat: .llama,
                         defaultContextWindow: 8192, defaultStopTokens: ["<|eot_id|>"]),

        // Mistral/Mixtral
        ModelFamilyConfig(family: "mistral", toolCallFormat: .mistral, reasoningFormat: .mistral,
                         defaultContextWindow: 32768, defaultStopTokens: ["</s>"]),
        ModelFamilyConfig(family: "mixtral", toolCallFormat: .mistral,
                         defaultContextWindow: 32768, defaultStopTokens: ["</s>"]),
        ModelFamilyConfig(family: "codestral", toolCallFormat: .mistral,
                         defaultContextWindow: 32768, defaultStopTokens: ["</s>"]),
        ModelFamilyConfig(family: "pixtral", toolCallFormat: .mistral, supportsVision: true,
                         defaultContextWindow: 32768, defaultStopTokens: ["</s>"]),

        // DeepSeek
        ModelFamilyConfig(family: "deepseek-r1", toolCallFormat: .deepseek, reasoningFormat: .deepseekR1,
                         defaultContextWindow: 65536, defaultStopTokens: ["<|end\u{2581}of\u{2581}sentence|>"]),
        ModelFamilyConfig(family: "deepseek-v3", toolCallFormat: .deepseek,
                         defaultContextWindow: 65536, defaultStopTokens: ["<|end\u{2581}of\u{2581}sentence|>"]),
        ModelFamilyConfig(family: "deepseek-v2", toolCallFormat: .deepseek,
                         defaultContextWindow: 32768, defaultStopTokens: ["<|end\u{2581}of\u{2581}sentence|>"]),

        // Nemotron (hybrid SSM)
        ModelFamilyConfig(family: "nemotron-h", toolCallFormat: .nemotron, isHybrid: true,
                         defaultContextWindow: 32768, defaultStopTokens: ["<|endoftext|>"]),
        ModelFamilyConfig(family: "nemotron", toolCallFormat: .nemotron,
                         defaultContextWindow: 32768, defaultStopTokens: ["<|endoftext|>"]),

        // Gemma
        ModelFamilyConfig(family: "gemma-3n", toolCallFormat: .generic, supportsVision: true, isHybrid: true,
                         defaultContextWindow: 32768, defaultStopTokens: ["<end_of_turn>"]),
        ModelFamilyConfig(family: "gemma", toolCallFormat: .generic,
                         defaultContextWindow: 8192, defaultStopTokens: ["<end_of_turn>"]),

        // InternVL
        ModelFamilyConfig(family: "internvl", toolCallFormat: .generic, supportsVision: true,
                         defaultContextWindow: 32768),

        // Phi
        ModelFamilyConfig(family: "phi-4", toolCallFormat: .generic, reasoningFormat: .qwen3,
                         defaultContextWindow: 16384, defaultStopTokens: ["<|endoftext|>"]),
        ModelFamilyConfig(family: "phi-3-vision", toolCallFormat: .generic, supportsVision: true,
                         defaultContextWindow: 4096),

        // GLM
        ModelFamilyConfig(family: "glm-4", toolCallFormat: .glm, reasoningFormat: .gptoss,
                         defaultContextWindow: 32768),

        // MiniMax
        ModelFamilyConfig(family: "minimax", toolCallFormat: .minimax, reasoningFormat: .qwen3,
                         thinkInTemplate: true,
                         defaultContextWindow: 32768),

        // Jamba (hybrid SSM)
        ModelFamilyConfig(family: "jamba", toolCallFormat: .generic, isHybrid: true,
                         defaultContextWindow: 262144),

        // Hermes
        ModelFamilyConfig(family: "hermes", toolCallFormat: .hermes,
                         defaultContextWindow: 8192, defaultStopTokens: ["<|im_end|>"]),

        // Functionary
        ModelFamilyConfig(family: "functionary", toolCallFormat: .functionary,
                         defaultContextWindow: 8192),

        // Granite
        ModelFamilyConfig(family: "granite", toolCallFormat: .granite,
                         defaultContextWindow: 8192),

        // xLAM
        ModelFamilyConfig(family: "xlam", toolCallFormat: .xlam,
                         defaultContextWindow: 32768),

        // StepFun
        ModelFamilyConfig(family: "step", toolCallFormat: .stepfun,
                         defaultContextWindow: 32768),
    ]

    /// Auto-detect model family from model name.
    /// Returns nil if no matching family found (use generic defaults).
    public static func detect(modelName: String) -> ModelFamilyConfig? {
        let name = modelName.lowercased()
            .replacingOccurrences(of: "_", with: "-")
            .replacingOccurrences(of: " ", with: "-")

        for config in configs {
            if name.contains(config.family.lowercased()) {
                return config
            }
        }
        return nil
    }

    /// Get config or return a generic default.
    public static func configFor(modelName: String) -> ModelFamilyConfig {
        detect(modelName: modelName) ?? ModelFamilyConfig(
            family: "generic",
            toolCallFormat: .generic,
            defaultContextWindow: 8192
        )
    }

    /// Get the tool call format for a model.
    public static func toolFormat(for modelName: String) -> ToolCallFormat {
        configFor(modelName: modelName).toolCallFormat
    }

    /// Get the reasoning format for a model.
    public static func reasoningFormat(for modelName: String) -> ReasoningFormat {
        configFor(modelName: modelName).reasoningFormat
    }

    /// Check if a model supports vision.
    public static func supportsVision(_ modelName: String) -> Bool {
        configFor(modelName: modelName).supportsVision
    }

    /// Check if a model is hybrid (SSM + attention).
    public static func isHybrid(_ modelName: String) -> Bool {
        configFor(modelName: modelName).isHybrid
    }
}
