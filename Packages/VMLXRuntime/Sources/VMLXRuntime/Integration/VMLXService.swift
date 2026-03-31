import Foundation

/// Protocol matching Osaurus's ModelService interface.
/// VMLXService conforms to this; the OsaurusCore integration layer bridges to the actual protocol.
public protocol VMLXModelService: Sendable {
    var serviceId: String { get }
    func isAvailable() -> Bool
    func handles(requestedModel: String?) -> Bool

    func generateOneShot(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        requestedModel: String?
    ) async throws -> String

    func streamDeltas(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        requestedModel: String?,
        stopSequences: [String]
    ) async throws -> AsyncThrowingStream<String, Error>
}

/// Protocol for tool-capable services (extends VMLXModelService).
public protocol VMLXToolCapableService: VMLXModelService {
    func respondWithTools(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        stopSequences: [String],
        tools: [VMLXToolDefinition],
        toolChoice: String?,
        requestedModel: String?
    ) async throws -> String

    func streamWithTools(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        stopSequences: [String],
        tools: [VMLXToolDefinition],
        toolChoice: String?,
        requestedModel: String?
    ) async throws -> AsyncThrowingStream<String, Error>
}

/// The main VMLXService — drop-in replacement for Osaurus's MLXService.
/// Delegates to VMLXRuntimeActor for all inference operations.
public actor VMLXService: VMLXToolCapableService {

    public static let shared = VMLXService()

    public nonisolated var serviceId: String { "vmlx" }

    private let runtime: VMLXRuntimeActor

    public init(runtime: VMLXRuntimeActor = .shared) {
        self.runtime = runtime
    }

    // MARK: - VMLXModelService

    public nonisolated func isAvailable() -> Bool {
        // Available if the runtime exists (model may or may not be loaded)
        true
    }

    public nonisolated func handles(requestedModel: String?) -> Bool {
        guard let model = requestedModel else { return true }
        let lower = model.lowercased()
        if lower.isEmpty || lower == "local" || lower == "default" || lower == "vmlx" {
            return true
        }
        // Match all model families supported by VMLXRuntime's StandardTransformerModel,
        // Qwen35Model, and GPTOSSModel. Unsupported types will fail at load time
        // and the ChatEngine fallback router retries with MLXService.
        let families = [
            "jang", "mlx",      // Quantization frameworks
            "llama", "qwen", "qwq",  // Meta, Alibaba
            "mistral", "mixtral", "codestral", "pixtral",  // Mistral AI
            "gemma",            // Google
            "phi",              // Microsoft
            "granite",          // IBM
            "deepseek",         // DeepSeek
            "minimax",          // MiniMax
            "gpt-oss", "gpt_oss",  // GPT-OSS (OpenAI)
            "glm",              // Zhipu AI
            "nemotron",         // NVIDIA
            "internlm", "internvl",  // Shanghai AI Lab
            "cohere", "command",  // Cohere
            "exaone",           // LG
            "olmo",             // AI2
            "starcoder",        // BigCode
            "stablelm",         // Stability AI
            "hermes",           // NousResearch
            "functionary",      // MeetKai
            "xlam",             // Salesforce
            "jamba",            // AI21
        ]
        return families.contains { lower.contains($0) }
    }

    public func generateOneShot(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        requestedModel: String?
    ) async throws -> String {
        let request = VMLXChatCompletionRequest(
            messages: messages,
            model: requestedModel,
            temperature: params.temperature,
            maxTokens: params.maxTokens,
            topP: params.topP,
            repetitionPenalty: params.repetitionPenalty,
            stop: params.stop,
            stream: false,
            enableThinking: params.enableThinking
        )
        return try await runtime.generate(request: request)
    }

    public func streamDeltas(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        requestedModel: String?,
        stopSequences: [String]
    ) async throws -> AsyncThrowingStream<String, Error> {
        let request = VMLXChatCompletionRequest(
            messages: messages,
            model: requestedModel,
            temperature: params.temperature,
            maxTokens: params.maxTokens,
            topP: params.topP,
            repetitionPenalty: params.repetitionPenalty,
            stop: stopSequences,
            stream: true,
            enableThinking: params.enableThinking,
            reasoningEffort: params.reasoningEffort
        )

        let eventStream = try await runtime.generateStream(request: request)

        // Transform VMLXEvent stream into String delta stream
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    for try await event in eventStream {
                        switch event {
                        case .tokens(let text):
                            continuation.yield(text)
                        case .thinking(let text):
                            // Pass thinking content as raw text with think tags.
                            // Osaurus's StreamingDeltaProcessor handles <think> parsing at the UI level.
                            continuation.yield(text)
                        case .toolInvocation(let name, let args, _):
                            continuation.yield("\u{FFFE}tool:" + name)
                            continuation.yield("\u{FFFE}args:" + args)
                        case .usage(let prompt, let completion, let cached,
                                    let ttft, let ppTPS, let decTPS, let detail):
                            // Encode stats as sentinel-prefixed JSON for the bridge to parse
                            let statsJSON = "{\"p\":\(prompt),\"c\":\(completion),\"k\":\(cached),"
                                + "\"ttft\":\(String(format:"%.3f",ttft)),"
                                + "\"pp\":\(String(format:"%.1f",ppTPS)),"
                                + "\"tg\":\(String(format:"%.1f",decTPS)),"
                                + "\"d\":\"\(detail ?? "miss")\"}"
                            continuation.yield("\u{FFFE}stats:" + statsJSON)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - VMLXToolCapableService

    public func respondWithTools(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        stopSequences: [String],
        tools: [VMLXToolDefinition],
        toolChoice: String?,
        requestedModel: String?
    ) async throws -> String {
        let request = VMLXChatCompletionRequest(
            messages: messages,
            model: requestedModel,
            temperature: params.temperature,
            maxTokens: params.maxTokens,
            topP: params.topP,
            repetitionPenalty: params.repetitionPenalty,
            stop: stopSequences,
            stream: false,
            tools: tools,
            toolChoice: toolChoice,
            enableThinking: params.enableThinking
        )
        return try await runtime.generate(request: request)
    }

    public func streamWithTools(
        messages: [VMLXChatMessage],
        params: SamplingParams,
        stopSequences: [String],
        tools: [VMLXToolDefinition],
        toolChoice: String?,
        requestedModel: String?
    ) async throws -> AsyncThrowingStream<String, Error> {
        let request = VMLXChatCompletionRequest(
            messages: messages,
            model: requestedModel,
            temperature: params.temperature,
            maxTokens: params.maxTokens,
            topP: params.topP,
            repetitionPenalty: params.repetitionPenalty,
            stop: stopSequences,
            stream: true,
            tools: tools,
            toolChoice: toolChoice,
            enableThinking: params.enableThinking
        )

        let eventStream = try await runtime.generateStream(request: request)

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    for try await event in eventStream {
                        switch event {
                        case .tokens(let text):
                            continuation.yield(text)
                        case .thinking(let text):
                            continuation.yield(text)
                        case .toolInvocation(let name, let args, _):
                            continuation.yield("\u{FFFE}tool:" + name)
                            continuation.yield("\u{FFFE}args:" + args)
                        case .usage(let prompt, let completion, let cached,
                                    let ttft, let ppTPS, let decTPS, let detail):
                            let statsJSON = "{\"p\":\(prompt),\"c\":\(completion),\"k\":\(cached),"
                                + "\"ttft\":\(String(format:"%.3f",ttft)),"
                                + "\"pp\":\(String(format:"%.1f",ppTPS)),"
                                + "\"tg\":\(String(format:"%.1f",decTPS)),"
                                + "\"d\":\"\(detail ?? "miss")\"}"
                            continuation.yield("\u{FFFE}stats:" + statsJSON)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Runtime Configuration Passthrough

    /// Forward user-facing runtime settings to VMLXRuntimeActor.
    public func applyUserConfig(
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        maxContextLength: Int? = nil,
        prefillStepSize: Int? = nil,
        enableDiskCache: Bool = false,
        diskCacheDir: URL? = nil,
        enableTurboQuant: Bool = false,
        cacheMemoryPercent: Float? = nil,
        usePagedCache: Bool? = nil
    ) async {
        await runtime.applyUserConfig(
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            maxContextLength: maxContextLength,
            prefillStepSize: prefillStepSize,
            enableDiskCache: enableDiskCache,
            diskCacheDir: diskCacheDir,
            enableTurboQuant: enableTurboQuant,
            cacheMemoryPercent: cacheMemoryPercent,
            usePagedCache: usePagedCache
        )
    }

    // MARK: - Model Management Passthrough

    /// Load a model from a directory path.
    public func loadModel(from path: URL) async throws {
        try await runtime.loadModel(from: path)
    }

    /// Load a model by name (scans well-known directories to resolve).
    public func loadModel(name: String) async throws {
        try await runtime.loadModel(name: name)
    }

    public func unloadModel() async {
        await runtime.unloadModel()
    }

    public var currentModelName: String? {
        get async { await runtime.currentModelName }
    }

    public var isModelLoaded: Bool {
        get async { await runtime.isModelLoaded }
    }
}
