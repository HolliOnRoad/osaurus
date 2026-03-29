import Foundation
import MLX

/// Events emitted during generation.
public enum VMLXEvent: Sendable {
    case tokens(String)
    case thinking(String)
    case toolInvocation(name: String, argsJSON: String, callId: String)
    case usage(promptTokens: Int, completionTokens: Int, cachedTokens: Int)
}

/// The central VMLXRuntime actor. Singleton that owns model loading,
/// cache coordination, scheduling, and generation.
/// Replaces Osaurus's ModelRuntime.
public actor VMLXRuntimeActor {

    public static let shared = VMLXRuntimeActor()

    // MARK: - State

    /// Current loaded model name.
    public private(set) var currentModelName: String?

    /// Whether a model is loaded and ready.
    public var isModelLoaded: Bool { currentModelName != nil }

    /// Scheduler configuration.
    private var schedulerConfig: SchedulerConfig

    /// Cache coordinator (manages all 5 cache layers).
    private var cacheCoordinator: CacheCoordinator

    /// Request queue for continuous batching.
    private var requestQueue: RequestQueue

    /// Active generation tasks, keyed by requestId.
    private var activeGenerations: [String: Task<Void, Never>] = [:]

    /// Whether the current model is hybrid (has SSM layers).
    private var isHybrid: Bool = false

    /// TurboQuant configuration (nil if TQ not enabled).
    private var turboQuantConfig: TurboQuantConfig?

    // MARK: - Init

    public init(config: SchedulerConfig = .autoDetect()) {
        self.schedulerConfig = config
        self.cacheCoordinator = CacheCoordinator(config: config.toCacheCoordinatorConfig())
        self.requestQueue = RequestQueue()
    }

    // MARK: - Model Management

    /// Load a model by name/path. Configures cache and TQ based on model properties.
    public func loadModel(name: String, isHybrid: Bool = false, turboQuant: TurboQuantConfig? = nil) async throws {
        // Unload previous model if any
        if currentModelName != nil {
            await unloadModel()
        }

        currentModelName = name
        self.isHybrid = isHybrid
        self.turboQuantConfig = turboQuant
        cacheCoordinator.setHybrid(isHybrid)

        // TODO: Actual model loading via MLX will go here
        // - Load safetensors weights
        // - Set up tokenizer
        // - Detect JANG format
        // - Detect hybrid pattern
        // - Configure TQ
    }

    /// Unload current model and free resources.
    public func unloadModel() async {
        // Cancel all active generations
        for (_, task) in activeGenerations {
            task.cancel()
        }
        activeGenerations.removeAll()

        // Clear caches
        cacheCoordinator.clearAll()

        currentModelName = nil
        isHybrid = false
        turboQuantConfig = nil
    }

    // MARK: - Generation

    /// Generate a streaming response for a chat completion request.
    /// Returns an AsyncThrowingStream of VMLXEvents.
    public func generateStream(
        request: VMLXChatCompletionRequest
    ) throws -> AsyncThrowingStream<VMLXEvent, Error> {
        guard isModelLoaded else {
            throw VMLXRuntimeError.noModelLoaded
        }

        let requestId = UUID().uuidString
        let samplingParams = request.toSamplingParams()
        let modelName = currentModelName ?? ""

        // Capture cache result synchronously while still on actor
        // TODO: Tokenize messages using loaded tokenizer
        let promptTokenIds: [Int] = []  // Placeholder
        let cacheResult = cacheCoordinator.fetch(tokens: promptTokenIds)

        // Build tool/reasoning parsers
        let toolParser: (any ToolCallParser)? = request.tools != nil
            ? autoDetectToolParser(modelName: modelName) : nil
        let reasoningParser: (any ReasoningParser)? = (request.enableThinking ?? false)
            ? autoDetectReasoningParser(modelName: modelName) : nil

        return AsyncThrowingStream { continuation in
            let task = Task { [cacheResult] in
                do {
                    var inferenceRequest = InferenceRequest(
                        requestId: requestId,
                        promptTokenIds: promptTokenIds,
                        samplingParams: samplingParams,
                        enableThinking: request.enableThinking ?? false,
                        reasoningEffort: request.reasoningEffort ?? "medium",
                        isMultimodal: request.isMultimodal
                    )

                    // Apply cache result
                    switch cacheResult {
                    case .hit(let cache, let remaining, _):
                        inferenceRequest.promptCache = cache
                        inferenceRequest.remainingTokenIds = remaining
                        inferenceRequest.cachedTokens = promptTokenIds.count - remaining.count

                    case .partialHit(let attentionCache, let remaining, _):
                        // Hybrid model: have KV but not SSM
                        // TODO: Trigger SSM re-derive or fall back to full prefill
                        inferenceRequest.promptCache = attentionCache
                        inferenceRequest.remainingTokenIds = remaining

                    case .miss:
                        inferenceRequest.remainingTokenIds = promptTokenIds
                    }

                    // Set up stream accumulator
                    var accumulator = StreamAccumulator(
                        toolParser: toolParser,
                        reasoningParser: reasoningParser,
                        stopSequences: samplingParams.stop
                    )

                    // TODO: Actual generation loop
                    // 1. Prefill uncached tokens through model
                    // 2. Decode loop: sample token, process through accumulator
                    // 3. Emit events via continuation
                    // 4. Store cache state after generation

                    // Finalize and emit remaining events
                    let events = accumulator.finalize()
                    for event in events {
                        switch event {
                        case .tokens(let text):
                            continuation.yield(.tokens(text))
                        case .thinking(let text):
                            continuation.yield(.thinking(text))
                        case .toolInvocation(let name, let args, let callId):
                            continuation.yield(.toolInvocation(name: name, argsJSON: args, callId: callId))
                        case .finished:
                            break
                        }
                    }

                    // Emit usage
                    continuation.yield(.usage(
                        promptTokens: promptTokenIds.count,
                        completionTokens: accumulator.generatedTokenIds.count,
                        cachedTokens: inferenceRequest.cachedTokens
                    ))

                    continuation.finish()

                } catch {
                    continuation.finish(throwing: error)
                }
            }

            // Track active generation
            Task { [requestId] in
                await self._trackGeneration(requestId: requestId, task: task)
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    /// Non-streaming generation. Collects all output into a single string.
    public func generate(request: VMLXChatCompletionRequest) async throws -> String {
        var result = ""
        let stream = try generateStream(request: request)
        for try await event in stream {
            if case .tokens(let text) = event {
                result += text
            }
        }
        return result
    }

    // MARK: - Cache Management

    /// Clear all caches.
    public func clearCache() {
        cacheCoordinator.clearAll()
    }

    /// Get cache statistics.
    public var cacheStats: CacheCoordinatorStats {
        cacheCoordinator.stats
    }

    /// Get scheduler config.
    public var config: SchedulerConfig { schedulerConfig }

    // MARK: - Private

    private func _trackGeneration(requestId: String, task: Task<Void, Never>) {
        activeGenerations[requestId] = task
        Task {
            await task.value
            activeGenerations.removeValue(forKey: requestId)
        }
    }
}

// MARK: - Errors

public enum VMLXRuntimeError: Error, LocalizedError, Sendable {
    case noModelLoaded
    case modelLoadFailed(String)
    case generationFailed(String)
    case cacheCorruption(String)
    case tokenizationFailed

    public var errorDescription: String? {
        switch self {
        case .noModelLoaded: return "No model loaded"
        case .modelLoadFailed(let msg): return "Model load failed: \(msg)"
        case .generationFailed(let msg): return "Generation failed: \(msg)"
        case .cacheCorruption(let msg): return "Cache corruption: \(msg)"
        case .tokenizationFailed: return "Tokenization failed"
        }
    }
}
