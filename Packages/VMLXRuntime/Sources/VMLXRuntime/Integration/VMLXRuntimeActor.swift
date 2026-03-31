import Foundation
import MLX
import MLXRandom
import MLXNN

private func _vmlxLog2(_ msg: String) {
    let line = "[\(Date())] \(msg)\n"
    let path = "/tmp/vmlx_debug.log"
    if !FileManager.default.fileExists(atPath: path) {
        FileManager.default.createFile(atPath: path, contents: nil)
    }
    if let fh = FileHandle(forWritingAtPath: path) {
        fh.seekToEndOfFile()
        fh.write(line.data(using: .utf8)!)
        fh.closeFile()
    }
}

/// Events emitted during generation.
public enum VMLXEvent: Sendable {
    case tokens(String)
    case thinking(String)
    case toolInvocation(name: String, argsJSON: String, callId: String)
    case usage(promptTokens: Int, completionTokens: Int, cachedTokens: Int)
}

// MARK: - Power Management

/// Power state for model lifecycle management.
public enum PowerState: Sendable {
    /// Model loaded, ready for inference.
    case active
    /// Caches cleared, model still in memory (reduced Metal usage).
    case softSleep
    /// Model unloaded, minimal memory.
    case deepSleep
    /// Auto-wake on next request.
    case jitWake
}

/// The central VMLXRuntime actor. Singleton that owns model loading,
/// cache coordination, scheduling, and generation.
///
/// Uses native model implementations (Qwen3.5, etc.) for the forward pass.
/// No external model library dependency -- only mlx-swift for the computation backend.
public actor VMLXRuntimeActor {

    public static let shared = VMLXRuntimeActor()

    // MARK: - State

    /// Current loaded model name.
    public private(set) var currentModelName: String?

    /// The VMLX model container (native model + tokenizer + metadata).
    private var modelContainer: VMLXModelContainer?

    /// SSM re-deriver for recovering SSM state when checkpoint is evicted.
    private var ssmReDeriver: SSMReDeriver?

    /// Whether a model is loaded and ready.
    public var isModelLoaded: Bool { modelContainer != nil }

    /// Scheduler owns request queue, cache coordinator, and batching logic.
    private var scheduler: Scheduler

    /// Active generation tasks, keyed by requestId.
    private var activeGenerations: [String: Task<Void, Never>] = [:]

    /// Last loaded model name (for wake after sleep).
    private var lastLoadedModelName: String?

    /// Last loaded model path (for wake after sleep).
    private var lastLoadedModelPath: URL?

    /// Current power state.
    public private(set) var powerState: PowerState = .deepSleep

    /// Whether JIT compilation (Metal kernel fusion) is enabled.
    public var jitEnabled: Bool = false

    // MARK: - Multi-Model Gateway

    /// Multiple loaded models, keyed by name/alias.
    private var loadedModels: [String: VMLXModelContainer] = [:]

    /// Currently active model (for single-model requests).
    public private(set) var activeModelName: String?

    // MARK: - Init

    public init(config: SchedulerConfig = .autoDetect()) {
        self.scheduler = Scheduler(config: config)
    }

    // MARK: - Runtime Configuration

    /// Apply user-facing runtime settings.
    /// Called by the host app (Osaurus) to forward UI settings to the scheduler.
    ///
    /// Settings map:
    ///   kvBits (2-8)     -> kvCacheQuantization ("q2"/"q4"/"q8" or "none")
    ///   kvGroup (1-256)  -> kvCacheGroupSize
    ///   maxKV (tokens)   -> maxNumBatchedTokens (caps context window)
    ///   prefillStep      -> prefillStepSize (tokens per prefill chunk)
    ///   topP             -> default topP for SamplingParams
    ///   enableDiskCache  -> enableDiskCache + diskCacheDir
    ///   enableTQ         -> enableTurboQuant
    public func applyUserConfig(
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        maxContextLength: Int? = nil,
        prefillStepSize: Int? = nil,
        enableDiskCache: Bool = false,
        diskCacheDir: URL? = nil,
        enableTurboQuant: Bool = false
    ) {
        if let bits = kvBits, bits < 16 {
            scheduler.config.kvCacheQuantization = "q\(bits)"
        } else {
            scheduler.config.kvCacheQuantization = "none"
        }
        scheduler.config.kvCacheGroupSize = kvGroupSize
        if let maxKV = maxContextLength {
            scheduler.config.maxNumBatchedTokens = maxKV
        }
        if let step = prefillStepSize {
            scheduler.config.prefillStepSize = step
        }
        scheduler.config.enableDiskCache = enableDiskCache
        if let dir = diskCacheDir {
            scheduler.config.diskCacheDir = dir
        } else if enableDiskCache && scheduler.config.diskCacheDir == nil {
            // Auto-configure disk cache directory based on model name
            let modelHash = currentModelName?.replacingOccurrences(of: "/", with: "_") ?? "default"
            let cacheBase = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
                ?? URL(fileURLWithPath: NSTemporaryDirectory())
            scheduler.config.diskCacheDir = cacheBase
                .appendingPathComponent("osaurus")
                .appendingPathComponent("kv-cache")
                .appendingPathComponent(modelHash)
        }
        scheduler.config.enableTurboQuant = enableTurboQuant

        // Rebuild CacheCoordinator so changed settings (disk cache, memory budget, etc.)
        // take effect. The coordinator is built from config at Scheduler.init; without
        // rebuilding, runtime config changes would be ignored.
        scheduler.rebuildCacheCoordinator()
    }

    // MARK: - Model Management

    /// Load a model from a directory path (primary method).
    ///
    /// 1. Calls `ModelLoader.load(from:)` which uses native model registry
    ///    to load weights, tokenizer, and build the correct model architecture
    /// 2. Wraps the result in a `VMLXModelContainer`
    /// 3. Configures the `Scheduler` with the model's properties
    public func loadModel(from path: URL) async throws {
        if modelContainer != nil {
            await unloadModel()
        }

        // 1. Load model using native model registry
        let loadedModel: LoadedModel
        do {
            loadedModel = try await ModelLoader.load(from: path)
        } catch {
            throw VMLXRuntimeError.modelLoadFailed(
                "Failed to load model at \(path.path): \(error.localizedDescription)"
            )
        }

        // 2. Wrap in VMLXModelContainer
        let container = VMLXModelContainer.create(model: loadedModel)

        // 3. Configure the Scheduler
        scheduler.configureForModel(
            isHybrid: container.isHybrid,
            layerPattern: container.layerPattern,
            stopTokenIds: container.eosTokenIds,
            enableTQ: container.turboQuantConfig != nil
        )

        // 4. Store state
        self.modelContainer = container
        self.currentModelName = container.name
        self.lastLoadedModelName = container.name
        self.lastLoadedModelPath = path
        self.powerState = .active

        // 4b. Wire SSM re-deriver
        if let ssmCache = scheduler.cache.ssmStateCache {
            let reDeriver = SSMReDeriver(ssmCache: ssmCache)
            self.ssmReDeriver = reDeriver
        } else {
            self.ssmReDeriver = nil
        }

        // 5. Register in multi-model gateway
        loadedModels[container.name] = container
        if activeModelName == nil {
            activeModelName = container.name
        }

        // Note: first 1-2 decode tokens after prefill are slower (~2s then ~0.4s)
        // due to MLX Metal pipeline backlog from prefill. This is normal MLX behavior
        // and happens in mlx-lm-server too. Subsequent tokens run at full speed (~25 tok/s).
    }

    /// Load a model with an optional alias for multi-model routing.
    public func loadModel(from path: URL, alias: String?) async throws {
        try await loadModel(from: path)

        if let alias = alias, let container = modelContainer {
            loadedModels.removeValue(forKey: container.name)
            loadedModels[alias] = container
            activeModelName = alias
            currentModelName = alias
        }
    }

    /// Load a model by name (convenience method).
    public func loadModel(name: String) async throws {
        let directURL = URL(fileURLWithPath: name)
        if FileManager.default.fileExists(atPath: directURL.appendingPathComponent("config.json").path) {
            try await loadModel(from: directURL)
            return
        }

        let available = ModelDetector.scanAvailableModels()
        let nameLower = name.lowercased()

        let matched = available.first(where: { $0.name.lowercased() == nameLower })
            ?? available.first(where: { $0.name.lowercased().contains(nameLower) })
            ?? available.first(where: { $0.modelPath.lastPathComponent.lowercased().contains(nameLower) })

        guard let model = matched else {
            let availableNames = available.map(\.name).joined(separator: ", ")
            throw VMLXRuntimeError.modelLoadFailed(
                "Model '\(name)' not found. Available: \(availableNames.isEmpty ? "(none)" : availableNames)"
            )
        }

        try await loadModel(from: model.modelPath)
    }

    /// Unload current model and free resources.
    public func unloadModel() async {
        for (_, task) in activeGenerations {
            task.cancel()
        }
        activeGenerations.removeAll()
        scheduler.shutdown()

        if let reDeriver = ssmReDeriver {
            await reDeriver.cancelAll()
            await reDeriver.setModel(nil)
        }
        ssmReDeriver = nil

        modelContainer = nil
        loadedModels.removeAll()
        currentModelName = nil

        // Force Metal memory cleanup before loading a new model
        Memory.clearCache()
    }

    // MARK: - Multi-Model Gateway

    public func resolveModel(_ requestedModel: String?) -> VMLXModelContainer? {
        guard let name = requestedModel else {
            return loadedModels[activeModelName ?? ""]
        }
        return loadedModels[name] ?? loadedModels.values.first {
            $0.name.lowercased().contains(name.lowercased())
        }
    }

    public var loadedModelNames: [String] {
        Array(loadedModels.keys)
    }

    public func unloadModel(name: String) async {
        loadedModels.removeValue(forKey: name)
        if activeModelName == name {
            activeModelName = loadedModels.keys.first
        }
        if currentModelName == name {
            modelContainer = nil
            currentModelName = activeModelName
            if let newActive = activeModelName {
                modelContainer = loadedModels[newActive]
            }
        }
    }

    // MARK: - Power Management

    public func softSleep() async {
        scheduler.cache.clearAll()
        powerState = .softSleep
    }

    public func deepSleep() async {
        await unloadModel()
        loadedModels.removeAll()
        activeModelName = nil
        powerState = .deepSleep
    }

    public func wake() async throws {
        guard powerState != .active else { return }
        if let path = lastLoadedModelPath {
            try await loadModel(from: path)
        } else if let name = lastLoadedModelName {
            try await loadModel(name: name)
        }
        powerState = .active
    }

    public func enableJITWake() {
        if powerState == .deepSleep {
            powerState = .jitWake
        }
    }

    public func enableJIT() {
        jitEnabled = true
    }

    // MARK: - Generation

    /// Generate a streaming response for a chat completion request.
    ///
    /// Uses native VMLXRuntime models for the forward pass.
    /// Implements autoregressive token-by-token generation with:
    /// - Chat template tokenization via swift-transformers
    /// - Greedy/sampling decoding
    /// - StreamAccumulator for tool/reasoning parsing
    /// - VMLXEvent emission for OpenAI-compatible streaming
    public func generateStream(
        request: VMLXChatCompletionRequest
    ) async throws -> AsyncThrowingStream<VMLXEvent, Error> {
        // Cancel any in-flight generation before starting a new one.
        // Metal command buffers cannot be shared across concurrent tasks.
        for (id, task) in activeGenerations {
            task.cancel()
            activeGenerations.removeValue(forKey: id)
        }

        if powerState == .jitWake {
            try await wake()
        }

        guard let container = modelContainer else {
            throw VMLXRuntimeError.noModelLoaded
        }

        let requestId = UUID().uuidString
        let modelName = currentModelName ?? ""

        // Build tool/reasoning parsers
        let toolParser: (any ToolCallParser)? = request.tools != nil
            ? autoDetectToolParser(modelName: modelName) : nil

        // Don't use reasoning parser here — Osaurus's StreamingDeltaProcessor
        // handles <think> tag parsing at the UI level. If we strip tags here,
        // the UI can't detect thinking blocks.
        let reasoningParser: (any ReasoningParser)? = nil

        // Tokenize via chat template
        let samplingParams = request.toSamplingParams()
        let enableThinking = request.enableThinking ?? true
        let tokens: [Int]
        do {
            tokens = try container.applyChatTemplate(
                messages: request.messages,
                addGenerationPrompt: true,
                enableThinking: enableThinking
            )
        } catch {
            throw VMLXRuntimeError.tokenizationFailed
        }

        // Compute gen_prompt_len: number of assistant header tokens appended by chat template.
        // Strip these from cache key so multi-turn conversations hit the same prefix.
        // e.g., "<|im_start|>assistant\n" or "<think>\n" — these change per turn.
        let genPromptLen = container.computeGenPromptLen(messages: request.messages)
        let cacheKeyTokens: [Int]
        if genPromptLen > 0 && genPromptLen < tokens.count {
            cacheKeyTokens = Array(tokens.dropLast(genPromptLen))
        } else {
            cacheKeyTokens = tokens
        }

        let promptTokenCount = tokens.count
        let maxTokens = samplingParams.maxTokens
        let temperature = samplingParams.temperature
        let topP = samplingParams.topP
        _ = samplingParams.repetitionPenalty  // TODO: implement repetition penalty
        let stopSequences = samplingParams.stop
        let eosTokenIds = container.eosTokenIds
        let stopTokenIds = Set(samplingParams.stopTokenIds)

        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    var accumulator = StreamAccumulator(
                        toolParser: toolParser,
                        reasoningParser: reasoningParser,
                        stopSequences: stopSequences
                    )

                    // Check CacheCoordinator for cached KV state from previous turns
                    let cache = container.newCache()
                    var inputTokens: MLXArray
                    var cachedTokenCount = 0

                    let fetchResult = self.scheduler.cache.fetch(tokens: cacheKeyTokens)
                    switch fetchResult {
                    case .hit(let cachedHybrid, let remaining, let detail, let ssmCheckpoint)
                        where cachedHybrid.layerCount == cache.count:
                        NSLog("[Gen] Cache HIT: \(cachedHybrid.layerCount) layers, \(remaining.count) remaining tokens, detail=\(detail)")
                        // Restore cached KV state into the VMLXKVCache objects
                        for (i, entry) in cachedHybrid.layers.enumerated() {
                            guard i < cache.count else { break }
                            switch entry {
                            case .attention(let kv):
                                if let kvSimple = cache[i] as? VMLXKVCacheSimple {
                                    kvSimple.state = [kv.keys, kv.values]
                                }
                            case .ssm(let ssm):
                                if let mambaCache = cache[i] as? VMLXMambaCache {
                                    mambaCache.state = ssm.state
                                }
                            }
                        }

                        // For hybrid models: inject SSM companion state from checkpoint.
                        // The HybridCache from memory/prefix may only contain KV layers.
                        // SSM states come separately via SSMStateCache → ssmCheckpoint.
                        if let checkpoint = ssmCheckpoint, !checkpoint.ssmStates.isEmpty {
                            var ssmIdx = 0
                            for c in cache {
                                if let mambaCache = c as? VMLXMambaCache,
                                   ssmIdx < checkpoint.ssmStates.count {
                                    mambaCache.state = checkpoint.ssmStates[ssmIdx].state
                                    ssmIdx += 1
                                }
                            }
                            NSLog("[Gen] Injected \(ssmIdx) SSM companion states from checkpoint")
                        }

                        // remaining = uncached portion of cacheKeyTokens (not full tokens).
                        // We also need to process gen_prompt_len suffix tokens.
                        let genSuffix = genPromptLen > 0
                            ? Array(tokens.suffix(genPromptLen)) : [Int]()

                        cachedTokenCount = cacheKeyTokens.count - remaining.count
                        if remaining.isEmpty {
                            // Full cache hit on cacheKeyTokens. Re-feed last cached token
                            // to get fresh logits, plus gen_prompt_len suffix.
                            for c in cache {
                                if let kvc = c as? VMLXKVCacheSimple { kvc.trim(1) }
                            }
                            cachedTokenCount -= 1
                            let refeedTokens = [cacheKeyTokens.last!] + genSuffix
                            inputTokens = MLXArray(refeedTokens.map { Int32($0) })
                        } else {
                            // Prefix hit: some cacheKeyTokens matched. Prefill remaining + suffix.
                            let allRemaining = remaining + genSuffix
                            inputTokens = MLXArray(allRemaining.map { Int32($0) })
                        }
                    case .partialHit(let attentionCache, let remaining, let detail):
                        if container.isHybrid {
                            // Hybrid model: SSM companion missing. SSM state is path-dependent —
                            // can't use KV cache without matching SSM state.
                            // Discard attention cache, full prefill.
                            // (Matches Python VMLX: "no SSM companion, full prefill")
                            NSLog("[Gen] Cache PARTIAL HIT (hybrid, SSM missing): discarding KV, full prefill \(tokens.count) tokens")
                            inputTokens = MLXArray(tokens.map { Int32($0) })
                        } else {
                            // Non-hybrid: no SSM layers, so partial hit = prefix hit.
                            // Restore attention KV and prefill only remaining + gen_prompt_len.
                            NSLog("[Gen] Cache PARTIAL HIT (non-hybrid): \(attentionCache.layerCount) layers, \(remaining.count) remaining, detail=\(detail)")
                            for (i, entry) in attentionCache.layers.enumerated() {
                                guard i < cache.count else { break }
                                if case .attention(let kv) = entry {
                                    if let kvSimple = cache[i] as? VMLXKVCacheSimple {
                                        kvSimple.state = [kv.keys, kv.values]
                                    }
                                }
                            }
                            let genSuffix = genPromptLen > 0
                                ? Array(tokens.suffix(genPromptLen)) : [Int]()
                            cachedTokenCount = cacheKeyTokens.count - remaining.count
                            if remaining.isEmpty {
                                for c in cache {
                                    if let kvc = c as? VMLXKVCacheSimple { kvc.trim(1) }
                                }
                                cachedTokenCount -= 1
                                let refeed = [cacheKeyTokens.last!] + genSuffix
                                inputTokens = MLXArray(refeed.map { Int32($0) })
                            } else {
                                let allRemaining = remaining + genSuffix
                                inputTokens = MLXArray(allRemaining.map { Int32($0) })
                            }
                        }

                    case .miss, .hit(_, _, _, _):
                        // Complete miss or layer count mismatch — full prefill
                        NSLog("[Gen] Cache MISS: prefilling \(tokens.count) tokens")
                        inputTokens = MLXArray(tokens.map { Int32($0) })
                    }

                    var generatedTokenCount = 0
                    var thinkingTokenCount = 0
                    var insideThinking = enableThinking
                    var thinkTagInjected = false
                    let thinkingBudget = maxTokens / 2  // Cap thinking at half of maxTokens

                    // Prefill: process input tokens in chunks (matches Python mlx-lm).
                    // Chunked prefill prevents OOM on long prompts and allows cache eval between chunks.
                    let prefillStep = self.scheduler.config.prefillStepSize
                    let totalPrefillTokens = inputTokens.dim(0)

                    // Process all-but-last tokens in chunks, then last token for logits
                    if totalPrefillTokens > prefillStep + 1 {
                        var processed = 0
                        let prefillEnd = totalPrefillTokens - 1
                        while processed < prefillEnd {
                            let chunkEnd = min(processed + prefillStep, prefillEnd)
                            let chunk = inputTokens[processed ..< chunkEnd]
                            _ = container.forward(chunk.expandedDimensions(axis: 0), cache: cache)
                            eval(cache)
                            Memory.clearCache()
                            processed = chunkEnd
                        }
                        inputTokens = inputTokens[(totalPrefillTokens - 1)...]
                    }

                    let prefillLogits = container.forward(
                        inputTokens.expandedDimensions(axis: 0), cache: cache)

                    var y: MLXArray
                    if temperature == 0 {
                        y = prefillLogits[0, -1].argMax()
                    } else {
                        y = MLXRandom.categorical(
                            (prefillLogits[0, -1] / temperature).expandedDimensions(axis: 0))
                    }

                    // SSM companion extraction happens in the post-generation store block below
                    // (CacheCoordinator.store automatically extracts and stores SSM layers
                    // with the correct boundary matching the KV store key).

                    // Double-buffered generation loop (matches mlx-lm Python pattern):
                    // Pipeline: build graph for NEXT token while GPU evaluates CURRENT token.
                    // asyncEval starts GPU work, then CPU does decode/yield concurrently.
                    // For MiniMax 122B: ~30ms GPU, ~10ms CPU → effective max(30,10) = 30ms vs 40ms sequential.

                    // Kick off first token eval
                    var nextY: MLXArray? = nil
                    asyncEval([y])

                    let _genStart = CFAbsoluteTimeGetCurrent()
                    for _step in 0 ..< maxTokens {
                        try Task.checkCancellation()

                        // Start computing NEXT token (GPU works on this while we process current)
                        if _step < maxTokens - 1 {
                            // Build graph for next token using current y (not yet materialized on CPU,
                            // but MLX graph can reference it). y[None] adds batch dim (1 op vs 2).
                            let stepLogits = container.forward(y.reshaped(1, 1), cache: cache)
                            if temperature == 0 {
                                nextY = stepLogits[0, -1].argMax()
                            } else {
                                nextY = MLXRandom.categorical(
                                    (stepLogits[0, -1] / temperature).expandedDimensions(axis: 0))
                            }
                            asyncEval([nextY!])
                        }

                        // Now materialize CURRENT token (blocks until GPU done with y)
                        let currentToken = y.item(Int.self)

                        if _step == 10 {
                            let _elapsed = CFAbsoluteTimeGetCurrent() - _genStart
                            _vmlxLog2("[Gen] 10 tokens in \(String(format: "%.2f", _elapsed))s = \(String(format: "%.1f", 10/_elapsed)) tok/s (includes prefill)")
                        }

                        generatedTokenCount += 1

                        // Check for EOS
                        if eosTokenIds.contains(currentToken) || stopTokenIds.contains(currentToken) {
                            break
                        }

                        // Decode token to text (CPU work overlaps with GPU computing nextY)
                        var text = container.decode([currentToken])

                        // Handle thinking state
                        if enableThinking {
                            if insideThinking {
                                thinkingTokenCount += 1
                                if text.contains("</think>") {
                                    insideThinking = false
                                } else if thinkingTokenCount >= thinkingBudget {
                                    insideThinking = false
                                    continuation.yield(.tokens("\n</think>\n"))
                                }
                            }
                        } else {
                            text = text.replacingOccurrences(of: "<think>", with: "")
                            text = text.replacingOccurrences(of: "</think>", with: "")
                            if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                y = nextY ?? y
                                continue
                            }
                        }

                        text = text.replacingOccurrences(of: "\u{FFFD}", with: "")

                        if enableThinking && !thinkTagInjected {
                            thinkTagInjected = true
                            continuation.yield(.tokens("<think>\n"))
                        }

                        let events = accumulator.process(text: text, tokenIds: [currentToken])
                        for event in events {
                            switch event {
                            case .tokens(let t):
                                continuation.yield(.tokens(t))
                            case .thinking(let t):
                                continuation.yield(.thinking(t))
                            case .toolInvocation(let name, let args, let callId):
                                continuation.yield(.toolInvocation(name: name, argsJSON: args, callId: callId))
                            case .finished:
                                break
                            }
                        }

                        // Swap: next becomes current
                        y = nextY ?? y

                        // Periodic Metal cache cleanup (matches Python's mx.clear_cache every 256)
                        if _step % 256 == 0 && _step > 0 {
                            Memory.clearCache()
                        }
                    }

                    // Finalize
                    let finalEvents = accumulator.finalize()
                    for event in finalEvents {
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

                    // Store cache for future turn reuse.
                    // Key: cacheKeyTokens (stripped of gen_prompt_len), truncated to len-1.
                    // Why len-1: on cache hit, we re-feed the last token to get fresh logits.
                    // This matches Python VMLX's _truncate_cache_to_prompt_length().
                    let storeTokens: [Int]
                    if cacheKeyTokens.count > 1 {
                        storeTokens = Array(cacheKeyTokens.dropLast(1))
                    } else {
                        storeTokens = cacheKeyTokens
                    }

                    if !storeTokens.isEmpty {
                        // Trim KV cache to match storeTokens length.
                        // Current cache offset = promptTokenCount + generatedTokenCount.
                        // We want offset = storeTokens.count = cacheKeyTokens.count - 1.
                        let targetOffset = storeTokens.count
                        for c in cache {
                            if let kvc = c as? VMLXKVCacheSimple, kvc.offset > targetOffset {
                                kvc.trim(kvc.offset - targetOffset)
                            }
                            // SSM (Mamba) state is cumulative — store as-is.
                            // SSM state at the end of prefill contains info from all prompt tokens.
                        }

                        var layers: [LayerCacheEntry] = []
                        for c in cache {
                            if let mc = c as? VMLXMambaCache {
                                layers.append(.ssm(SSMStateLayer(state: mc.state)))
                            } else if let kvc = c as? VMLXKVCacheSimple {
                                let s = kvc.state
                                if s.count == 2 {
                                    layers.append(.attention(KVCacheLayer(
                                        keys: s[0], values: s[1], offset: kvc.offset)))
                                }
                            }
                        }
                        if !layers.isEmpty {
                            let hybridCache = HybridCache(layers: layers)
                            hybridCache.materialized()
                            self.scheduler.cache.store(tokens: storeTokens, cache: hybridCache)
                            NSLog("[Gen] Stored cache: \(storeTokens.count) tokens (stripped \(genPromptLen) gen_prompt + \(generatedTokenCount) generated + 1 last)")
                        }
                    }

                    // Emit usage
                    continuation.yield(.usage(
                        promptTokens: promptTokenCount,
                        completionTokens: generatedTokenCount,
                        cachedTokens: cachedTokenCount
                    ))

                    continuation.finish()

                } catch is CancellationError {
                    // Generation was stopped — don't store partial cache
                    // Clear any stale cache that might have wrong shapes
                    self.scheduler.cache.clearAll()
                    continuation.finish()
                } catch {
                    self.scheduler.cache.clearAll()
                    continuation.finish(throwing: error)
                }
            }

            Task { @MainActor [requestId] in
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
        let stream = try await generateStream(request: request)
        for try await event in stream {
            if case .tokens(let text) = event {
                result += text
            }
        }
        return result
    }

    // MARK: - Cache Management

    public func clearCache() {
        scheduler.cache.clearAll()
    }

    public var cacheStats: CacheCoordinatorStats {
        scheduler.cacheStats
    }

    public var config: SchedulerConfig { scheduler.config }

    public var container: VMLXModelContainer? { modelContainer }

    // MARK: - Private

    private func _storeCache(tokens: [Int], cache: HybridCache) {
        scheduler.cache.store(tokens: tokens, cache: cache)
    }

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
