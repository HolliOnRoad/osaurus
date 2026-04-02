import Foundation
import Cmlx
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

// MARK: - String Helpers

private extension String {
    /// Strip junk characters that cause visual artifacts in the UI:
    /// replacement chars (U+FFFD), zero-width spaces/joiners, byte order marks,
    /// and C0 control chars (except newline, tab, carriage return).
    var vmlxStripped: String {
        // Fast path: most tokens have no junk
        let hasJunk = self.unicodeScalars.contains { s in
            s.value == 0xFFFD || s.value == 0xFEFF  // replacement, BOM
                || s.value == 0x200B || s.value == 0x200C || s.value == 0x200D  // zero-width
                || s.value == 0x2060 || s.value == 0xFFF9  // word joiner, interlinear
                || (s.value < 0x20 && s.value != 0x0A && s.value != 0x0D && s.value != 0x09)
        }
        guard hasJunk else { return self }
        return String(self.unicodeScalars.filter { s in
            !(s.value == 0xFFFD || s.value == 0xFEFF
              || s.value == 0x200B || s.value == 0x200C || s.value == 0x200D
              || s.value == 0x2060 || s.value == 0xFFF9
              || (s.value < 0x20 && s.value != 0x0A && s.value != 0x0D && s.value != 0x09))
        })
    }
}

/// Events emitted during generation.
public enum VMLXEvent: Sendable {
    case tokens(String)
    case thinking(String)
    case toolInvocation(name: String, argsJSON: String, callId: String)
    case usage(promptTokens: Int, completionTokens: Int, cachedTokens: Int,
               ttft: Double, prefillToksPerSec: Double, decodeToksPerSec: Double,
               cacheDetail: String?, cacheBytes: Int)
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

    /// The loaded model's family config (reasoning format, tool format, etc.).
    /// Returns nil if no model is loaded.
    public var loadedFamilyConfig: ModelFamilyConfig? {
        modelContainer?.familyConfig
    }

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

    private func estimateLiveCacheBytes(_ cache: [any VMLXKVCache]) -> Int {
        cache.reduce(into: 0) { total, layer in
            total += layer.estimatedBytes
        }
    }

    private func adaptivePrefillStep(
        for container: VMLXModelContainer,
        configuredStep: Int,
        totalPrefillTokens: Int
    ) -> Int {
        guard totalPrefillTokens > 1 else { return configuredStep }

        let detectedExperts = container.model.detected.numExperts ?? 0
        // Only throttle for very large MoE (512+ experts like MiniMax-M2.5).
        // Qwen3.5 has 256 experts but only activates 4 per token (3B active),
        // so it runs fine at full prefill step. Python VMLX uses 2048 chunks.
        let isVeryLargeMoE = detectedExperts >= 512
        guard isVeryLargeMoE else { return configuredStep }

        // For 512+ expert models, cap prefill to avoid OOM from expert routing
        let cap: Int
        if totalPrefillTokens <= 256 {
            cap = 64
        } else if totalPrefillTokens <= 1024 {
            cap = 128
        } else {
            cap = 256
        }

        return min(configuredStep, cap)
    }

    // MARK: - Init

    public init(config: SchedulerConfig = .autoDetect()) {
        self.scheduler = Scheduler(config: config)
    }

    // MARK: - Runtime Configuration

    /// Fingerprint of the last applied cache-related config.
    /// Used to avoid rebuilding CacheCoordinator (which clears all cached KV data)
    /// when nothing changed. Without this, every request would wipe the cache.
    private var lastCacheConfigFingerprint: String = ""

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
        enableTurboQuant: Bool = false,
        cacheMemoryPercent: Float? = nil,
        usePagedCache: Bool? = nil
    ) -> Bool {
        let kvQuantStr: String
        if let bits = kvBits, bits < 16 {
            kvQuantStr = "q\(bits)"
        } else {
            kvQuantStr = "none"
        }

        // Build fingerprint of cache-affecting settings BEFORE applying.
        // Only rebuild CacheCoordinator if these actually changed.
        let resolvedDiskDir: URL?
        if let dir = diskCacheDir {
            resolvedDiskDir = dir
        } else if enableDiskCache && scheduler.config.diskCacheDir == nil {
            let modelHash = currentModelName?.replacingOccurrences(of: "/", with: "_") ?? "default"
            let cacheBase = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
                ?? URL(fileURLWithPath: NSTemporaryDirectory())
            resolvedDiskDir = cacheBase
                .appendingPathComponent("osaurus")
                .appendingPathComponent("kv-cache")
                .appendingPathComponent(modelHash)
        } else {
            resolvedDiskDir = scheduler.config.diskCacheDir
        }

        let fingerprint = "\(enableDiskCache)|\(resolvedDiskDir?.path ?? "nil")|\(enableTurboQuant)|\(cacheMemoryPercent ?? scheduler.config.cacheMemoryPercent)|\(usePagedCache ?? scheduler.config.usePagedCache)"

        // Apply all settings to scheduler config
        scheduler.config.kvCacheQuantization = kvQuantStr
        scheduler.config.kvCacheGroupSize = kvGroupSize
        if let maxKV = maxContextLength {
            scheduler.config.maxNumBatchedTokens = maxKV
        }
        if let step = prefillStepSize {
            scheduler.config.prefillStepSize = step
        }
        scheduler.config.enableDiskCache = enableDiskCache
        if let dir = resolvedDiskDir {
            scheduler.config.diskCacheDir = dir
        }
        scheduler.config.enableTurboQuant = enableTurboQuant
        if let memPercent = cacheMemoryPercent {
            scheduler.config.cacheMemoryPercent = memPercent
        }
        if let paged = usePagedCache {
            scheduler.config.usePagedCache = paged
        }

        // Only rebuild CacheCoordinator when cache-affecting settings actually changed.
        // Rebuilding creates a new coordinator which CLEARS all cached KV data —
        // doing this on every request would make multi-turn cache hits impossible.
        if fingerprint != lastCacheConfigFingerprint {
            scheduler.rebuildCacheCoordinator()
            lastCacheConfigFingerprint = fingerprint
            NSLog("[VMLX] CacheCoordinator rebuilt (config changed)")
            return true
        }
        return false
    }

    // MARK: - Model Management

    /// Load a model from a directory path (primary method).
    ///
    /// 1. Calls `ModelLoader.load(from:)` which uses native model registry
    ///    to load weights, tokenizer, and build the correct model architecture
    /// 2. Wraps the result in a `VMLXModelContainer`
    /// 3. Configures the `Scheduler` with the model's properties
    public func loadModel(from path: URL) async throws {
        _vmlxLog2("[LoadModel] START path=\(path.lastPathComponent)")
        if modelContainer != nil {
            _vmlxLog2("[LoadModel] Unloading previous model")
            await unloadModel()
        }

        // 1. Load model using native model registry
        _vmlxLog2("[LoadModel] Step 1: ModelLoader.load")
        let loadedModel: LoadedModel
        do {
            loadedModel = try await ModelLoader.load(from: path)
            _vmlxLog2("[LoadModel] Step 1 OK: \(loadedModel.detected.name), type=\(loadedModel.detected.modelType ?? "nil")")
        } catch {
            _vmlxLog2("[LoadModel] Step 1 FAILED: \(error)")
            throw VMLXRuntimeError.modelLoadFailed(
                "Failed to load model at \(path.path): \(error.localizedDescription)"
            )
        }

        // 2. Wrap in VMLXModelContainer
        _vmlxLog2("[LoadModel] Step 2: ModelContainer.create")
        let container = VMLXModelContainer.create(model: loadedModel)
        _vmlxLog2("[LoadModel] Step 2 OK: hybrid=\(container.isHybrid), tq=\(container.turboQuantConfig != nil), pattern=\(container.layerPattern?.prefix(5).map { "\($0)" } ?? ["nil"])")

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

        // 4b. Wire SSM re-deriver for hybrid models with SSM layers.
        // The re-deriver runs full forward pass to extract SSM state when the
        // SSM companion cache entry has been evicted. Uses VMLXModelContainer
        // directly for the forward pass.
        // Primary path: CacheCoordinator.store() saves SSM companion after generation.
        // Re-deriver path: targeted re-derivation when SSM evicted (partialHit).
        if container.isHybrid, let ssmCache = scheduler.cache.ssmStateCache {
            let reDeriver = SSMReDeriver(ssmCache: ssmCache)
            await reDeriver.setModel(container)
            self.ssmReDeriver = reDeriver
        } else {
            self.ssmReDeriver = nil
        }

        // 5. Register in multi-model gateway
        loadedModels[container.name] = container
        if activeModelName == nil {
            activeModelName = container.name
        }

        // 6. Pin model weights to wired GPU-accessible memory.
        // Without this, macOS may page model buffers to SSD between tokens,
        // adding 10-15ms latency per token on large models.
        // Matches Python VMLX: mx.set_wired_limit(max_recommended_working_set_size)
        let maxWired = GPU.maxRecommendedWorkingSetBytes()
        _vmlxLog2("[LoadModel] GPU maxRecommendedWorkingSetBytes = \(maxWired.map { "\($0 / (1024*1024))MB" } ?? "nil")")
        let wiredBytes = maxWired ?? Int(ProcessInfo.processInfo.physicalMemory * 3 / 4)
        if wiredBytes > 0 {
            var previous: size_t = 0
            mlx_set_wired_limit(&previous, size_t(wiredBytes))
            _vmlxLog2("[LoadModel] Wired memory limit set to \(wiredBytes / (1024*1024))MB (previous: \(previous / (1024*1024))MB)")
        }

        // 7. Set Metal allocator cache limit to 25% of max working set (floor 512MB).
        let cacheLimit = max(512 * 1024 * 1024, wiredBytes / 4)
        Memory.cacheLimit = cacheLimit
        _vmlxLog2("[LoadModel] Metal cache limit set to \(cacheLimit / (1024*1024))MB")
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
        // Cancel all active generations first
        for (_, task) in activeGenerations {
            task.cancel()
        }
        // Wait for tasks to actually stop (GPU must finish current op)
        // Use withTaskGroup to add a timeout so we don't hang forever
        await withTaskGroup(of: Void.self) { group in
            for (_, task) in activeGenerations {
                group.addTask { await task.value }
            }
            // Give tasks up to 2 seconds to finish
            group.addTask {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
            // Return after first completion (either all tasks done or timeout)
            await group.next()
            group.cancelAll()
        }
        activeGenerations.removeAll()
        scheduler.shutdown()
        scheduler.cache.clearAll()

        if let reDeriver = ssmReDeriver {
            await reDeriver.cancelAll()
            await reDeriver.setModel(nil)
        }
        ssmReDeriver = nil

        modelContainer = nil
        loadedModels.removeAll()
        activeModelName = nil
        currentModelName = nil
        powerState = .deepSleep

        // Ensure all in-flight Metal commands complete before releasing memory.
        // Without this, MLXService's background prefix-cache tasks or other GPU work
        // could still be referencing buffers we're about to free.
        Stream.gpu.synchronize()
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
        // Cancel any in-flight generation and WAIT for GPU to finish.
        // Without waiting, the old task's MLX operations and the new task's
        // run concurrently, corrupting MLX's CustomKernelCache (unordered_map
        // race → EXC_BAD_ACCESS in CustomKernel::eval_gpu).
        if !activeGenerations.isEmpty {
            for (_, task) in activeGenerations {
                task.cancel()
            }
            // Wait for tasks to actually stop (GPU must finish current op).
            // Timeout after 3s to avoid hanging if a task is stuck.
            await withTaskGroup(of: Void.self) { group in
                for (_, task) in activeGenerations {
                    group.addTask { await task.value }
                }
                group.addTask {
                    try? await Task.sleep(nanoseconds: 3_000_000_000)
                }
                await group.next()
                group.cancelAll()
            }
            activeGenerations.removeAll()
        }

        if powerState == .jitWake {
            try await wake()
        }

        guard let container = modelContainer else {
            throw VMLXRuntimeError.noModelLoaded
        }

        let requestId = UUID().uuidString

        // Build tool parser: per-model override → auto-detect from model_type.
        // Per-model override flows from ModelOptionsStore → GenerationParameters →
        // SamplingParams → VMLXChatCompletionRequest.toolParserOverride.
        let toolParser: (any ToolCallParser)? = {
            guard request.tools != nil else { return nil }
            if let override = request.toolParserOverride,
               let format = ToolCallFormat(rawValue: override) {
                return toolParserForFormat(format)
            }
            // Fall back to auto-detect from model_type config
            return toolParserForFormat(container.familyConfig.toolCallFormat)
        }()

        // Build reasoning parser: per-model override → family config.
        // VMLXService wraps `.thinking` events back into `<think>` markers so
        // Osaurus's StreamingDeltaProcessor can keep using the same UI path.
        let reasoningParser: (any ReasoningParser)? = {
            if let override = request.reasoningParserOverride {
                switch override {
                case "think":
                    return reasoningParserForFormat(.qwen3)
                case "gptoss":
                    return reasoningParserForFormat(.gptoss)
                case "mistral":
                    return reasoningParserForFormat(.mistral)
                case "none":
                    return nil
                default:
                    break
                }
            }
            return reasoningParserForFormat(container.familyConfig.reasoningFormat)
        }()

        // Tokenize via chat template
        let samplingParams = request.toSamplingParams()
        let enableThinking = request.enableThinking ?? true
        let tokens: [Int]
        do {
            tokens = try container.applyChatTemplate(
                messages: request.messages,
                addGenerationPrompt: true,
                enableThinking: enableThinking,
                reasoningEffort: request.reasoningEffort
            )
        } catch {
            throw VMLXRuntimeError.tokenizationFailed
        }

        // Think tags are handled at the UI level (StreamingDeltaProcessor + middleware)

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
        let stopSequences = samplingParams.stop
        let eosTokenIds = container.eosTokenIds
        let stopTokenIds = Set(samplingParams.stopTokenIds)

        // Capture the generation Task so we can register it AFTER
        // the AsyncThrowingStream init closure returns (still on actor).
        var generationTask: Task<Void, Never>?

        let stream = AsyncThrowingStream<VMLXEvent, Error> { continuation in
            let task = Task { @Sendable in
                let pagedWriteSession = self.scheduler.cache.beginPagedWriteSession(
                    requestId: requestId
                )
                do {
                    let requestStart = CFAbsoluteTimeGetCurrent()
                    var ttftTime: Double = 0
                    var firstTokenEmitted = false
                    var cacheDetailStr: String? = nil

                    var accumulator = StreamAccumulator(
                        toolParser: toolParser,
                        reasoningParser: reasoningParser,
                        stopSequences: stopSequences
                    )

                    // Check CacheCoordinator for cached KV state from previous turns.
                    // Apply KV quantization (q4/q8) if configured — reduces GPU memory
                    // during inference. Separate from TurboQuant (post-prefill compression).
                    let kvBitsStr = self.scheduler.config.kvCacheQuantization
                    let kvBits: Int? = kvBitsStr.hasPrefix("q") ? Int(kvBitsStr.dropFirst()) : nil
                    let kvGroupSize = self.scheduler.config.kvCacheGroupSize
                    let tqConfig = self.scheduler.config.enableTurboQuant ? container.turboQuantConfig : nil
                    let tqEnabled = tqConfig != nil
                    _vmlxLog2("[Gen] Config: kvQuant=\(kvBitsStr) tq=\(tqEnabled) hybrid=\(container.isHybrid) thinking=\(enableThinking)")
                    let cache = container.newCache(config: VMLXLiveCacheConfig(
                        kvBits: kvBits,
                        kvGroupSize: kvGroupSize,
                        turboQuantConfig: tqConfig
                    ))
                    var inputTokens = MLXArray()
                    var cachedTokenCount = 0

                    func _makeTQState(
                        encodedKeys: EncodedKeys,
                        encodedValues: EncodedValues
                    ) -> TurboQuantEncoder.EncoderState {
                        let dim = encodedKeys.shape.last ?? 128
                        let keyBits = encodedKeys.indexBits + 1
                        return TurboQuantEncoder.EncoderState(
                            dim: dim,
                            keyBits: keyBits,
                            valueBits: encodedValues.indexBits,
                            seed: encodedKeys.seed
                        )
                    }

                    func _makeTQState(for cachedHybrid: HybridCache) -> TurboQuantEncoder.EncoderState? {
                        guard
                            let firstTQ = cachedHybrid.layers.first(where: {
                                if case .compressedAttention = $0 { return true }
                                return false
                            }),
                            case .compressedAttention(let ek, let ev, _) = firstTQ
                        else {
                            return nil
                        }
                        return _makeTQState(encodedKeys: ek, encodedValues: ev)
                    }

                    @discardableResult
                    func _restoreCachedHybrid(_ cachedHybrid: HybridCache) -> Int {
                        let tqState = _makeTQState(for: cachedHybrid)
                        let restoreOptions = VMLXCacheRestoreOptions(turboQuantState: tqState)
                        var restoredLayers = 0

                        for (i, entry) in cachedHybrid.layers.enumerated() {
                            guard i < cache.count else { break }
                            if cache[i].restore(from: entry, options: restoreOptions) {
                                restoredLayers += 1
                            }
                        }

                        return restoredLayers
                    }

                    @discardableResult
                    func _injectSSMCheckpoint(_ checkpoint: SSMCheckpoint) -> Int {
                        var ssmIdx = 0
                        for c in cache {
                            guard ssmIdx < checkpoint.ssmStates.count else { break }
                            if c.restore(
                                from: .ssm(checkpoint.ssmStates[ssmIdx]),
                                options: .init()
                            ) {
                                ssmIdx += 1
                            }
                        }
                        return ssmIdx
                    }

                    func _configureCachedPrefixState(
                        restoredBoundary: Int,
                        replayTokens: [Int],
                        trimAttentionToBoundary: Bool
                    ) {
                        if trimAttentionToBoundary {
                            for c in cache where c.isTrimmable && !(c is VMLXMambaCache) && c.offset > restoredBoundary {
                                c.trim(c.offset - restoredBoundary)
                            }
                        }

                        let genSuffix = genPromptLen > 0 ? Array(tokens.suffix(genPromptLen)) : [Int]()
                        cachedTokenCount = restoredBoundary
                        let allReplay = replayTokens + genSuffix
                        inputTokens = MLXArray(allReplay.map { Int32($0) })
                        _vmlxLog2("[Gen] CachedPrefix: boundary=\(restoredBoundary), replay=\(replayTokens.count), genSuffix=\(genSuffix.count), total=\(allReplay.count), trim=\(trimAttentionToBoundary)")
                    }

                    func _captureCurrentSSMSnapshot() -> [[MLXArray]] {
                        cache.compactMap { c -> [MLXArray]? in
                            guard let mamba = c as? VMLXMambaCache else { return nil }
                            // Force real buffer copy via multiply-by-1 (not $0[.ellipsis] which is a lazy view).
                            // SSM state is modified in-place by forward pass — lazy view would be corrupted.
                            return mamba.state.map { $0 * 1 }
                        }
                    }

                    func _attentionCacheOffset() -> Int {
                        cache.first(where: { !($0 is VMLXMambaCache) })?.offset ?? 0
                    }

                    func _exportLiveHybridCache(
                        targetOffset: Int,
                        ssmSnapshot: [[MLXArray]]? = nil
                    ) -> HybridCache? {
                        guard targetOffset > 0 else { return nil }

                        var layers: [LayerCacheEntry] = []
                        var ssmSnapshotIdx = 0

                        for c in cache {
                            if c is VMLXMambaCache,
                               let snapshot = ssmSnapshot,
                               ssmSnapshotIdx < snapshot.count {
                                layers.append(.ssm(SSMStateLayer(
                                    state: snapshot[ssmSnapshotIdx].map { $0[.ellipsis] }
                                )))
                                ssmSnapshotIdx += 1
                                continue
                            }

                            guard let entry = c.exportCacheEntry() else { return nil }

                            switch entry {
                            case .attention(let kv):
                                if kv.offset > targetOffset {
                                    layers.append(.attention(kv.truncated(to: targetOffset)))
                                } else {
                                    layers.append(entry)
                                }

                            case .compressedAttention(let encodedKeys, let encodedValues, let offset):
                                if offset > targetOffset {
                                    guard let sliced = TurboQuantLayerCache.sliceCompressedAttention(
                                        encodedKeys,
                                        encodedValues,
                                        range: 0..<targetOffset
                                    ) else {
                                        return nil
                                    }
                                    layers.append(sliced)
                                } else {
                                    layers.append(entry)
                                }

                            case .ssm, .placeholder:
                                layers.append(entry)
                            }
                        }

                        let hybridCache = HybridCache(layers: layers)
                        hybridCache.materialized()
                        return hybridCache
                    }

                    _vmlxLog2("[Gen] Fetch cache: \(cacheKeyTokens.count) cacheKeyTokens, \(tokens.count) total tokens, genPromptLen=\(genPromptLen)")
                    let fetchResult = self.scheduler.cache.fetch(tokens: cacheKeyTokens)
                    switch fetchResult {
                    case .hit(let cachedHybrid, let remaining, let detail, let ssmCheckpoint)
                        where cachedHybrid.layerCount == cache.count:
                        cacheDetailStr = "\(detail)"
                        _vmlxLog2("[Gen] Cache HIT: \(cachedHybrid.layerCount) layers, \(remaining.count) remaining tokens, detail=\(detail)")

                        let restoredBoundary = remaining.isEmpty
                            ? max(0, cacheKeyTokens.count - 1)
                            : cacheKeyTokens.count - remaining.count
                        var effectiveCheckpoint = ssmCheckpoint
                        var canUseHit = true

                        if container.isHybrid,
                           remaining.isEmpty,
                           restoredBoundary > 0,
                           ssmCheckpoint?.boundary != restoredBoundary {
                            do {
                                effectiveCheckpoint = try await self.ssmReDeriver?.requestReDerive(
                                    tokens: cacheKeyTokens,
                                    stableBoundary: restoredBoundary
                                )
                                if effectiveCheckpoint != nil {
                                    cacheDetailStr = "rederived/\(detail)"
                                    _vmlxLog2("[Gen] Exact hybrid hit: recovered boundary-aligned SSM at \(restoredBoundary) tokens")
                                } else {
                                    let requiresSSM = request.enableThinking ?? true
                                    if requiresSSM {
                                        canUseHit = false
                                        _vmlxLog2("[Gen] Exact hybrid hit: boundary-aligned SSM unavailable at \(restoredBoundary) tokens; falling back to full prefill (SSM required for thinking)")
                                    } else {
                                        cacheDetailStr = "async-rederive/\(detail)"
                                        _vmlxLog2("[Gen] Exact hybrid hit: async re-derive started, proceeding with KV-only for this turn (thinking disabled)")
                                    }
                                }
                            } catch {
                                canUseHit = false
                                _vmlxLog2("[Gen] Exact hybrid hit: boundary re-derive failed at \(restoredBoundary) tokens: \(error.localizedDescription); full prefill")
                            }
                        }

                        if canUseHit {
                            let restoredLayers = _restoreCachedHybrid(cachedHybrid)
                            // Log cache state after restore for debugging
                            let cacheType = cache.first.map { String(describing: type(of: $0)) } ?? "unknown"
                            let restoredOffset = (cache.first as? VMLXBaseKVCache)?.offset ?? -1
                            let firstEntryType: String = cachedHybrid.layers.first.map {
                                switch $0 {
                                case .attention: return "attention"
                                case .compressedAttention: return "compressedAttention(TQ)"
                                case .ssm: return "ssm"
                                case .placeholder: return "placeholder"
                                }
                            } ?? "empty"
                            _vmlxLog2("[Gen] Restored \(restoredLayers)/\(cachedHybrid.layerCount) layers, type=\(cacheType), offset=\(restoredOffset), entryType=\(firstEntryType)")

                            // For hybrid models only: inject SSM companion state from checkpoint.
                            // Non-hybrid models have no VMLXMambaCache entries, so this is a no-op.
                            if container.isHybrid, let checkpoint = effectiveCheckpoint, !checkpoint.ssmStates.isEmpty {
                                let ssmIdx = _injectSSMCheckpoint(checkpoint)
                                _vmlxLog2("[Gen] Injected \(ssmIdx) SSM companion states from checkpoint")
                            }

                            // Force-evaluate restored cache to complete any lazy TQ decode
                            // BEFORE the forward pass starts. Without this, the TQ decode
                            // computation graph (QJL matmul × N layers) runs during inference,
                            // causing GPU stalls and 3-10x decode slowdowns.
                            eval(cache)
                            Memory.clearCache()

                            if remaining.isEmpty {
                                // Exact hits are normalized back to the standard N-1
                                // boundary so the final cached token can be replayed for logits.
                                let replayTokens = cacheKeyTokens.isEmpty ? [Int]() : [cacheKeyTokens.last!]
                                _configureCachedPrefixState(
                                    restoredBoundary: restoredBoundary,
                                    replayTokens: replayTokens,
                                    trimAttentionToBoundary: true
                                )
                            } else {
                                // Prefix hit: some cacheKeyTokens matched. Prefill remaining + suffix.
                                _configureCachedPrefixState(
                                    restoredBoundary: restoredBoundary,
                                    replayTokens: remaining,
                                    trimAttentionToBoundary: false
                                )
                            }
                        } else {
                            inputTokens = MLXArray(tokens.map { Int32($0) })
                        }
                    case .partialHit(let attentionCache, let remaining, let detail):
                        cacheDetailStr = "partial/\(detail)"
                        if container.isHybrid {
                            let matchedBoundary = cacheKeyTokens.count - remaining.count
                            let recoveryBoundary: Int
                            let replayTokens: [Int]
                            let trimAttentionToBoundary: Bool

                            if remaining.isEmpty, matchedBoundary > 0 {
                                recoveryBoundary = matchedBoundary - 1
                                replayTokens = [cacheKeyTokens.last!]
                                trimAttentionToBoundary = true
                            } else {
                                recoveryBoundary = matchedBoundary
                                replayTokens = remaining
                                trimAttentionToBoundary = false
                            }

                            if recoveryBoundary > 0 {
                                do {
                                    if let checkpoint = try await self.ssmReDeriver?.requestReDerive(
                                        tokens: cacheKeyTokens,
                                        stableBoundary: recoveryBoundary
                                    ) {
                                        let restoredLayers = _restoreCachedHybrid(attentionCache)
                                        let injectedLayers = _injectSSMCheckpoint(checkpoint)
                                        eval(cache)
                                        Memory.clearCache()

                                        _configureCachedPrefixState(
                                            restoredBoundary: recoveryBoundary,
                                            replayTokens: replayTokens,
                                            trimAttentionToBoundary: trimAttentionToBoundary
                                        )
                                        cacheDetailStr = "rederived/\(detail)"
                                        _vmlxLog2("[Gen] Cache PARTIAL HIT (hybrid): recovered SSM at boundary=\(recoveryBoundary), restored=\(restoredLayers), injected=\(injectedLayers), remaining=\(replayTokens.count)")
                                    } else {
                                        let requiresSSM = request.enableThinking ?? true
                                        if requiresSSM {
                                            _vmlxLog2("[Gen] Cache PARTIAL HIT (hybrid): re-derive pending/unavailable at boundary=\(recoveryBoundary); full prefill \(tokens.count) tokens (SSM required for thinking)")
                                            inputTokens = MLXArray(tokens.map { Int32($0) })
                                        } else {
                                            let restoredLayers = _restoreCachedHybrid(attentionCache)
                                            eval(cache)
                                            Memory.clearCache()

                                            _configureCachedPrefixState(
                                                restoredBoundary: recoveryBoundary,
                                                replayTokens: replayTokens,
                                                trimAttentionToBoundary: trimAttentionToBoundary
                                            )
                                            cacheDetailStr = "async-rederive/\(detail)"
                                            _vmlxLog2("[Gen] Cache PARTIAL HIT (hybrid): async re-derive started, proceeding with KV-only (restored=\(restoredLayers)) for this turn (thinking disabled)")
                                        }
                                    }
                                } catch {
                                    _vmlxLog2("[Gen] Cache PARTIAL HIT (hybrid): re-derive failed at boundary=\(recoveryBoundary): \(error.localizedDescription); full prefill \(tokens.count) tokens")
                                    inputTokens = MLXArray(tokens.map { Int32($0) })
                                }
                            } else {
                                _vmlxLog2("[Gen] Cache PARTIAL HIT (hybrid): boundary too small for SSM recovery; full prefill \(tokens.count) tokens")
                                inputTokens = MLXArray(tokens.map { Int32($0) })
                            }
                        } else {
                            // Non-hybrid: no SSM layers, so partial hit = prefix hit.
                            // Restore attention KV and prefill only remaining + gen_prompt_len.
                            _vmlxLog2("[Gen] Cache PARTIAL HIT (non-hybrid): \(attentionCache.layerCount) layers, \(remaining.count) remaining, detail=\(detail)")
                            _ = _restoreCachedHybrid(attentionCache)
                            if remaining.isEmpty {
                                _configureCachedPrefixState(
                                    restoredBoundary: max(0, cacheKeyTokens.count - 1),
                                    replayTokens: cacheKeyTokens.isEmpty ? [Int]() : [cacheKeyTokens.last!],
                                    trimAttentionToBoundary: true
                                )
                            } else {
                                _configureCachedPrefixState(
                                    restoredBoundary: cacheKeyTokens.count - remaining.count,
                                    replayTokens: remaining,
                                    trimAttentionToBoundary: false
                                )
                            }
                        }

                    case .miss, .hit(_, _, _, _):
                        // Complete miss or layer count mismatch — full prefill
                        _vmlxLog2("[Gen] Cache MISS: prefilling \(tokens.count) tokens")
                        inputTokens = MLXArray(tokens.map { Int32($0) })
                    }

                    var generatedTokenCount = 0

                    // Cancel any background SSM re-derivation before starting
                    // forward passes. The re-deriver calls container.forward()
                    // which must not run concurrently with generation's forward()
                    // — MLX's internal state (compiled kernel caches, Metal command
                    // queues) is not safe for concurrent model inference.
                    if let reDeriver = self.ssmReDeriver, await reDeriver.activeTaskCount > 0 {
                        await reDeriver.cancelAll()
                    }

                    // Hybrid models need a boundary-aligned SSM snapshot for the
                    // cache store. The target boundary is storeTokens.count, which
                    // can already be satisfied by a restored cache hit.
                    let totalPrefillTokens = inputTokens.dim(0)
                    let configuredPrefillStep = self.scheduler.config.prefillStepSize
                    let prefillStep = self.adaptivePrefillStep(
                        for: container,
                        configuredStep: configuredPrefillStep,
                        totalPrefillTokens: totalPrefillTokens
                    )
                    if prefillStep != configuredPrefillStep {
                        _vmlxLog2(
                            "[Gen] Adaptive prefill step: \(prefillStep) "
                                + "(configured \(configuredPrefillStep), family=\(container.familyConfig.family), "
                                + "experts=\(container.model.detected.numExperts ?? 0), total=\(totalPrefillTokens))"
                        )
                    }
                    let storeTokens: [Int]
                    if cacheKeyTokens.count > 1 {
                        storeTokens = Array(cacheKeyTokens.dropLast(1))
                    } else {
                        storeTokens = cacheKeyTokens
                    }
                    let storeTokensCount = storeTokens.count
                    var prefillSSMSnapshot: [[MLXArray]]? = nil
                    let needSSMSnapshot = container.isHybrid && storeTokensCount > 0
                    let boundaryAdvance = max(0, storeTokensCount - cachedTokenCount)

                    func _syncPagedPrefixIfNeeded() {
                        guard let pagedWriteSession, storeTokensCount > 0 else { return }
                        let currentPrefixCount = min(storeTokensCount, _attentionCacheOffset())
                        guard currentPrefixCount > 0 else { return }

                        let pagedBlockSize = self.scheduler.config.pagedCacheBlockSize
                        let fullyCommittedTokens =
                            pagedWriteSession.committedBlockCount * pagedBlockSize
                        guard currentPrefixCount >= pagedBlockSize,
                              currentPrefixCount > fullyCommittedTokens else {
                            return
                        }

                        let syncTokenCount =
                            (currentPrefixCount / pagedBlockSize)
                            * pagedBlockSize
                        guard syncTokenCount > fullyCommittedTokens else { return }

                        if let hybridCache = _exportLiveHybridCache(targetOffset: syncTokenCount) {
                            pagedWriteSession.sync(
                                tokens: Array(storeTokens.prefix(syncTokenCount)),
                                cache: hybridCache
                            )
                        }
                    }

                    func _captureBoundarySnapshotIfNeeded(_ reason: String) {
                        guard needSSMSnapshot, prefillSSMSnapshot == nil else { return }
                        prefillSSMSnapshot = _captureCurrentSSMSnapshot()
                        _vmlxLog2("[Gen] SSM snapshot: \(prefillSSMSnapshot!.count) layers (\(reason), boundary=\(storeTokensCount))")
                    }

                    if needSSMSnapshot && cachedTokenCount == storeTokensCount {
                        _captureBoundarySnapshotIfNeeded("restored")
                    }
                    _syncPagedPrefixIfNeeded()

                    // Chunked prefill helper
                    func _chunkedPrefill(_ start: Int, _ end: Int) {
                        var pos = start
                        while pos < end {
                            var chunkEnd = min(pos + prefillStep, end)
                            if needSSMSnapshot,
                               prefillSSMSnapshot == nil,
                               boundaryAdvance > pos,
                               boundaryAdvance < chunkEnd {
                                chunkEnd = boundaryAdvance
                            }
                            _vmlxLog2("[Gen] Prefill chunk \(pos)..<\(chunkEnd) (\(chunkEnd - pos) tokens)")
                            let chunkInput = inputTokens[pos ..< chunkEnd].expandedDimensions(axis: 0)
                            _vmlxLog2("[Gen] Prefill: calling forward (input shape: \(chunkInput.shape))")
                            let prefillLogits = container.forward(chunkInput, cache: cache)
                            _vmlxLog2("[Gen] Prefill: forward returned (logits shape: \(prefillLogits.shape)), calling MLX.eval on cache")
                            MLX.eval(cache)
                            _vmlxLog2("[Gen] Prefill chunk \(pos)..<\(chunkEnd) forward+eval OK")
                            pos = chunkEnd
                            if needSSMSnapshot && prefillSSMSnapshot == nil && pos == boundaryAdvance {
                                _captureBoundarySnapshotIfNeeded("post-prefill")
                            }
                            _syncPagedPrefixIfNeeded()
                        }
                    }

                    // Single-pass prefill for all models (including hybrid SSM).
                    // For hybrid cache storage, capture SSM exactly when the cache
                    // reaches the store boundary instead of snapshotting only on
                    // uncached turns or after the whole prompt.
                    _vmlxLog2("[Gen] Prefill: \(totalPrefillTokens) tokens, step=\(prefillStep), cacheTypes=\(cache.prefix(3).map { String(describing: type(of: $0)) })")

                    if totalPrefillTokens > 1 {
                        _chunkedPrefill(0, totalPrefillTokens - 1)
                        _vmlxLog2("[Gen] Chunked prefill done")
                    }
                    // Clear Metal cache once after all prefill (not per-chunk).
                    // MoE prefill generates many temporaries (routing tensors, expert
                    // masks, sorted indices) — clearing per-chunk forces expensive
                    // dealloc/realloc cycles.
                    Memory.clearCache()

                    // Final token → logits for first generated token
                    let _prefillStart = CFAbsoluteTimeGetCurrent()
                    _vmlxLog2("[Gen] Final token forward pass starting")
                    let lastToken = inputTokens[(totalPrefillTokens - 1)...]
                    let prefillLogits = container.forward(
                        lastToken.expandedDimensions(axis: 0), cache: cache)
                    _vmlxLog2("[Gen] Final token forward pass done, sampling...")
                    let firstTokenId = Sampler.sample(
                        logits: prefillLogits[0, -1], params: samplingParams)
                    var y = MLXArray(Int32(firstTokenId))
                    eval(y)
                    let _prefillMs = (CFAbsoluteTimeGetCurrent() - _prefillStart) * 1000
                    _vmlxLog2("[Gen] Final prefill token: \(String(format: "%.0f", _prefillMs))ms")

                    if tqEnabled {
                        let tqStart = CFAbsoluteTimeGetCurrent()
                        var tqLayers = 0
                        for c in cache {
                            if c.finalizePrefillIfNeeded() {
                                tqLayers += 1
                            }
                        }
                        if tqLayers > 0 {
                            eval(cache)
                            Memory.clearCache()
                            let tqMs = (CFAbsoluteTimeGetCurrent() - tqStart) * 1000
                            _vmlxLog2("[Gen] TQ finalize: \(tqLayers) layers in \(String(format: "%.0f", tqMs))ms")
                        }
                    }

                    // Double-buffered generation loop (matches mlx-lm Python pattern):
                    // Pipeline: build graph for NEXT token while GPU evaluates CURRENT token.
                    // asyncEval starts GPU work, then CPU does decode/yield concurrently.
                    // For MiniMax 122B: ~30ms GPU, ~10ms CPU → effective max(30,10) = 30ms vs 40ms sequential.

                    // Think tag handling is done entirely at the UI level:
                    // - PrependThinkTagMiddleware prepends <think> if </think> is detected
                    // - StreamingDeltaProcessor parses <think>/</ think> tags
                    // VMLXRuntimeActor just passes raw token text through.

                    // Pre-compute cache state array references once — avoids O(num_layers)
                    // Swift allocations per decode step. MLXArray references stay valid as
                    // cache objects mutate in-place (matching Python's pattern).
                    var nextY: MLXArray? = nil
                    let _cacheStateArrays: [MLXArray] = {
                        var arrays: [MLXArray] = []
                        for c in cache { arrays.append(contentsOf: c.innerState()) }
                        return arrays
                    }()
                    asyncEval([y] + _cacheStateArrays)

                    // Compiled forward pass for non-hybrid models (matches Python VMLX --jit).
                    // Wraps the entire model.__call__ with compile(shapeless:true), fusing
                    // hundreds of Metal kernel dispatches into a single compiled graph.
                    // MoE models benefit enormously: 36 layers × ~20 ops = 720 dispatches per
                    // token become a single graph replay.
                    //
                    // Hybrid SSM models (Qwen3.5 GatedDeltaNet) crash during graph tracing —
                    // MLX can't trace through mixed VMLXMambaCache + KV cache state updates.
                    // Pure attention+MoE models (Gemma4, Mistral4, MiniMax, Qwen3.5-MoE) work.
                    // Compiled forward pass for non-hybrid, non-TQ models (matches Python VMLX --jit).
                    // Wraps model.__call__ with compile(shapeless:true), fusing hundreds of
                    // Metal kernel dispatches into a single compiled graph.
                    //
                    // Disabled when:
                    // - Hybrid SSM models: compile can't trace mixed cache types
                    // - TurboQuant enabled: TQ cache has complex state (QJL, unified buffers,
                    //   scatter writes) that crashes the compile tracer (EXC_BAD_ACCESS)
                    let _compiledForward: ((@Sendable (MLXArray) -> MLXArray))?
                    if !container.isHybrid && !tqEnabled {
                        let nativeModel = container.nativeModel
                        let modelModule = nativeModel as Module
                        let cacheOutputs: [any Updatable] = cache.map { $0 as any Updatable }
                        _compiledForward = compile(
                            inputs: [modelModule],
                            outputs: cacheOutputs,
                            shapeless: true
                        ) { (tokens: MLXArray) -> MLXArray in
                            nativeModel(tokens, cache: cache)
                        }
                        _vmlxLog2("[Gen] Compiled forward pass enabled (non-hybrid, non-TQ)")
                    } else {
                        _compiledForward = nil
                        _vmlxLog2("[Gen] Compiled forward pass DISABLED (hybrid=\(container.isHybrid) tq=\(tqEnabled))")
                    }

                    // Repetition penalty state: track generated tokens to penalize repeats.
                    let repPenalty = samplingParams.repetitionPenalty
                    let applyRepPenalty = repPenalty != 1.0
                    var generatedTokenIds: [Int] = []
                    var channelState: String = ""  // GPT-OSS channel protocol state

                    // Decode parameters for the hot loop
                    let isGreedy = samplingParams.isGreedy
                    let temp = samplingParams.temperature

                    let _genStart = CFAbsoluteTimeGetCurrent()
                    var _steadyStart: Double = 0
                    let _warmupTokens = 3  // First N tokens are slow (Metal pipeline warmup)

                    for _step in 0 ..< maxTokens {
                        try Task.checkCancellation()

                        // Double-buffered: build graph for NEXT token while GPU evaluates CURRENT.
                        let _stepStart = CFAbsoluteTimeGetCurrent()
                        if _step < maxTokens - 1 {
                            let stepLogits: MLXArray
                            if let cf = _compiledForward {
                                stepLogits = cf(y.reshaped(1, 1))
                            } else {
                                stepLogits = container.forward(y.reshaped(1, 1), cache: cache)
                            }
                            var logits = stepLogits[0, -1]

                            if applyRepPenalty && !generatedTokenIds.isEmpty {
                                logits = Sampler.applyRepetitionPenalty(
                                    logits: logits, tokens: generatedTokenIds, penalty: repPenalty)
                            }

                            if isGreedy {
                                nextY = logits.argMax()
                            } else if temp > 0 {
                                nextY = compiledCategoricalSample(logits, MLXArray(temp))
                            } else {
                                nextY = MLXRandom.categorical(logits.expandedDimensions(axis: 0))
                            }
                            asyncEval([nextY!] + _cacheStateArrays)
                        }

                        // Materialize CURRENT token (blocks until GPU done with y)
                        let currentToken = y.item(Int.self)
                        generatedTokenIds.append(currentToken)

                        // Per-token timing for first 5 tokens (profiling TQ cache impact)
                        if _step < 5 {
                            let _stepMs = (CFAbsoluteTimeGetCurrent() - _stepStart) * 1000
                            _vmlxLog2("[Gen] tok[\(_step)]: \(String(format: "%.1f", _stepMs))ms")
                        }

                        // Track steady-state start (after Metal pipeline warmup)
                        if _step == _warmupTokens {
                            _steadyStart = CFAbsoluteTimeGetCurrent()
                        }

                        if _step == 5 || _step == 20 || _step == 100 {
                            let _elapsed = CFAbsoluteTimeGetCurrent() - _genStart
                            let _steadyToks = _step - _warmupTokens
                            let _steadyElapsed = _steadyStart > 0 ? CFAbsoluteTimeGetCurrent() - _steadyStart : 0
                            let _steadyTPS = _steadyElapsed > 0 ? Double(_steadyToks) / _steadyElapsed : 0
                            _vmlxLog2("[Gen] \(_step) tok: \(String(format: "%.1f", Double(_step)/_elapsed)) avg | \(String(format: "%.1f", _steadyTPS)) steady tok/s")
                        }

                        generatedTokenCount += 1

                        // Emit live stats periodically so the UI shows speed during streaming.
                        // First emission at token 10 (after Metal warmup stabilizes), then every 20 tokens.
                        // This is lightweight — just a struct yield, no GPU work.
                        if _step == 10 || (_step > 10 && _step % 20 == 0) {
                            let _now = CFAbsoluteTimeGetCurrent()
                            let _steadyToks = _step - _warmupTokens
                            let _steadyElapsed = _steadyStart > 0 ? _now - _steadyStart : _now - _genStart
                            let _liveTPS = _steadyElapsed > 0 ? Double(max(1, _steadyToks)) / _steadyElapsed : 0
                            let _prefillElapsed = max(0.001, (_genStart - requestStart))
                            let _prefillToks = max(1, promptTokenCount - cachedTokenCount)
                            continuation.yield(.usage(
                                promptTokens: promptTokenCount,
                                completionTokens: generatedTokenCount,
                                cachedTokens: cachedTokenCount,
                                ttft: ttftTime,
                                prefillToksPerSec: Double(_prefillToks) / _prefillElapsed,
                                decodeToksPerSec: _liveTPS,
                                cacheDetail: cacheDetailStr,
                                cacheBytes: 0  // Skip expensive cache estimate during hot loop
                            ))
                        }

                        // Check for EOS
                        if eosTokenIds.contains(currentToken) || stopTokenIds.contains(currentToken) {
                            break
                        }

                        // Decode token to text (CPU work overlaps with GPU computing nextY)
                        var text = container.decode([currentToken])

                        // GPT-OSS channel protocol: suppress internal framing tokens
                        // and transform channel names to <think>/</ think>.
                        if text == "<|channel|>" || text == "<|message|>"
                            || text == "<|end|>" || text == "<|start|>"
                            || text == "<|endoftext|>" {
                            channelState = text
                            y = nextY ?? y
                            continue
                        }
                        if channelState == "<|channel|>" {
                            // Channel name token — transform to think tags
                            if text == "analysis" || text.hasPrefix("analysis") {
                                text = "<think>"
                            } else if text == "final" || text == "reply" || text == "assistant" {
                                text = "</think>"
                            }
                            channelState = ""
                        } else if channelState == "<|start|>" {
                            // After <|start|>, consume the role token ("assistant", "system", etc.)
                            // — it's a template artifact, not model output.
                            channelState = ""
                            y = nextY ?? y
                            continue
                        } else if channelState == "<|message|>" || channelState == "<|end|>" {
                            channelState = ""
                            // First content token after <|message|> — pass through
                        }

                        let emitText = text.vmlxStripped

                        if emitText.isEmpty {
                            y = nextY ?? y
                            continue
                        }

                        let events = accumulator.process(text: emitText, tokenIds: [currentToken])
                        for event in events {
                            if !firstTokenEmitted {
                                ttftTime = CFAbsoluteTimeGetCurrent() - requestStart
                                firstTokenEmitted = true
                            }
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

                        // Periodic Metal cache cleanup — release temporary GPU allocations.
                        // At 90+ tok/s, every 1024 tokens = ~11s intervals. Was 256 (stall every ~2.8s).
                        // GPU sync is expensive (~0.5-1ms) and breaks double-buffer pipelining.
                        if _step % 1024 == 0 && _step > 0 {
                            Memory.clearCache()
                        }
                    }
                    // end decode loop

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

                    // Compute timing stats BEFORE cache store (TQ encode can take
                    // hundreds of ms and shouldn't be included in decode speed).
                    let decodeEndTime = CFAbsoluteTimeGetCurrent()
                    let totalElapsed = decodeEndTime - requestStart
                    let decodeElapsed = decodeEndTime - _genStart
                    let prefillElapsed = max(0.001, totalElapsed - decodeElapsed)
                    let prefillTokenCount = max(1, promptTokenCount - cachedTokenCount)
                    let prefillTPS = Double(prefillTokenCount) / prefillElapsed

                    // Use steady-state speed if we generated enough tokens
                    let steadyTokens = generatedTokenCount - _warmupTokens
                    let steadyElapsed = _steadyStart > 0 ? decodeEndTime - _steadyStart : decodeElapsed
                    let decodeTPS: Double
                    if steadyTokens > 5 && _steadyStart > 0 {
                        decodeTPS = Double(steadyTokens) / steadyElapsed
                    } else {
                        decodeTPS = generatedTokenCount > 0
                            ? Double(generatedTokenCount) / decodeElapsed : 0
                    }

                    // Emit usage with timing and live cache footprint
                    let liveCacheBytes = self.estimateLiveCacheBytes(cache)
                    _vmlxLog2("[Gen] Stats: ttft=\(String(format:"%.3f",ttftTime))s pp=\(String(format:"%.1f",prefillTPS))t/s tg=\(String(format:"%.1f",decodeTPS))t/s prompt=\(promptTokenCount) gen=\(generatedTokenCount) cached=\(cachedTokenCount)")
                    continuation.yield(.usage(
                        promptTokens: promptTokenCount,
                        completionTokens: generatedTokenCount,
                        cachedTokens: cachedTokenCount,
                        ttft: ttftTime,
                        prefillToksPerSec: prefillTPS,
                        decodeToksPerSec: decodeTPS,
                        cacheDetail: cacheDetailStr,
                        cacheBytes: liveCacheBytes
                    ))

                    // Pre-compute the cache export BEFORE finishing the stream.
                    // This is the expensive part (trim + export 40 layers + materialize).
                    // Doing it before finish() means the GPU work overlaps with the
                    // last token's eval, and the store after finish() is just a fast
                    // pointer copy into the cache coordinator.
                    var preparedCache: HybridCache? = nil
                    var pagedHandledByLiveSession = false
                    if !storeTokens.isEmpty {
                        let targetOffset = storeTokens.count
                        for c in cache {
                            if !(c is VMLXMambaCache) && c.isTrimmable && c.offset > targetOffset {
                                c.trim(c.offset - targetOffset)
                            }
                        }

                        if needSSMSnapshot && prefillSSMSnapshot == nil {
                            _vmlxLog2("[Gen] SSM snapshot skipped: no boundary-aligned snapshot (post-decode state is contaminated)")
                        }

                        var ssmSnapshotIdx = 0
                        var layers: [LayerCacheEntry] = []
                        for c in cache {
                            if c is VMLXMambaCache {
                                if let snapshots = prefillSSMSnapshot,
                                   ssmSnapshotIdx < snapshots.count {
                                    layers.append(.ssm(SSMStateLayer(
                                        state: snapshots[ssmSnapshotIdx])))
                                    ssmSnapshotIdx += 1
                                } else {
                                    layers.append(.placeholder)
                                }
                            } else if let entry = c.exportCacheEntry() {
                                switch entry {
                                case .attention(let kv):
                                    let trimmedEntry: LayerCacheEntry
                                    if kv.offset > targetOffset {
                                        trimmedEntry = .attention(kv.truncated(to: targetOffset))
                                    } else {
                                        trimmedEntry = entry
                                    }
                                    layers.append(trimmedEntry)
                                case .compressedAttention, .ssm, .placeholder:
                                    layers.append(entry)
                                }
                            }
                        }
                        if !layers.isEmpty {
                            let hybridCache = HybridCache(layers: layers)
                            hybridCache.materialized()
                            preparedCache = hybridCache

                            if let pagedWriteSession,
                               let pagedHybridCache = _exportLiveHybridCache(
                                targetOffset: targetOffset,
                                ssmSnapshot: prefillSSMSnapshot
                               ) {
                                pagedWriteSession.finalize(
                                    tokens: storeTokens,
                                    cache: pagedHybridCache
                                )
                                pagedHandledByLiveSession = true
                            }
                        }
                    }

                    // Finish the stream IMMEDIATELY — UI can now dismiss
                    // the stop button and show stats without waiting for
                    // the cache store.
                    continuation.finish()

                    // Store the pre-computed cache (fast — just pointer copies).
                    if let hybridCache = preparedCache {
                        self.scheduler.cache.store(
                            tokens: storeTokens,
                            cache: hybridCache,
                            includePaged: !pagedHandledByLiveSession
                        )
                        _vmlxLog2("[Gen] Stored cache: \(storeTokens.count) tokens (stripped \(genPromptLen) gen_prompt + \(generatedTokenCount) generated + 1 last)")
                    }

                } catch is CancellationError {
                    // Generation was stopped before cache store completed.
                    // Do not nuke the cache stack here — that destroys unrelated
                    // L2 entries and makes cancellations look like cache failures.
                    pagedWriteSession?.abort()
                    continuation.finish()
                } catch {
                    // Request-scoped recovery: invalidate cache key, clear volatile
                    // layers, and release GPU memory. This handles Metal OOM and other
                    // GPU errors by freeing temporary allocations before fallback.
                    pagedWriteSession?.abort()
                    self.scheduler.cache.invalidate(tokens: cacheKeyTokens)
                    self.scheduler.cache.clearVolatile()
                    Memory.clearCache()
                    _vmlxLog2("[Gen] Error recovery: \(error.localizedDescription) — cache invalidated, GPU memory released")
                    continuation.finish(throwing: error)
                }
            }

            generationTask = task

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }

        // Register the task IMMEDIATELY on the actor — no @MainActor dispatch.
        // The old pattern (Task { @MainActor in await self._trackGeneration(...) })
        // introduced a 2-hop scheduling delay (MainActor → VMLXRuntimeActor) which
        // caused the task to be missing from activeGenerations when the next request
        // tried to cancel it, leading to concurrent GPU access and crashes.
        if let task = generationTask {
            _trackGeneration(requestId: requestId, task: task)
        }

        return stream
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
