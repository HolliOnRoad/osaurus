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
               cacheDetail: String?)
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
    ) {
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
        }
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

        if let reDeriver = ssmReDeriver {
            await reDeriver.cancelAll()
            await reDeriver.setModel(nil)
        }
        ssmReDeriver = nil

        modelContainer = nil
        loadedModels.removeAll()
        currentModelName = nil

        // Force Metal memory cleanup AFTER all generation tasks have stopped
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

        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
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
                    let tqEnabled = self.scheduler.config.enableTurboQuant
                    _vmlxLog2("[Gen] Config: kvQuant=\(kvBitsStr) tq=\(tqEnabled) hybrid=\(container.isHybrid)")
                    let cache = container.newCache(kvBits: kvBits, kvGroupSize: kvGroupSize)
                    var inputTokens: MLXArray
                    var cachedTokenCount = 0

                    _vmlxLog2("[Gen] Fetch cache: \(cacheKeyTokens.count) cacheKeyTokens, \(tokens.count) total tokens, genPromptLen=\(genPromptLen)")
                    let fetchResult = self.scheduler.cache.fetch(tokens: cacheKeyTokens)
                    switch fetchResult {
                    case .hit(let cachedHybrid, let remaining, let detail, let ssmCheckpoint)
                        where cachedHybrid.layerCount == cache.count:
                        cacheDetailStr = "\(detail)"
                        _vmlxLog2("[Gen] Cache HIT: \(cachedHybrid.layerCount) layers, \(remaining.count) remaining tokens, detail=\(detail)")
                        // Restore cached KV state into the VMLXKVCache objects.
                        // Uses protocol-based .state setter — works with both VMLXKVCacheSimple
                        // AND VMLXQuantizedKVCache (which re-quantizes on assignment).
                        // Create TQ encoder state ONCE for all layers (avoid recreating
                        // 128×128 QJL matrix + codebook per layer — was 124× overhead).
                        let hasTQ = cachedHybrid.layers.contains {
                            if case .compressedAttention = $0 { return true }
                            return false
                        }
                        let tqState: TurboQuantEncoder.EncoderState?
                        if hasTQ, let firstTQ = cachedHybrid.layers.first(where: {
                            if case .compressedAttention = $0 { return true }
                            return false
                        }), case .compressedAttention(let ek, _, _) = firstTQ {
                            let dim = ek.shape.last ?? 128
                            let keyBits = ek.indexBits + 1
                            tqState = TurboQuantEncoder.EncoderState(
                                dim: dim, keyBits: keyBits,
                                valueBits: ek.indexBits, seed: ek.seed)
                        } else {
                            tqState = nil
                        }

                        var restoredLayers = 0
                        for (i, entry) in cachedHybrid.layers.enumerated() {
                            guard i < cache.count else { break }
                            switch entry {
                            case .attention(let kv):
                                if let kvBase = cache[i] as? VMLXBaseKVCache,
                                   !(cache[i] is VMLXMambaCache) {
                                    kvBase.state = [kv.keys, kv.values]
                                    restoredLayers += 1
                                }
                            case .compressedAttention(let ek, let ev, _):
                                if let kvBase = cache[i] as? VMLXBaseKVCache,
                                   !(cache[i] is VMLXMambaCache) {
                                    let dk: MLXArray
                                    let dv: MLXArray
                                    if let st = tqState {
                                        dk = TurboQuantEncoder.decodeKeys(ek, state: st)
                                        dv = TurboQuantEncoder.decodeValues(ev, state: st)
                                    } else {
                                        dk = TurboQuantEncoder.decodeKeys(ek, seed: ek.seed)
                                        dv = TurboQuantEncoder.decodeValues(ev, seed: ev.seed)
                                    }
                                    eval(dk, dv)
                                    kvBase.state = [dk, dv]
                                    restoredLayers += 1
                                }
                            case .ssm(let ssm):
                                if let mambaCache = cache[i] as? VMLXMambaCache {
                                    mambaCache.state = ssm.state
                                    restoredLayers += 1
                                }
                            }
                        }
                        // Log cache state after restore for debugging
                        let cacheType = cache.first.map { String(describing: type(of: $0)) } ?? "unknown"
                        let restoredOffset = (cache.first as? VMLXBaseKVCache)?.offset ?? -1
                        let firstEntryType: String = cachedHybrid.layers.first.map {
                            switch $0 {
                            case .attention: return "attention"
                            case .compressedAttention: return "compressedAttention(TQ)"
                            case .ssm: return "ssm"
                            }
                        } ?? "empty"
                        _vmlxLog2("[Gen] Restored \(restoredLayers)/\(cachedHybrid.layerCount) layers, type=\(cacheType), offset=\(restoredOffset), entryType=\(firstEntryType)")

                        // For hybrid models only: inject SSM companion state from checkpoint.
                        // Non-hybrid models have no VMLXMambaCache entries, so this is a no-op.
                        if container.isHybrid, let checkpoint = ssmCheckpoint, !checkpoint.ssmStates.isEmpty {
                            var ssmIdx = 0
                            for c in cache {
                                if let mambaCache = c as? VMLXMambaCache,
                                   ssmIdx < checkpoint.ssmStates.count {
                                    mambaCache.state = checkpoint.ssmStates[ssmIdx].state
                                    ssmIdx += 1
                                }
                            }
                            _vmlxLog2("[Gen] Injected \(ssmIdx) SSM companion states from checkpoint")
                        }

                        // Force-evaluate restored cache to complete any lazy TQ decode
                        // BEFORE the forward pass starts. Without this, the TQ decode
                        // computation graph (QJL matmul × N layers) runs during inference,
                        // causing GPU stalls and 3-10x decode slowdowns.
                        eval(cache)
                        Memory.clearCache()

                        // remaining = uncached portion of cacheKeyTokens (not full tokens).
                        // We also need to process gen_prompt_len suffix tokens.
                        let genSuffix = genPromptLen > 0
                            ? Array(tokens.suffix(genPromptLen)) : [Int]()

                        cachedTokenCount = cacheKeyTokens.count - remaining.count
                        if remaining.isEmpty {
                            // Full cache hit on cacheKeyTokens. Re-feed last cached token
                            // to get fresh logits, plus gen_prompt_len suffix.
                            for c in cache where c.isTrimmable && !(c is VMLXMambaCache) {
                                c.trim(1)
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
                        cacheDetailStr = "partial/\(detail)"
                        if container.isHybrid {
                            // Hybrid model: SSM companion missing. SSM state is path-dependent —
                            // can't use KV cache without matching SSM state.
                            // Safe fallback: discard attention cache, full prefill.
                            // The forward pass re-derives SSM state as a side effect.
                            // After generation, CacheCoordinator.store() will save SSM companion
                            // for the next turn (self-healing).
                            //
                            // Future optimization (SSMReDeriver): background re-derivation to
                            // avoid redundant attention recomputation. Requires ModelForwardPass
                            // protocol update to accept [VMLXKVCache] instead of [MLXArray].
                            _vmlxLog2("[Gen] Cache PARTIAL HIT (hybrid, SSM missing): full prefill \(tokens.count) tokens (SSM will be stored after)")
                            inputTokens = MLXArray(tokens.map { Int32($0) })
                        } else {
                            // Non-hybrid: no SSM layers, so partial hit = prefix hit.
                            // Restore attention KV and prefill only remaining + gen_prompt_len.
                            _vmlxLog2("[Gen] Cache PARTIAL HIT (non-hybrid): \(attentionCache.layerCount) layers, \(remaining.count) remaining, detail=\(detail)")
                            for (i, entry) in attentionCache.layers.enumerated() {
                                guard i < cache.count else { break }
                                switch entry {
                                case .attention(let kv):
                                    if let kvBase = cache[i] as? VMLXBaseKVCache,
                                       !(cache[i] is VMLXMambaCache) {
                                        kvBase.state = [kv.keys, kv.values]
                                    }
                                case .compressedAttention(let ek, let ev, _):
                                    if let kvBase = cache[i] as? VMLXBaseKVCache,
                                       !(cache[i] is VMLXMambaCache) {
                                        kvBase.state = [
                                            TurboQuantEncoder.decodeKeys(ek, seed: ek.seed),
                                            TurboQuantEncoder.decodeValues(ev, seed: ev.seed)
                                        ]
                                    }
                                default:
                                    break  // SSM layers handled separately
                                }
                            }
                            let genSuffix = genPromptLen > 0
                                ? Array(tokens.suffix(genPromptLen)) : [Int]()
                            cachedTokenCount = cacheKeyTokens.count - remaining.count
                            if remaining.isEmpty {
                                for c in cache where c.isTrimmable && !(c is VMLXMambaCache) {
                                    c.trim(1)
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
                        _vmlxLog2("[Gen] Cache MISS: prefilling \(tokens.count) tokens")
                        inputTokens = MLXArray(tokens.map { Int32($0) })
                    }

                    var generatedTokenCount = 0

                    // Prefill with two-phase SSM snapshot for hybrid models.
                    // Split at storeTokens boundary so SSM state matches stored KV.
                    let prefillStep = self.scheduler.config.prefillStepSize
                    let totalPrefillTokens = inputTokens.dim(0)
                    let storeTokensCount = max(0, cacheKeyTokens.count - 1)
                    var prefillSSMSnapshot: [[MLXArray]]? = nil

                    let needSSMSnapshot = container.isHybrid && cachedTokenCount == 0
                        && storeTokensCount > 0 && storeTokensCount < totalPrefillTokens

                    // Chunked prefill helper
                    func _chunkedPrefill(_ start: Int, _ end: Int) {
                        var pos = start
                        while pos < end {
                            let chunkEnd = min(pos + prefillStep, end)
                            _ = container.forward(
                                inputTokens[pos ..< chunkEnd].expandedDimensions(axis: 0),
                                cache: cache)
                            eval(cache)
                            Memory.clearCache()
                            pos = chunkEnd
                        }
                    }

                    if needSSMSnapshot {
                        // Phase 1: prefill tokens[0..<storeTokensCount]
                        let _p1Start = CFAbsoluteTimeGetCurrent()
                        _chunkedPrefill(0, storeTokensCount)

                        // Capture SSM state at EXACT storeTokens boundary
                        prefillSSMSnapshot = cache.compactMap { c -> [MLXArray]? in
                            guard let mamba = c as? VMLXMambaCache else { return nil }
                            return mamba.state.map { $0[.ellipsis] }
                        }
                        _vmlxLog2("[Gen] SSM snapshot: \(prefillSSMSnapshot!.count) layers at offset \(storeTokensCount)/\(totalPrefillTokens)")

                        _vmlxLog2("[Gen] Phase 1 done: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent()-_p1Start)*1000))ms for \(storeTokensCount) tokens")

                        // Phase 2: prefill remaining tokens (last cacheKey + genPrompt suffix)
                        if storeTokensCount < totalPrefillTokens - 1 {
                            let _p2Start = CFAbsoluteTimeGetCurrent()
                            _chunkedPrefill(storeTokensCount, totalPrefillTokens - 1)
                            _vmlxLog2("[Gen] Phase 2 done: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent()-_p2Start)*1000))ms for \(totalPrefillTokens - 1 - storeTokensCount) tokens")
                        }
                    } else {
                        // Standard single-phase prefill
                        if totalPrefillTokens > 1 {
                            _chunkedPrefill(0, totalPrefillTokens - 1)
                        }
                    }

                    // Final token → logits for first generated token
                    let _prefillStart = CFAbsoluteTimeGetCurrent()
                    let lastToken = inputTokens[(totalPrefillTokens - 1)...]
                    let prefillLogits = container.forward(
                        lastToken.expandedDimensions(axis: 0), cache: cache)
                    let firstTokenId = Sampler.sample(
                        logits: prefillLogits[0, -1], params: samplingParams)
                    var y = MLXArray(Int32(firstTokenId))
                    eval(y)
                    let _prefillMs = (CFAbsoluteTimeGetCurrent() - _prefillStart) * 1000
                    _vmlxLog2("[Gen] Final prefill token: \(String(format: "%.0f", _prefillMs))ms")

                    // TurboQuant in-memory compression: encode prefill KV → decode once → replace.
                    // After this, attention KV buffers are still float (baseline SDPA speed)
                    // but the original high-precision data is freed, saving ~5x memory.
                    // Only applies to attention layers (not SSM/Mamba layers).
                    //
                    // IMPORTANT: Save original float KV BEFORE compression for the cross-turn
                    // cache store. Storing TQ-decoded (lossy) data across turns causes quality
                    // degradation because the lossy reconstruction compounds through layers.
                    var preTQCacheSnapshot: [(keys: MLXArray, values: MLXArray)]? = nil
                    if tqEnabled && cachedTokenCount == 0 {
                        // Snapshot original float KV for cross-turn storage
                        var snapshot: [(keys: MLXArray, values: MLXArray)] = []
                        for c in cache {
                            if let kvBase = c as? VMLXBaseKVCache, !(c is VMLXMambaCache) {
                                let s = kvBase.state
                                if s.count == 2 {
                                    snapshot.append((keys: s[0], values: s[1]))
                                } else {
                                    snapshot.append((keys: MLXArray(), values: MLXArray()))
                                }
                            } else {
                                snapshot.append((keys: MLXArray(), values: MLXArray()))
                            }
                        }
                        preTQCacheSnapshot = snapshot

                        // Now compress in-place for memory reduction during decode
                        let tqStart = CFAbsoluteTimeGetCurrent()
                        var tqLayers = 0
                        for c in cache {
                            guard let kvBase = c as? VMLXBaseKVCache,
                                  !(c is VMLXMambaCache),
                                  kvBase.offset > TurboQuantEncoder.defaultSinkTokens else { continue }
                            let s = kvBase.state
                            guard s.count == 2 else { continue }
                            let keys = s[0]
                            let values = s[1]
                            let dim = keys.dim(keys.ndim - 1)
                            let state = TurboQuantEncoder.EncoderState(dim: dim, keyBits: 3, valueBits: 3, seed: 42)
                            let ek = TurboQuantEncoder.encodeKeys(keys, state: state)
                            let ev = TurboQuantEncoder.encodeValues(values, state: state)
                            let dk = TurboQuantEncoder.decodeKeys(ek, state: state)
                            let dv = TurboQuantEncoder.decodeValues(ev, state: state)
                            eval(dk, dv)
                            kvBase.state = [dk, dv]
                            tqLayers += 1
                        }
                        if tqLayers > 0 {
                            Memory.clearCache()
                            let tqMs = (CFAbsoluteTimeGetCurrent() - tqStart) * 1000
                            _vmlxLog2("[Gen] TQ compress: \(tqLayers) layers in \(String(format: "%.0f", tqMs))ms")
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

                    // Kick off first token eval
                    var nextY: MLXArray? = nil
                    asyncEval([y])

                    // Repetition penalty state: track generated tokens to penalize repeats.
                    // Without this, models degenerate into repetition loops on long generations.
                    // The penalty is cheap (element-wise ops on logits, O(vocab_size)) — unlike
                    // top-p which requires sorting (removed from hot loop for being 2.5x slower).
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
                            let stepLogits = container.forward(y.reshaped(1, 1), cache: cache)
                            var logits = stepLogits[0, -1]

                            // Apply repetition penalty to previously generated tokens.
                            // Element-wise ops only — no sorting, no softmax, no cumsum.
                            if applyRepPenalty && !generatedTokenIds.isEmpty {
                                logits = Sampler.applyRepetitionPenalty(
                                    logits: logits, tokens: generatedTokenIds, penalty: repPenalty)
                            }

                            if isGreedy {
                                nextY = logits.argMax()
                            } else {
                                let scaled = temp > 0 ? logits / temp : logits
                                nextY = MLXRandom.categorical(scaled.expandedDimensions(axis: 0))
                            }
                            asyncEval([nextY!])
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

                    // Emit usage with timing
                    _vmlxLog2("[Gen] Stats: ttft=\(String(format:"%.3f",ttftTime))s pp=\(String(format:"%.1f",prefillTPS))t/s tg=\(String(format:"%.1f",decodeTPS))t/s prompt=\(promptTokenCount) gen=\(generatedTokenCount) cached=\(cachedTokenCount)")
                    continuation.yield(.usage(
                        promptTokens: promptTokenCount,
                        completionTokens: generatedTokenCount,
                        cachedTokens: cachedTokenCount,
                        ttft: ttftTime,
                        prefillToksPerSec: prefillTPS,
                        decodeToksPerSec: decodeTPS,
                        cacheDetail: cacheDetailStr
                    ))

                    continuation.finish()

                    // Store cache AFTER stream is finished — TQ encode can take
                    // hundreds of ms and must not block the UI stream.
                    let storeTokens: [Int]
                    if cacheKeyTokens.count > 1 {
                        storeTokens = Array(cacheKeyTokens.dropLast(1))
                    } else {
                        storeTokens = cacheKeyTokens
                    }

                    if !storeTokens.isEmpty {
                        let targetOffset = storeTokens.count
                        for c in cache {
                            if !(c is VMLXMambaCache) && c.isTrimmable && c.offset > targetOffset {
                                c.trim(c.offset - targetOffset)
                            }
                        }

                        // Store ORIGINAL float KV for multi-turn cache (not TQ-decoded lossy data).
                        // If TQ was applied during this generation, use the pre-TQ snapshot.
                        // Storing TQ-decoded data across turns causes quality degradation.
                        var ssmSnapshotIdx = 0
                        var layers: [LayerCacheEntry] = []
                        for (layerIdx, c) in cache.enumerated() {
                            if c is VMLXMambaCache {
                                if let snapshots = prefillSSMSnapshot,
                                   ssmSnapshotIdx < snapshots.count {
                                    layers.append(.ssm(SSMStateLayer(
                                        state: snapshots[ssmSnapshotIdx])))
                                    ssmSnapshotIdx += 1
                                }
                            } else if let kvBase = c as? VMLXBaseKVCache {
                                // Use pre-TQ original float if available, otherwise current state.
                                // preTQCacheSnapshot is indexed by layerIdx (same order as cache array).
                                let s: [MLXArray]
                                if let snapshot = preTQCacheSnapshot,
                                   layerIdx < snapshot.count,
                                   snapshot[layerIdx].keys.ndim > 0 {
                                    s = [snapshot[layerIdx].keys, snapshot[layerIdx].values]
                                } else {
                                    s = kvBase.state
                                }
                                guard s.count == 2 else { continue }
                                // Trim to storeTokens length (snapshot has full prefill tokens)
                                let storeKeys: MLXArray
                                let storeValues: MLXArray
                                if s[0].dim(2) > targetOffset {
                                    storeKeys = s[0][.ellipsis, ..<targetOffset, 0...]
                                    storeValues = s[1][.ellipsis, ..<targetOffset, 0...]
                                } else {
                                    storeKeys = s[0]
                                    storeValues = s[1]
                                }
                                layers.append(.attention(KVCacheLayer(
                                    keys: storeKeys, values: storeValues, offset: targetOffset)))
                            }
                        }
                        if !layers.isEmpty {
                            let hybridCache = HybridCache(layers: layers)
                            hybridCache.materialized()
                            self.scheduler.cache.store(tokens: storeTokens, cache: hybridCache)
                            _vmlxLog2("[Gen] Stored cache: \(storeTokens.count) tokens (stripped \(genPromptLen) gen_prompt + \(generatedTokenCount) generated + 1 last)")
                        }
                    }

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
