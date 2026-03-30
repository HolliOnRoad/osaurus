import Foundation
import os

/// Output from a scheduling cycle.
public struct SchedulerOutput: Sendable {
    /// Request IDs that were newly scheduled this cycle.
    public let scheduledRequestIds: [String]
    /// Total tokens across all scheduled requests.
    public let numScheduledTokens: Int
    /// Request IDs that finished this cycle.
    public let finishedRequestIds: Set<String>
    /// Per-request outputs.
    public let outputs: [RequestOutput]
    /// Whether there's more work to do.
    public let hasWork: Bool
}

/// Continuous batching scheduler.
/// Coordinates cache lookup, request batching, and generation lifecycle.
///
/// Lifecycle:
///   1. addRequest() — adds to waiting queue
///   2. schedule() — moves waiting → running, performs cache lookups
///   3. Generation engine processes running requests
///   4. collectOutputs() — gathers new tokens, detects finished requests
///   5. cleanup() — frees resources for finished requests
public final class Scheduler: @unchecked Sendable {

    public let config: SchedulerConfig
    private let requestQueue: RequestQueue
    private let cacheCoordinator: CacheCoordinator
    private let lock = OSAllocatedUnfairLock()

    /// Whether the current model is hybrid (SSM + attention).
    public private(set) var isHybrid: Bool = false

    /// Whether TurboQuant is active.
    public private(set) var isTQActive: Bool = false

    /// Layer pattern for hybrid models.
    public private(set) var layerPattern: [LayerType]?

    /// Model stop tokens (EOS tokens).
    public private(set) var stopTokenIds: Set<Int> = []

    // Stats
    public private(set) var totalRequestsProcessed: Int = 0
    public private(set) var totalTokensGenerated: Int = 0

    public init(config: SchedulerConfig = .autoDetect()) {
        self.config = config
        self.requestQueue = RequestQueue()
        self.cacheCoordinator = CacheCoordinator(config: config.toCacheCoordinatorConfig())
    }

    // MARK: - Configuration

    /// Configure for a specific model. Call after model loading.
    public func configureForModel(
        isHybrid: Bool,
        layerPattern: [LayerType]? = nil,
        stopTokenIds: Set<Int> = [],
        enableTQ: Bool = false
    ) {
        lock.withLock {
            self.isHybrid = isHybrid
            self.layerPattern = layerPattern
            self.stopTokenIds = stopTokenIds
            self.isTQActive = enableTQ
            cacheCoordinator.setHybrid(isHybrid)
        }
    }

    // MARK: - Request Management

    /// Add a new request to the scheduler.
    public func addRequest(_ request: InferenceRequest) {
        requestQueue.addRequest(request)
    }

    /// Abort a request.
    public func abortRequest(_ requestId: String) {
        requestQueue.abortRequest(requestId)
        cacheCoordinator.pagedCache?.deleteBlockTable(requestId)
    }

    // MARK: - Scheduling Cycle

    /// Run one scheduling cycle:
    /// 1. Schedule waiting requests into running batch
    /// 2. Perform cache lookups for newly scheduled requests
    /// 3. Return scheduler output with scheduled request info
    public func schedule() -> SchedulerOutput {
        lock.withLock {
            // Move requests from waiting → running
            let scheduled = requestQueue.schedule(
                maxNumSeqs: config.maxNumSeqs,
                maxBatchedTokens: config.maxNumBatchedTokens
            )

            var totalTokens = 0

            // Perform cache lookups for newly scheduled requests
            for requestId in scheduled {
                guard let request = requestQueue.getRequest(requestId) else { continue }

                let cacheResult = cacheCoordinator.fetch(
                    tokens: request.promptTokenIds
                )

                // Compute tokens for this request based on cache result
                let requestTokens: Int
                switch cacheResult {
                case .hit(let cache, let remaining, _):
                    let cachedCount = request.promptTokenIds.count - remaining.count
                    requestQueue.updateRequest(requestId) { req in
                        req.promptCache = cache
                        req.remainingTokenIds = remaining
                        req.cachedTokens = cachedCount
                    }
                    requestTokens = remaining.count

                case .partialHit(let attentionCache, let remaining, _):
                    let cachedCount = request.promptTokenIds.count - remaining.count
                    requestQueue.updateRequest(requestId) { req in
                        req.promptCache = attentionCache
                        req.remainingTokenIds = remaining
                        req.cachedTokens = cachedCount
                    }
                    requestTokens = remaining.count

                case .miss:
                    requestQueue.updateRequest(requestId) { req in
                        req.remainingTokenIds = req.promptTokenIds
                        req.cachedTokens = 0
                    }
                    requestTokens = request.promptTokenIds.count
                }

                totalTokens += requestTokens
            }

            // Drain finished requests from previous cycle
            let finished = requestQueue.drainFinished()

            return SchedulerOutput(
                scheduledRequestIds: scheduled,
                numScheduledTokens: totalTokens,
                finishedRequestIds: finished,
                outputs: [],  // Populated by generation engine
                hasWork: requestQueue.runningCount > 0 || requestQueue.waitingCount > 0
            )
        }
    }

    /// Record generated output for a request.
    public func recordOutput(requestId: String, tokenId: Int, text: String) {
        requestQueue.updateRequest(requestId) { req in
            req.appendOutputToken(tokenId)
            req.outputText += text
        }
        lock.withLock {
            totalTokensGenerated += 1
        }
    }

    /// Mark a request as finished.
    public func finishRequest(_ requestId: String, reason: FinishReason) {
        requestQueue.finishRequest(requestId, reason: reason)

        // Store cache for finished request
        if let request = requestQueue.getRequest(requestId),
           let cache = request.promptCache {
            let allTokens = request.promptTokenIds + request.outputTokenIds
            cacheCoordinator.store(tokens: allTokens, cache: cache)
        }

        lock.withLock {
            totalRequestsProcessed += 1
        }
    }

    /// Check if a token is a stop token (EOS).
    public func isStopToken(_ tokenId: Int) -> Bool {
        lock.withLock { stopTokenIds.contains(tokenId) }
    }

    // MARK: - Queries

    /// Get a request by ID.
    public func getRequest(_ requestId: String) -> InferenceRequest? {
        requestQueue.getRequest(requestId)
    }

    /// All currently running requests.
    public var runningRequests: [InferenceRequest] {
        requestQueue.runningRequests
    }

    /// Number of waiting requests.
    public var waitingCount: Int { requestQueue.waitingCount }

    /// Number of running requests.
    public var runningCount: Int { requestQueue.runningCount }

    /// Cache statistics.
    public var cacheStats: CacheCoordinatorStats {
        cacheCoordinator.stats
    }

    /// Get the cache coordinator (for direct cache operations).
    public var cache: CacheCoordinator { cacheCoordinator }

    // MARK: - Shutdown

    /// Abort all requests and clean up.
    public func shutdown() {
        for requestId in requestQueue.runningRequestIds {
            abortRequest(requestId)
        }
    }
}
