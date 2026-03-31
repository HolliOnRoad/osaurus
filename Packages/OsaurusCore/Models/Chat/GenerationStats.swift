import Foundation

/// Performance statistics from a generation run, displayed below assistant messages.
/// Populated by the inference engine after generation completes.
public struct GenerationStats: Sendable {
    /// Time to first token (seconds) — from request start to first token yielded.
    public let ttft: Double

    /// Prefill speed (tokens/second) — prompt processing throughput.
    public let prefillTokensPerSecond: Double

    /// Decode speed (tokens/second) — generation throughput.
    public let decodeTokensPerSecond: Double

    /// Number of prompt tokens processed.
    public let promptTokens: Int

    /// Number of tokens generated.
    public let completionTokens: Int

    /// Number of tokens served from cache (0 = full prefill).
    public let cachedTokens: Int

    /// Which cache layer hit (nil = miss).
    public let cacheDetail: String?

    /// Formatted summary for display in the chat bubble.
    public var summary: String {
        var parts: [String] = []

        // TTFT
        if ttft < 1.0 {
            parts.append(String(format: "TTFT %.0fms", ttft * 1000))
        } else {
            parts.append(String(format: "TTFT %.1fs", ttft))
        }

        // Prefill speed (only show if meaningful)
        if prefillTokensPerSecond > 0 && promptTokens > 0 {
            parts.append(String(format: "PP %.0f t/s", prefillTokensPerSecond))
        }

        // Decode speed
        if decodeTokensPerSecond > 0 {
            parts.append(String(format: "TG %.1f t/s", decodeTokensPerSecond))
        }

        // Token counts
        let tokenInfo = "\(completionTokens) tok"
        if cachedTokens > 0 {
            parts.append("\(tokenInfo) (\(cachedTokens) cached)")
        } else {
            parts.append(tokenInfo)
        }

        return parts.joined(separator: " | ")
    }

    public init(
        ttft: Double = 0,
        prefillTokensPerSecond: Double = 0,
        decodeTokensPerSecond: Double = 0,
        promptTokens: Int = 0,
        completionTokens: Int = 0,
        cachedTokens: Int = 0,
        cacheDetail: String? = nil
    ) {
        self.ttft = ttft
        self.prefillTokensPerSecond = prefillTokensPerSecond
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.cachedTokens = cachedTokens
        self.cacheDetail = cacheDetail
    }
}
