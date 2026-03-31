//
//  StreamingMiddleware.swift
//  osaurus
//
//  Transforms raw streaming deltas before they reach StreamingDeltaProcessor's
//  tag parser. Model-specific streaming behavior lives here, keeping the
//  processor itself model-agnostic.
//

/// Transforms raw streaming deltas before they reach the tag parser.
/// Stateful — create a new instance per streaming session.
@MainActor
protocol StreamingMiddleware: AnyObject {
    func process(_ delta: String) -> String
}

// MARK: - Middleware Implementations

/// Buffers early deltas for models that emit `</think>` without `<think>`.
/// Only prepends `<think>` if a `</think>` is detected in the first N tokens,
/// confirming the model is actually reasoning. Otherwise, flushes buffered
/// content as-is (no false thinking box).
@MainActor
final class PrependThinkTagMiddleware: StreamingMiddleware {
    private var state: State = .buffering
    private var buffer: String = ""
    private var deltaCount = 0
    private static let maxBufferDeltas = 20  // Check first ~20 tokens

    private enum State {
        case buffering   // Accumulating early deltas to check for </think>
        case confirmed   // </think> found — already prepended <think>
        case passthrough // No </think> detected — no thinking, pass through
    }

    func process(_ delta: String) -> String {
        switch state {
        case .confirmed, .passthrough:
            return delta

        case .buffering:
            deltaCount += 1
            buffer += delta

            // Check if </think> appeared — confirms model is reasoning
            if buffer.contains("</think>") {
                state = .confirmed
                let result = "<think>" + buffer
                buffer = ""
                return result
            }

            // If we've buffered enough without seeing </think>, give up
            if deltaCount >= Self.maxBufferDeltas {
                state = .passthrough
                let result = buffer
                buffer = ""
                return result
            }

            // Still buffering — suppress output for now
            return ""
        }
    }
}

// MARK: - Channel Tag Middleware (GPT-OSS)

/// Transforms GPT-OSS `<|channel|>analysis<|message|>` / `<|channel|>reply<|message|>`
/// into standard `<think>` / `</think>` tags that StreamingDeltaProcessor handles.
/// Buffers token-by-token input to detect multi-token channel tags.
@MainActor
final class ChannelTagMiddleware: StreamingMiddleware {
    private var buffer = ""

    private static let analysisTag = "<|channel|>analysis<|message|>"
    private static let replyTag = "<|channel|>reply<|message|>"
    private static let tagPrefix = "<|"

    func process(_ delta: String) -> String {
        buffer += delta

        // Check for complete channel tags and replace with think tags
        var output = ""
        while !buffer.isEmpty {
            if let range = buffer.range(of: Self.analysisTag) {
                // Emit everything before the tag, then emit <think>
                output += buffer[..<range.lowerBound]
                output += "<think>"
                buffer = String(buffer[range.upperBound...])
            } else if let range = buffer.range(of: Self.replyTag) {
                // Emit everything before the tag, then emit </think>
                output += buffer[..<range.lowerBound]
                output += "</think>"
                buffer = String(buffer[range.upperBound...])
            } else if buffer.contains(Self.tagPrefix) {
                // Might be a partial channel tag — check if it could complete
                if let prefixRange = buffer.range(of: Self.tagPrefix, options: .backwards) {
                    let suffix = String(buffer[prefixRange.lowerBound...])
                    // Check if either full tag starts with this suffix
                    if Self.analysisTag.hasPrefix(suffix) || Self.replyTag.hasPrefix(suffix) {
                        // Partial tag — emit everything before it, keep suffix buffered
                        output += buffer[..<prefixRange.lowerBound]
                        buffer = suffix
                        break
                    }
                }
                // Not a channel tag prefix — emit all
                output += buffer
                buffer = ""
            } else {
                // No channel tag markers — emit all
                output += buffer
                buffer = ""
            }
        }

        return output
    }
}

// MARK: - Resolver

enum StreamingMiddlewareResolver {
    @MainActor
    static func resolve(
        for modelId: String,
        modelOptions: [String: ModelOptionValue] = [:]
    ) -> StreamingMiddleware? {
        let thinkingDisabled = modelOptions["disableThinking"]?.boolValue == true
        let id = modelId.lowercased()

        // GPT-OSS: transform <|channel|>analysis/reply tags to <think>/</ think>
        if !thinkingDisabled && (id.contains("gpt-oss") || id.contains("gpt_oss")) {
            return ChannelTagMiddleware()
        }

        // PrependThinkTagMiddleware: for models that output </think> but NOT <think>
        let needsPrependThink =
            !thinkingDisabled
            && (id.contains("glm") && id.contains("flash"))

        return needsPrependThink ? PrependThinkTagMiddleware() : nil
    }

    /// Matches parameter-count tokens like "4b" while ignoring
    /// quantization suffixes like "4bit" that share a prefix.
    private static func hasParamSize(_ id: String, anyOf sizes: String...) -> Bool {
        sizes.contains { id.range(of: "\($0)(?!it)", options: .regularExpression) != nil }
    }
}
