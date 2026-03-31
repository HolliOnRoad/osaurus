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

/// Prepends `<think>` to the first non-empty delta for models that only
/// emit `</think>` without the opening tag (e.g. GLM-4.7-flash).
@MainActor
final class PrependThinkTagMiddleware: StreamingMiddleware {
    private var hasFired = false

    func process(_ delta: String) -> String {
        guard !hasFired else { return delta }
        hasFired = true
        return "<think>" + delta
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

        // PrependThinkTagMiddleware is for models that output </think> but NOT <think>.
        // VMLX Qwen3.5 models output <think> natively via the chat template,
        // so they do NOT need the middleware. Only enable for non-VMLX edge cases
        // (e.g., remote GLM-flash API that strips the opening tag).
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
