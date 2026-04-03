//
//  VMLXSSEParser.swift
//  osaurus
//
//  Parses Server-Sent Events from the vmlx Python engine's
//  OpenAI-compatible /v1/chat/completions streaming endpoint.
//

import Foundation

/// Usage stats from an SSE chunk.
struct VMLXUsage: Sendable {
    var promptTokens: Int = 0
    var completionTokens: Int = 0
    var totalTokens: Int = 0
    var cachedTokens: Int = 0
    var cacheDetail: String?
}

/// A single parsed SSE chunk from the vmlx engine.
struct VMLXSSEChunk: Sendable {
    /// Text content delta (nil if this chunk has no content)
    var content: String?
    /// Reasoning/thinking content delta (nil if no reasoning)
    var reasoningContent: String?
    /// Tool calls array (nil if no tool calls in this chunk)
    var toolCalls: [VMLXToolCallDelta]?
    /// Finish reason: "stop", "length", "tool_calls", or nil if streaming
    var finishReason: String?
    /// Whether this is the [DONE] sentinel
    var isDone: Bool = false
    /// Usage stats (present when stream_options.include_usage is true)
    var usage: VMLXUsage?
}

/// A tool call delta from a streaming chunk.
/// First chunk has id + name + partial args. Continuation chunks have only index + args.
struct VMLXToolCallDelta: Sendable {
    let index: Int
    let id: String        // Empty string for continuation chunks
    let functionName: String  // Empty string for continuation chunks
    let arguments: String
}

/// Parses SSE lines from the vmlx engine into structured chunks.
///
/// SSE format from the engine:
/// ```
/// data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":123,"model":"m","choices":[{"index":0,"delta":{"content":"hi","reasoning_content":"think"},"finish_reason":null}]}
/// data: [DONE]
/// ```
enum VMLXSSEParser {

    /// Parse a single SSE line (including the "data: " prefix).
    /// Returns nil for empty lines, comments, or unparseable data.
    static func parse(line: String) -> VMLXSSEChunk? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)

        // Skip empty lines and SSE comments (keep-alive)
        guard !trimmed.isEmpty, !trimmed.hasPrefix(":") else { return nil }

        // Must start with "data: "
        guard trimmed.hasPrefix("data: ") else { return nil }

        let payload = String(trimmed.dropFirst(6))

        // Check for [DONE] sentinel
        if payload == "[DONE]" {
            return VMLXSSEChunk(isDone: true)
        }

        // Parse JSON
        guard let data = payload.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let firstChoice = choices.first else {
            return nil
        }

        var chunk = VMLXSSEChunk()

        // Extract finish_reason
        if let reason = firstChoice["finish_reason"] as? String {
            chunk.finishReason = reason
        }

        // Extract delta
        guard let delta = firstChoice["delta"] as? [String: Any] else {
            return chunk
        }

        // Content
        if let content = delta["content"] as? String {
            chunk.content = content
        }

        // Reasoning content (OpenAI O1-style)
        if let reasoning = delta["reasoning_content"] as? String {
            chunk.reasoningContent = reasoning
        }

        // Tool calls — first chunk has id+name, continuations have only index+args
        if let toolCallsArray = delta["tool_calls"] as? [[String: Any]] {
            chunk.toolCalls = toolCallsArray.compactMap { tc in
                guard let index = tc["index"] as? Int else { return nil }
                let id = tc["id"] as? String ?? ""
                let function = tc["function"] as? [String: Any]
                let name = function?["name"] as? String ?? ""
                let arguments = function?["arguments"] as? String ?? ""
                return VMLXToolCallDelta(index: index, id: id, functionName: name, arguments: arguments)
            }
        }

        // Usage stats (top-level, not inside delta)
        if let usageDict = json["usage"] as? [String: Any] {
            var usage = VMLXUsage()
            usage.promptTokens = usageDict["prompt_tokens"] as? Int ?? 0
            usage.completionTokens = usageDict["completion_tokens"] as? Int ?? 0
            usage.totalTokens = usageDict["total_tokens"] as? Int ?? 0
            if let details = usageDict["prompt_tokens_details"] as? [String: Any] {
                usage.cachedTokens = details["cached_tokens"] as? Int ?? 0
                usage.cacheDetail = details["cache_detail"] as? String
            }
            chunk.usage = usage
        }

        return chunk
    }

    /// Parse a multi-line SSE buffer into chunks.
    /// SSE events are separated by double newlines; each "data:" line is one event.
    static func parseBuffer(_ buffer: String) -> [VMLXSSEChunk] {
        var chunks: [VMLXSSEChunk] = []
        for line in buffer.components(separatedBy: "\n") {
            if let chunk = parse(line: line) {
                chunks.append(chunk)
            }
        }
        return chunks
    }
}
