import Foundation

/// Result from processing a token through a tool call parser.
public enum ToolParserResult: Sendable {
    /// Token is normal text content, pass through.
    case text(String)
    /// Token is being buffered (potential tool call in progress).
    case buffered
    /// Complete tool call detected.
    case toolCall(ParsedToolCall)
}

/// A parsed tool/function call extracted from model output.
public struct ParsedToolCall: Sendable {
    public let name: String
    public let argumentsJSON: String
    public let id: String  // Call ID for multi-turn

    public init(name: String, argumentsJSON: String, id: String = "") {
        self.name = name
        self.argumentsJSON = argumentsJSON
        self.id = id.isEmpty ? "call_\(UUID().uuidString.prefix(8))" : id
    }
}

/// Protocol for model-specific tool call parsers.
/// Each model family has its own format for expressing tool calls.
public protocol ToolCallParser: Sendable {
    /// Model families this parser handles (for auto-detection).
    static var supportedModels: [String] { get }

    /// Process a text chunk. Returns how to handle it.
    mutating func processChunk(_ text: String) -> [ToolParserResult]

    /// Finalize: check if any buffered content forms a complete tool call.
    mutating func finalize() -> [ParsedToolCall]

    /// Reset state for next generation.
    mutating func reset()
}

/// Auto-detect the appropriate tool parser for a model.
/// Uses a factory registry so new parser types can be added without type casts.
public func autoDetectToolParser(modelName: String) -> (any ToolCallParser)? {
    let name = modelName.lowercased()

    // (patterns, factory) pairs, ordered by specificity.
    // More specific model families should appear before generic fallbacks.
    let registry: [(patterns: [String], factory: () -> any ToolCallParser)] = [
        // Future: add model-specific parsers here first
        // (["qwen"], { QwenToolParser() }),
        // (["llama"], { LlamaToolParser() }),
        (["generic", "default"], { GenericToolParser() }),
    ]

    for entry in registry {
        for pattern in entry.patterns {
            if name.contains(pattern) {
                return entry.factory()
            }
        }
    }

    // Default: generic JSON parser works for most models
    return GenericToolParser()
}
