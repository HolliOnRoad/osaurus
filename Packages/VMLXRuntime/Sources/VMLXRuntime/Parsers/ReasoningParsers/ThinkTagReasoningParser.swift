import Foundation

/// Reasoning parser for models using <think>...</think> tags.
/// Works for Qwen3, DeepSeek-R1, and similar models.
public struct ThinkTagReasoningParser: ReasoningParser {

    public static var supportedModels: [String] { ["qwen3", "deepseek-r1", "qwen2.5"] }

    private var inThinking: Bool = false
    private var buffer: String = ""

    // Tag detection
    private static let openTag = "<think>"
    private static let closeTag = "</think>"

    public init() {}

    public mutating func processChunk(_ text: String) -> ReasoningResult {
        buffer += text

        if !inThinking {
            // Look for opening tag
            if let range = buffer.range(of: Self.openTag) {
                let beforeTag = String(buffer[buffer.startIndex..<range.lowerBound])
                buffer = String(buffer[range.upperBound...])
                inThinking = true

                // Check if close tag is also in remaining buffer
                if let closeRange = buffer.range(of: Self.closeTag) {
                    let thinking = String(buffer[buffer.startIndex..<closeRange.lowerBound])
                    buffer = String(buffer[closeRange.upperBound...])
                    inThinking = false
                    return ReasoningResult(
                        reasoning: thinking.isEmpty ? nil : thinking,
                        content: beforeTag.isEmpty ? (buffer.isEmpty ? nil : buffer) : beforeTag,
                        inThinking: false
                    )
                }

                // Still inside thinking
                let thinking = buffer
                buffer = ""
                return ReasoningResult(
                    reasoning: thinking.isEmpty ? nil : thinking,
                    content: beforeTag.isEmpty ? nil : beforeTag,
                    inThinking: true
                )
            }

            // Check for partial tag at end of buffer
            let partialLen = _partialTagMatchLength(Self.openTag)
            if partialLen > 0 {
                let safeEnd = buffer.index(buffer.endIndex, offsetBy: -partialLen)
                let safe = String(buffer[buffer.startIndex..<safeEnd])
                buffer = String(buffer[safeEnd...])
                return ReasoningResult(reasoning: nil, content: safe.isEmpty ? nil : safe, inThinking: false)
            }

            // No tag activity
            let content = buffer
            buffer = ""
            return ReasoningResult(reasoning: nil, content: content.isEmpty ? nil : content, inThinking: false)

        } else {
            // Inside thinking block — look for close tag
            if let closeRange = buffer.range(of: Self.closeTag) {
                let thinking = String(buffer[buffer.startIndex..<closeRange.lowerBound])
                buffer = String(buffer[closeRange.upperBound...])
                inThinking = false
                return ReasoningResult(
                    reasoning: thinking.isEmpty ? nil : thinking,
                    content: buffer.isEmpty ? nil : buffer,
                    inThinking: false
                )
            }

            // Check for partial close tag
            let partialLen = _partialTagMatchLength(Self.closeTag)
            if partialLen > 0 {
                let safeEnd = buffer.index(buffer.endIndex, offsetBy: -partialLen)
                let safe = String(buffer[buffer.startIndex..<safeEnd])
                buffer = String(buffer[safeEnd...])
                return ReasoningResult(reasoning: safe.isEmpty ? nil : safe, content: nil, inThinking: true)
            }

            // No close tag — all is reasoning
            let thinking = buffer
            buffer = ""
            return ReasoningResult(reasoning: thinking.isEmpty ? nil : thinking, content: nil, inThinking: true)
        }
    }

    public mutating func finalize() -> ReasoningResult {
        let remaining = buffer
        buffer = ""
        if inThinking {
            inThinking = false
            return ReasoningResult(reasoning: remaining.isEmpty ? nil : remaining, content: nil, inThinking: false)
        }
        return ReasoningResult(reasoning: nil, content: remaining.isEmpty ? nil : remaining, inThinking: false)
    }

    public mutating func reset() {
        buffer = ""
        inThinking = false
    }

    // MARK: - Private

    private func _partialTagMatchLength(_ tag: String) -> Int {
        for len in (1..<tag.count).reversed() {
            let tagPrefix = String(tag.prefix(len))
            if buffer.hasSuffix(tagPrefix) {
                return len
            }
        }
        return 0
    }
}
