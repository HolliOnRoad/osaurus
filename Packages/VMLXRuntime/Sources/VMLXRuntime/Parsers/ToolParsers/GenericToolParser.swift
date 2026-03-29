import Foundation

/// Generic JSON-based tool call parser. Works as fallback for any model.
/// Detects JSON objects containing {"name": "...", "arguments": {...}} patterns.
public struct GenericToolParser: ToolCallParser {

    public static var supportedModels: [String] { ["generic", "default"] }

    private var buffer: String = ""
    private var inJSON: Bool = false
    private var braceDepth: Int = 0

    public init() {}

    public mutating func processChunk(_ text: String) -> [ToolParserResult] {
        var results: [ToolParserResult] = []

        for char in text {
            if !inJSON {
                if char == "{" {
                    inJSON = true
                    braceDepth = 1
                    buffer = "{"
                } else {
                    results.append(.text(String(char)))
                }
            } else {
                buffer.append(char)
                if char == "{" { braceDepth += 1 }
                if char == "}" {
                    braceDepth -= 1
                    if braceDepth == 0 {
                        // Complete JSON object
                        if let toolCall = _tryParseToolCall(buffer) {
                            results.append(.toolCall(toolCall))
                        } else {
                            // Not a tool call, emit as text
                            results.append(.text(buffer))
                        }
                        buffer = ""
                        inJSON = false
                    }
                }
            }
        }

        if inJSON {
            results.append(.buffered)
        }

        return results
    }

    public mutating func finalize() -> [ParsedToolCall] {
        // If there's buffered content, try to parse it
        if !buffer.isEmpty {
            if let toolCall = _tryParseToolCall(buffer) {
                buffer = ""
                inJSON = false
                return [toolCall]
            }
        }
        buffer = ""
        inJSON = false
        return []
    }

    public mutating func reset() {
        buffer = ""
        inJSON = false
        braceDepth = 0
    }

    // MARK: - Private

    private func _tryParseToolCall(_ json: String) -> ParsedToolCall? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String else {
            return nil
        }

        // Arguments can be a dict or a string
        let argsJSON: String
        if let argsDict = obj["arguments"] as? [String: Any],
           let argsData = try? JSONSerialization.data(withJSONObject: argsDict),
           let argsStr = String(data: argsData, encoding: .utf8) {
            argsJSON = argsStr
        } else if let argsStr = obj["arguments"] as? String {
            argsJSON = argsStr
        } else {
            argsJSON = "{}"
        }

        let id = obj["id"] as? String ?? ""
        return ParsedToolCall(name: name, argumentsJSON: argsJSON, id: id)
    }
}
