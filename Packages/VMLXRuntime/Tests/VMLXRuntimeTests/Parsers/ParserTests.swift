import Testing
@testable import VMLXRuntime

@Suite("GenericToolParser")
struct GenericToolParserTests {

    @Test("Detects JSON tool call")
    func detectsToolCall() {
        var parser = GenericToolParser()
        let results = parser.processChunk("""
            {"name": "get_weather", "arguments": {"location": "NYC"}}
            """)

        let toolCalls = results.compactMap { result -> ParsedToolCall? in
            if case .toolCall(let tc) = result { return tc }
            return nil
        }
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].name == "get_weather")
    }

    @Test("Passes through non-tool text")
    func passesThrough() {
        var parser = GenericToolParser()
        let results = parser.processChunk("Hello world!")
        let texts = results.compactMap { r -> String? in
            if case .text(let t) = r { return t }
            return nil
        }
        #expect(texts.joined() == "Hello world!")
    }

    @Test("Buffers incomplete JSON")
    func buffersIncomplete() {
        var parser = GenericToolParser()
        let r1 = parser.processChunk("{\"name\": \"test\"")
        let hasBuffered = r1.contains { if case .buffered = $0 { return true }; return false }
        #expect(hasBuffered)

        let r2 = parser.processChunk(", \"arguments\": {}}")
        let toolCalls = r2.compactMap { r -> ParsedToolCall? in
            if case .toolCall(let tc) = r { return tc }
            return nil
        }
        #expect(toolCalls.count == 1)
    }

    @Test("Reset clears state")
    func reset() {
        var parser = GenericToolParser()
        _ = parser.processChunk("{incomplete")
        parser.reset()
        let results = parser.processChunk("plain text")
        let texts = results.compactMap { r -> String? in
            if case .text(let t) = r { return t }
            return nil
        }
        #expect(texts.joined() == "plain text")
    }
}

@Suite("ThinkTagReasoningParser")
struct ThinkTagReasoningParserTests {

    @Test("Extracts thinking block")
    func extractsThinking() {
        var parser = ThinkTagReasoningParser()
        let r = parser.processChunk("<think>I need to calculate...</think>The answer is 42.")
        #expect(r.reasoning != nil)
        #expect(r.inThinking == false)
    }

    @Test("Streaming think tags")
    func streamingThink() {
        var parser = ThinkTagReasoningParser()
        let r1 = parser.processChunk("<think>thinking")
        #expect(r1.inThinking == true)
        #expect(r1.reasoning != nil)

        let r2 = parser.processChunk(" more</think>response")
        #expect(r2.inThinking == false)
        #expect(r2.reasoning != nil)
    }

    @Test("No think tags passes content through")
    func noThinkTags() {
        var parser = ThinkTagReasoningParser()
        let r = parser.processChunk("Just a normal response")
        #expect(r.content == "Just a normal response")
        #expect(r.reasoning == nil)
        #expect(r.inThinking == false)
    }

    @Test("Reset clears thinking state")
    func reset() {
        var parser = ThinkTagReasoningParser()
        _ = parser.processChunk("<think>partial")
        parser.reset()
        let r = parser.processChunk("fresh start")
        #expect(r.inThinking == false)
        #expect(r.content == "fresh start")
    }

    @Test("Finalize flushes remaining")
    func finalize() {
        var parser = ThinkTagReasoningParser()
        // First chunk opens the thinking block and flushes content
        _ = parser.processChunk("<think>start")
        // Second chunk has content ending with partial close tag, leaving buffer non-empty
        _ = parser.processChunk(" more</thi")
        let r = parser.finalize()
        // Finalize should flush buffered partial content as reasoning and reset state
        #expect(r.reasoning != nil)
        #expect(r.inThinking == false)
    }

    @Test("Auto-detect finds parser for Qwen3")
    func autoDetectQwen3() {
        let parser = autoDetectReasoningParser(modelName: "Qwen3-8B-JANG")
        #expect(parser != nil)
    }
}
