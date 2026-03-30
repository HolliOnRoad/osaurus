import Testing
import Foundation
import MLX
@testable import VMLXRuntime

@Suite("Thinking Toggle")
struct ThinkingToggleTest {
    @Test("Thinking OFF produces no think tags")
    func thinkingOff() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent("jang/models/Qwen3.5-4B-JANG_2S")
        guard FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path) else {
            print("SKIP"); return
        }

        let service = VMLXService()
        try await service.loadModel(from: path)

        let params = SamplingParams(maxTokens: 30, temperature: 0, topP: 1.0, enableThinking: false)
        let stream = try await service.streamDeltas(
            messages: [VMLXChatMessage(role: "user", content: "What is 2+2?")],
            params: params, requestedModel: nil, stopSequences: [])

        var output = ""
        for try await delta in stream { output += delta }

        print("THINKING OFF output: '\(output.prefix(100))'")
        print("Contains <think>: \(output.contains("<think>"))")
        #expect(!output.contains("<think>"), "Output should not contain <think> when thinking is off")

        await service.unloadModel()
    }

    @Test("Thinking ON starts with think tag")
    func thinkingOn() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent("jang/models/Qwen3.5-4B-JANG_2S")
        guard FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path) else {
            print("SKIP"); return
        }

        let service = VMLXService()
        try await service.loadModel(from: path)

        let params = SamplingParams(maxTokens: 20, temperature: 0, topP: 1.0, enableThinking: true)
        let stream = try await service.streamDeltas(
            messages: [VMLXChatMessage(role: "user", content: "What is 2+2?")],
            params: params, requestedModel: nil, stopSequences: [])

        var output = ""
        for try await delta in stream { output += delta }

        print("THINKING ON output: '\(output.prefix(100))'")
        print("Starts with <think>: \(output.hasPrefix("<think>"))")
        #expect(output.hasPrefix("<think>"), "Output should start with <think> when thinking is on")

        await service.unloadModel()
    }
}
