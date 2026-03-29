import Testing
import Foundation
@testable import VMLXRuntime

@Suite("VMLXRuntimeActor")
struct VMLXRuntimeActorTests {

    @Test("Initial state: no model loaded")
    func initialState() async {
        let runtime = VMLXRuntimeActor()
        let loaded = await runtime.isModelLoaded
        #expect(!loaded)
        let name = await runtime.currentModelName
        #expect(name == nil)
    }

    @Test("Load model sets name")
    func loadModel() async throws {
        let runtime = VMLXRuntimeActor()
        try await runtime.loadModel(name: "test-model")
        let loaded = await runtime.isModelLoaded
        #expect(loaded)
        let name = await runtime.currentModelName
        #expect(name == "test-model")
    }

    @Test("Unload clears state")
    func unloadModel() async throws {
        let runtime = VMLXRuntimeActor()
        try await runtime.loadModel(name: "test-model")
        await runtime.unloadModel()
        let loaded = await runtime.isModelLoaded
        #expect(!loaded)
    }

    @Test("Load with hybrid flag")
    func hybridModel() async throws {
        let runtime = VMLXRuntimeActor()
        let tqConfig = TurboQuantConfig(defaultKeyBits: 3)
        try await runtime.loadModel(name: "nemotron-h", isHybrid: true, turboQuant: tqConfig)
        let loaded = await runtime.isModelLoaded
        #expect(loaded)
    }

    @Test("Generate without model throws")
    func generateNoModel() async {
        let runtime = VMLXRuntimeActor()
        let request = VMLXChatCompletionRequest(
            messages: [VMLXChatMessage(role: "user", content: "Hi")]
        )
        do {
            _ = try await runtime.generateStream(request: request)
            Issue.record("Expected error")
        } catch {
            // Expected: noModelLoaded
        }
    }

    @Test("Error types have descriptions")
    func errorDescriptions() {
        let errors: [VMLXRuntimeError] = [
            .noModelLoaded,
            .modelLoadFailed("test"),
            .generationFailed("test"),
            .cacheCorruption("test"),
            .tokenizationFailed
        ]
        for error in errors {
            #expect(error.errorDescription != nil)
        }
    }

    @Test("Cache stats accessible")
    func cacheStats() async {
        let runtime = VMLXRuntimeActor()
        let stats = await runtime.cacheStats
        #expect(stats.memoryCacheHits == 0)
    }

    @Test("Auto-detect config applied")
    func autoDetectConfig() async {
        let runtime = VMLXRuntimeActor()
        let config = await runtime.config
        #expect(config.prefillStepSize >= 1024)
    }

    @Test("Reload replaces previous model")
    func reloadModel() async throws {
        let runtime = VMLXRuntimeActor()
        try await runtime.loadModel(name: "model-a")
        try await runtime.loadModel(name: "model-b")
        let name = await runtime.currentModelName
        #expect(name == "model-b")
    }

    @Test("Clear cache does not crash when no model loaded")
    func clearCacheNoModel() async {
        let runtime = VMLXRuntimeActor()
        await runtime.clearCache()
        // No crash = pass
    }

    @Test("Generate stream emits usage after model loaded")
    func generateStreamUsage() async throws {
        let runtime = VMLXRuntimeActor()
        try await runtime.loadModel(name: "test-model")
        let request = VMLXChatCompletionRequest(
            messages: [VMLXChatMessage(role: "user", content: "Hello")]
        )
        let stream = try await runtime.generateStream(request: request)
        var gotUsage = false
        for try await event in stream {
            if case .usage = event {
                gotUsage = true
            }
        }
        #expect(gotUsage)
    }

    @Test("Non-streaming generate returns string")
    func nonStreamingGenerate() async throws {
        let runtime = VMLXRuntimeActor()
        try await runtime.loadModel(name: "test-model")
        let request = VMLXChatCompletionRequest(
            messages: [VMLXChatMessage(role: "user", content: "Hi")]
        )
        // With placeholder generation, result is empty but should not throw
        let result = try await runtime.generate(request: request)
        #expect(result.isEmpty)  // No actual model loaded, so no tokens generated
    }
}
