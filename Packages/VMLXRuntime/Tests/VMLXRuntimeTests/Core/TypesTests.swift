import Testing
import MLX
@testable import VMLXRuntime

@Suite("Core Types")
struct TypesTests {
    @Test("Request defaults")
    func requestDefaults() {
        let req = InferenceRequest(requestId: "test-1", promptTokenIds: [1, 2, 3, 4, 5])
        #expect(req.status == .waiting)
        #expect(req.numPromptTokens == 5)
        #expect(req.outputTokenIds.isEmpty)
        #expect(req.isFinished == false)
        #expect(req.samplingParams.temperature == 0.7)
    }

    @Test("Request lifecycle")
    func requestLifecycle() {
        var req = InferenceRequest(requestId: "test-2", promptTokenIds: [1, 2, 3])
        req.status = .running
        req.appendOutputToken(100)
        req.appendOutputToken(101)
        #expect(req.numOutputTokens == 2)
        req.finish(reason: .stop)
        #expect(req.isFinished)
        #expect(req.finishReason == .stop)
    }

    @Test("SamplingParams greedy")
    func samplingGreedy() {
        let p = SamplingParams(
            maxTokens: 1024,
            temperature: 0.0,
            topP: 0.95,
            topK: 50,
            minP: 0.05,
            repetitionPenalty: 1.1
        )
        #expect(p.isGreedy)
    }

    @Test("RequestOutput")
    func requestOutput() {
        let o = RequestOutput(requestId: "t", newTokenIds: [1, 2, 3], newText: "hi")
        #expect(o.newTokenIds.count == 3)
        #expect(o.finishReason == nil)
    }

    @Test("RequestStatus comparison")
    func statusComparison() {
        #expect(RequestStatus.waiting < RequestStatus.running)
        #expect(RequestStatus.finishedStopped.isFinished)
        #expect(!RequestStatus.running.isFinished)
    }
}
