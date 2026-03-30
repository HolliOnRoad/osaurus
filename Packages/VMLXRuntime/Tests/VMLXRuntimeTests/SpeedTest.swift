import Testing
import Foundation
import MLX
@testable import VMLXRuntime

@Suite("Speed")
struct SpeedTest {
    @Test("MiniMax raw forward speed")
    func rawSpeed() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent("jang/models/MiniMax-M2.5-JANG_2L")
        guard FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path) else {
            print("SKIP"); return
        }

        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)
        let cache = container.newCache()

        let msgs = [VMLXChatMessage(role: "user", content: "Hi")]
        let tokenIds = try container.applyChatTemplate(messages: msgs, addGenerationPrompt: true, enableThinking: true)
        let input = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, tokenIds.count)

        // Prefill
        let prefillLogits = container.forward(input, cache: cache)
        var y = prefillLogits[0, -1].argMax()
        asyncEval([y])

        // Warmup 5 tokens
        for _ in 0..<5 {
            let logits = container.forward(y.reshaped(1, 1), cache: cache)
            let nextY = logits[0, -1].argMax()
            asyncEval([nextY])
            _ = y.item(Int.self)
            y = nextY
        }

        // Measure 30 tokens — PURE forward+sample, NO decode/string/accumulator
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<30 {
            let logits = container.forward(y.reshaped(1, 1), cache: cache)
            let nextY = logits[0, -1].argMax()
            asyncEval([nextY])
            _ = y.item(Int.self)
            y = nextY
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        let tps = 30.0 / elapsed
        print("RAW SPEED: \(String(format: "%.1f", tps)) tok/s (\(String(format: "%.1f", elapsed/30*1000))ms/token)")
    }
}
