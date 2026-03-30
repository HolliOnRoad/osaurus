import Testing
import Foundation
import MLX
@testable import VMLXRuntime

@Suite("Speed")
struct SpeedTest {
    @Test("Qwen 3.5 4B speed")
    func qwen4BSpeed() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent("jang/models/Qwen3.5-4B-JANG_2S")
        guard FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path) else {
            print("SKIP"); return
        }
        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)
        let cache = container.newCache()
        let msgs = [VMLXChatMessage(role: "user", content: "Hi")]
        let tokenIds = try container.applyChatTemplate(messages: msgs, addGenerationPrompt: true, enableThinking: true)
        let input = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, tokenIds.count)
        let prefillLogits = container.forward(input, cache: cache)
        var y = prefillLogits[0, -1].argMax()
        MLX.eval(y)
        for _ in 0..<5 {
            let logits = container.forward(y.reshaped(1, 1), cache: cache)
            y = logits[0, -1].argMax()
            MLX.eval(y)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<30 {
            let logits = container.forward(y.reshaped(1, 1), cache: cache)
            let ny = logits[0, -1].argMax()
            asyncEval([ny])
            _ = y.item(Int.self)
            y = ny
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print("Qwen3.5 4B SPEED: \(String(format: "%.1f", 30/elapsed)) tok/s (\(String(format: "%.1f", elapsed/30*1000))ms/token)")
    }

    @Test("MiniMax speed")
    func miniMaxSpeed() async throws {
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
        let prefillLogits = container.forward(input, cache: cache)
        var y = prefillLogits[0, -1].argMax()
        MLX.eval(y)
        for _ in 0..<5 {
            let logits = container.forward(y.reshaped(1, 1), cache: cache)
            y = logits[0, -1].argMax()
            MLX.eval(y)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<30 {
            let logits = container.forward(y.reshaped(1, 1), cache: cache)
            let ny = logits[0, -1].argMax()
            asyncEval([ny])
            _ = y.item(Int.self)
            y = ny
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print("MiniMax SPEED: \(String(format: "%.1f", 30/elapsed)) tok/s (\(String(format: "%.1f", elapsed/30*1000))ms/token)")
    }

    @Test("Llama 3.2 1B speed")
    func llamaSpeed() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent(".cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/08231374eeacb049a0eade7922910865b8fce912")
        guard FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path) else {
            print("SKIP"); return
        }
        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)
        let cache = container.newCache()
        let input = MLXArray([Int32(128000), Int32(9906), Int32(0)]).reshaped(1, 3)
        let prefillLogits = container.forward(input, cache: cache)
        var y = prefillLogits[0, -1].argMax()
        MLX.eval(y)
        for _ in 0..<5 {
            let logits = container.forward(y.reshaped(1, 1), cache: cache)
            y = logits[0, -1].argMax()
            MLX.eval(y)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<30 {
            let logits = container.forward(y.reshaped(1, 1), cache: cache)
            let ny = logits[0, -1].argMax()
            asyncEval([ny])
            _ = y.item(Int.self)
            y = ny
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print("Llama 1B SPEED: \(String(format: "%.1f", 30/elapsed)) tok/s (\(String(format: "%.1f", elapsed/30*1000))ms/token)")
    }
}
