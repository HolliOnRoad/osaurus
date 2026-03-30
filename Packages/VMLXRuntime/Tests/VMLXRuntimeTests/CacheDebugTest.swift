import Testing
import Foundation
import MLX
@testable import VMLXRuntime

@Suite("Cache Debug")
struct CacheDebugTest {
    @Test("Profile single decode step")
    func singleStep() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent(".cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/08231374eeacb049a0eade7922910865b8fce912")
        guard FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path) else {
            print("SKIP"); return
        }
        let loaded = try await ModelLoader.load(from: path)
        let model = loaded.nativeModel as! StandardTransformerModel
        let cache = model.newCache()

        // Prefill 3 tokens
        let input = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)
        let t0 = CFAbsoluteTimeGetCurrent()
        let logits = model(input, cache: cache)
        MLX.eval(logits)
        print("Prefill 3 tokens: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent()-t0)*1000))ms")
        print("Logits shape: \(logits.shape)")

        // Single decode
        let t1 = CFAbsoluteTimeGetCurrent()
        let decode1 = model(MLXArray([Int32(4)]).reshaped(1, 1), cache: cache)
        MLX.eval(decode1)
        print("Decode 1 token: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent()-t1)*1000))ms")
        print("Decode logits shape: \(decode1.shape)")

        // Another decode
        let t2 = CFAbsoluteTimeGetCurrent()
        let decode2 = model(MLXArray([Int32(5)]).reshaped(1, 1), cache: cache)
        MLX.eval(decode2)
        print("Decode 1 more: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent()-t2)*1000))ms")

        // 10 decodes in a row
        let t3 = CFAbsoluteTimeGetCurrent()
        for i in 0..<10 {
            let d = model(MLXArray([Int32(6+i)]).reshaped(1, 1), cache: cache)
            MLX.eval(d)
        }
        let avg = (CFAbsoluteTimeGetCurrent()-t3)/10*1000
        print("Avg decode (10 tokens): \(String(format: "%.1f", avg))ms = \(String(format: "%.1f", 1000/avg)) tok/s")
    }
}

    @Test("MiniMax direct speed")
    func miniMaxDirect() async throws {
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
        let pl = container.forward(input, cache: cache)
        MLX.eval(pl)
        // Warmup
        var y = pl[0,-1].argMax()
        for _ in 0..<5 {
            let d = container.forward(y.reshaped(1,1), cache: cache)
            y = d[0,-1].argMax()
            MLX.eval(y)
        }
        // Measure
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<20 {
            let d = container.forward(y.reshaped(1,1), cache: cache)
            y = d[0,-1].argMax()
            MLX.eval(y)
        }
        let avg = (CFAbsoluteTimeGetCurrent()-t0)/20*1000
        print("MiniMax DIRECT: \(String(format: "%.1f", avg))ms = \(String(format: "%.1f", 1000/avg)) tok/s")
    }

    @Test("Qwen3.5 4B direct speed")
    func qwen4BDirect() async throws {
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
        let pl = container.forward(input, cache: cache)
        MLX.eval(pl)
        var y = pl[0,-1].argMax()
        for _ in 0..<5 {
            let d = container.forward(y.reshaped(1,1), cache: cache)
            y = d[0,-1].argMax()
            MLX.eval(y)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<20 {
            let d = container.forward(y.reshaped(1,1), cache: cache)
            y = d[0,-1].argMax()
            MLX.eval(y)
        }
        let avg = (CFAbsoluteTimeGetCurrent()-t0)/20*1000
        print("Qwen3.5 4B DIRECT: \(String(format: "%.1f", avg))ms = \(String(format: "%.1f", 1000/avg)) tok/s")
    }
}
