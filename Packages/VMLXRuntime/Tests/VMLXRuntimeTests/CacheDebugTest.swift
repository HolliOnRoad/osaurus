import Testing
import Foundation
import MLX
import MLXNN
@testable import VMLXRuntime

@Suite("Speed Check")
struct SpeedCheckTest {
    @Test("Graph vs compute timing")
    func graphVsCompute() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent("jang/models/MiniMax-M2.5-JANG_2L")
        guard FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path) else {
            print("SKIP"); return
        }
        let loaded = try await ModelLoader.load(from: path)
        let container = VMLXModelContainer.create(model: loaded)
        let cache = container.newCache()
        let input = MLXArray([Int32(1)]).reshaped(1, 1)
        let pl = container.forward(input, cache: cache)
        var y = pl[0,-1].argMax()
        MLX.eval(y)
        for _ in 0..<5 {
            let tok = y.item(Int.self)
            let d = container.forward(MLXArray([Int32(tok)]).reshaped(1,1), cache: cache)
            y = d[0,-1].argMax()
            MLX.eval(y)
        }

        var graphTimes: [Double] = []
        var evalTimes: [Double] = []
        for _ in 0..<10 {
            let tok = y.item(Int.self)
            let t0 = CFAbsoluteTimeGetCurrent()
            let d = container.forward(MLXArray([Int32(tok)]).reshaped(1,1), cache: cache)
            let yNew = d[0,-1].argMax()
            let t1 = CFAbsoluteTimeGetCurrent()
            MLX.eval(yNew)
            let t2 = CFAbsoluteTimeGetCurrent()
            y = yNew
            graphTimes.append(t1-t0)
            evalTimes.append(t2-t1)
        }

        let avgGraph = graphTimes.reduce(0,+)/10*1000
        let avgEval = evalTimes.reduce(0,+)/10*1000
        print("Swift graph construction: \(String(format: "%.2f", avgGraph))ms")
        print("Swift GPU compute (eval): \(String(format: "%.2f", avgEval))ms")
        print("Swift total: \(String(format: "%.2f", avgGraph+avgEval))ms = \(String(format: "%.1f", 1000/(avgGraph+avgEval))) tok/s")
    }
}
