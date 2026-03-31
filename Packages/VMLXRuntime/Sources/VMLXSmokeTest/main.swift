import Foundation
import MLX
import MLXNN
import VMLXRuntime

// ============================================================
// VMLXRuntime Smoke Test — Automated GPU Integration Test
// ============================================================
// Tests: model detection, loading, forward pass, cache store/fetch,
//        SSM companion, TurboQuant round-trip, gen_prompt_len, KV quant.
//
// Usage: swift run --package-path Packages/VMLXRuntime VMLXSmokeTest
// ============================================================

func log(_ msg: String) { print("[SmokeTest] \(msg)") }
func pass(_ msg: String) { print("[PASS] \(msg)") }
func fail(_ msg: String) { print("[FAIL] \(msg)") }

@main
struct SmokeTest {
    static func main() async {
        let home = FileManager.default.homeDirectoryForCurrentUser
        var modelPath: URL?

        // Find smallest Qwen3.5-4B model
        for suffix in ["4S", "4K", "2S"] {
            let p = home.appendingPathComponent("jang/models/Qwen3.5-4B-JANG_\(suffix)")
            if FileManager.default.fileExists(atPath: p.appendingPathComponent("config.json").path) {
                modelPath = p
                break
            }
        }

        guard let modelPath else {
            fail("No Qwen3.5-4B model found in ~/jang/models/")
            return
        }
        log("Using model: \(modelPath.lastPathComponent)")

        var totalTests = 0
        var passedTests = 0

        // ---- TEST 1: Model Detection ----
        totalTests += 1
        do {
            let detected = try ModelDetector.detect(at: modelPath)
            guard detected.isJang else { fail("T1: not detected as JANG"); return }
            guard detected.modelType == "qwen3_5" else { fail("T1: wrong model_type: \(detected.modelType ?? "nil")"); return }
            guard detected.isHybrid else { fail("T1: not detected as hybrid"); return }
            guard detected.hasSSM else { fail("T1: hasSSM=false"); return }
            pass("T1 Model Detection: \(detected.name), type=\(detected.modelType!), hybrid=\(detected.isHybrid), jang=\(detected.isJang)")
            passedTests += 1
        } catch {
            fail("T1 Model Detection: \(error)")
        }

        // ---- TEST 2: Model Loading ----
        totalTests += 1
        let container: VMLXModelContainer
        do {
            log("T2 Loading model (this may take 5-15s)...")
            let startLoad = CFAbsoluteTimeGetCurrent()
            let loaded = try await ModelLoader.load(from: modelPath)
            let elapsed = CFAbsoluteTimeGetCurrent() - startLoad
            container = VMLXModelContainer.create(model: loaded)

            guard container.isHybrid else { fail("T2: container not hybrid"); return }
            guard loaded.vocabSize > 0 else { fail("T2: vocab=0"); return }
            guard loaded.numLayers == 32 else { fail("T2: layers=\(loaded.numLayers), expected 32"); return }

            pass("T2 Model Load: \(String(format: "%.1fs", elapsed)), vocab=\(loaded.vocabSize), layers=\(loaded.numLayers)")
            passedTests += 1
        } catch {
            fail("T2 Model Load: \(error)")
            return
        }

        // ---- TEST 3: Cache Types ----
        totalTests += 1
        let cache = container.newCache()
        var ssmCount = 0, attnCount = 0
        for c in cache {
            if c is VMLXMambaCache { ssmCount += 1 }
            else if c is VMLXKVCacheSimple { attnCount += 1 }
        }
        if cache.count == 32 && ssmCount == 24 && attnCount == 8 {
            pass("T3 Cache Types: \(cache.count) total, \(ssmCount) SSM + \(attnCount) attention")
            passedTests += 1
        } else {
            fail("T3 Cache Types: count=\(cache.count), ssm=\(ssmCount), attn=\(attnCount)")
        }

        // ---- TEST 4: Forward Pass ----
        totalTests += 1
        do {
            let inputTokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)
            let startFwd = CFAbsoluteTimeGetCurrent()
            let logits = container.forward(inputTokens, cache: cache)
            MLX.eval(logits)
            let elapsed = CFAbsoluteTimeGetCurrent() - startFwd

            let shape = logits.shape
            guard shape.count == 3 else { fail("T4: logits ndim=\(shape.count)"); return }
            guard shape[0] == 1 && shape[1] == 3 else { fail("T4: shape \(shape)"); return }
            guard shape[2] == container.nativeModel.vocabularySize else {
                fail("T4: vocab mismatch \(shape[2]) vs \(container.nativeModel.vocabularySize)"); return
            }

            // Check attention cache populated
            var attnOffset = 0
            for c in cache {
                if let kvc = c as? VMLXKVCacheSimple { attnOffset = kvc.offset; break }
            }
            guard attnOffset == 3 else { fail("T4: attn offset=\(attnOffset), expected 3"); return }

            pass("T4 Forward Pass: \(String(format: "%.2fs", elapsed)), logits \(shape), attn_offset=\(attnOffset)")
            passedTests += 1
        }

        // ---- TEST 5: Decode Step (single token) ----
        totalTests += 1
        do {
            let decodeInput = MLXArray([Int32(100)]).reshaped(1, 1)
            let startDec = CFAbsoluteTimeGetCurrent()
            let decLogits = container.forward(decodeInput, cache: cache)
            MLX.eval(decLogits)
            let elapsed = CFAbsoluteTimeGetCurrent() - startDec

            guard decLogits.shape[1] == 1 else { fail("T5: decode shape \(decLogits.shape)"); return }

            var attnOffset = 0
            for c in cache { if let kvc = c as? VMLXKVCacheSimple { attnOffset = kvc.offset; break } }
            guard attnOffset == 4 else { fail("T5: attn offset=\(attnOffset), expected 4"); return }

            pass("T5 Decode Step: \(String(format: "%.3fs", elapsed)), offset=\(attnOffset)")
            passedTests += 1
        }

        // ---- TEST 6: Cache Store + Fetch Round-Trip ----
        totalTests += 1
        do {
            let coordinator = CacheCoordinator(config: CacheCoordinatorConfig(
                usePagedCache: false, useMemoryAwareCache: true, enableDiskCache: false
            ))
            coordinator.setHybrid(true)

            // Build HybridCache from current state
            var layers: [LayerCacheEntry] = []
            for c in cache {
                if let mc = c as? VMLXMambaCache {
                    layers.append(.ssm(SSMStateLayer(state: mc.state)))
                } else if let kvc = c as? VMLXKVCacheSimple {
                    let s = kvc.state
                    if s.count == 2 {
                        layers.append(.attention(KVCacheLayer(keys: s[0], values: s[1], offset: kvc.offset)))
                    }
                }
            }
            let hybridCache = HybridCache(layers: layers)
            hybridCache.materialized()

            let storeTokens = [1, 2, 3]  // dropLast(1) from [1,2,3,4]
            coordinator.store(tokens: storeTokens, cache: hybridCache)

            let fetchTokens = [1, 2, 3, 4]
            let result = coordinator.fetch(tokens: fetchTokens)
            switch result {
            case .hit(let cached, let remaining, let detail, let ssmCheckpoint):
                guard remaining.count == 1 && remaining[0] == 4 else {
                    fail("T6: remaining=\(remaining)"); return
                }
                guard cached.layerCount == 32 else {
                    fail("T6: layerCount=\(cached.layerCount)"); return
                }
                let hasSsmCp = ssmCheckpoint != nil && !(ssmCheckpoint!.ssmStates.isEmpty)
                pass("T6 Cache Round-Trip: HIT, remaining=\(remaining), detail=\(detail), ssm_cp=\(hasSsmCp)")
                passedTests += 1
            case .partialHit(_, let remaining, _):
                fail("T6: partialHit (SSM companion missing), remaining=\(remaining.count)")
            case .miss:
                fail("T6: MISS")
            }
        }

        // ---- TEST 7: TurboQuant Encode/Decode ----
        totalTests += 1
        do {
            // Find first attention layer with data
            for c in cache {
                guard let kvc = c as? VMLXKVCacheSimple else { continue }
                let s = kvc.state
                guard s.count == 2 else { continue }
                let keys = s[0]
                let values = s[1]
                let origShape = keys.shape

                let ek = TurboQuantEncoder.encodeKeys(keys: keys, bits: 3, seed: 42, sinkTokens: 0)
                let ev = TurboQuantEncoder.encodeValues(values: values, bits: 3, seed: 42, sinkTokens: 0)
                let dk = TurboQuantEncoder.decodeKeys(ek, seed: ek.seed)
                let dv = TurboQuantEncoder.decodeValues(ev, seed: ev.seed)

                guard dk.shape == origShape else { fail("T7: shape mismatch \(dk.shape) vs \(origShape)"); break }
                guard dv.shape == values.shape else { fail("T7: value shape mismatch"); break }

                let keySum = dk.sum().item(Float.self)
                guard keySum.isFinite else { fail("T7: decoded keys not finite"); break }

                let origBytes = keys.nbytes + values.nbytes
                let compBytes = ek.estimatedBytes + ev.estimatedBytes
                let ratio = Float(origBytes) / Float(compBytes)

                pass("T7 TurboQuant: shape \(origShape) preserved, ratio=\(String(format: "%.1fx", ratio)), sum=\(String(format: "%.2f", keySum))")
                passedTests += 1
                break
            }
        }

        // ---- TEST 8: TQ with Sink Tokens ----
        totalTests += 1
        do {
            for c in cache {
                guard let kvc = c as? VMLXKVCacheSimple else { continue }
                let s = kvc.state
                guard s.count == 2 else { continue }
                let keys = s[0]
                let values = s[1]
                let seqLen = keys.dim(2)

                // Encode with sink preservation (sink=2, since we only have 4 tokens)
                let sinkCount = min(2, seqLen)
                let ek = TurboQuantEncoder.encodeKeys(keys: keys, bits: 3, seed: 42, sinkTokens: sinkCount)
                let ev = TurboQuantEncoder.encodeValues(values: values, bits: 3, seed: 42, sinkTokens: sinkCount)

                guard ek.sinkCount == sinkCount else { fail("T8: sink count \(ek.sinkCount) != \(sinkCount)"); break }

                let dk = TurboQuantEncoder.decodeKeys(ek, seed: ek.seed)
                // Decoded should have full seq length (sink + compressed)
                guard dk.dim(2) == seqLen else { fail("T8: decoded seqLen \(dk.dim(2)) != \(seqLen)"); break }

                pass("T8 Sink Tokens: sinkCount=\(sinkCount), decoded_seqLen=\(dk.dim(2)) (matches original \(seqLen))")
                passedTests += 1
                break
            }
        }

        // ---- TEST 9: gen_prompt_len ----
        totalTests += 1
        do {
            let messages: [VMLXChatMessage] = [
                VMLXChatMessage(role: "user", content: "Hello")
            ]
            let genPromptLen = container.computeGenPromptLen(messages: messages)
            let tokensWithGen = try container.applyChatTemplate(messages: messages, addGenerationPrompt: true)
            let tokensNoGen = try container.applyChatTemplate(messages: messages, addGenerationPrompt: false)

            guard genPromptLen > 0 else { fail("T9: genPromptLen=0"); return }
            guard genPromptLen == tokensWithGen.count - tokensNoGen.count else {
                fail("T9: genPromptLen=\(genPromptLen) but diff=\(tokensWithGen.count - tokensNoGen.count)")
                return
            }

            pass("T9 gen_prompt_len: \(genPromptLen) tokens, withGen=\(tokensWithGen.count), noGen=\(tokensNoGen.count)")
            passedTests += 1
        } catch {
            fail("T9: \(error)")
        }

        // ---- TEST 10: KV Quantized Cache ----
        totalTests += 1
        do {
            let qCache = container.newCache(kvBits: 4, kvGroupSize: 64)
            var qkvCount = 0
            for c in qCache {
                if c is VMLXQuantizedKVCache { qkvCount += 1 }
            }
            // Should replace attention caches (8) with quantized, SSM (24) unchanged
            guard qkvCount == 8 else { fail("T10: quantized count=\(qkvCount), expected 8"); return }

            // Run a forward pass with quantized cache
            let qInput = MLXArray([Int32(1)]).reshaped(1, 1)
            let qLogits = container.forward(qInput, cache: qCache)
            MLX.eval(qLogits)
            guard qLogits.shape[2] == container.nativeModel.vocabularySize else {
                fail("T10: vocab mismatch"); return
            }

            pass("T10 KV Quantized: \(qkvCount) quantized caches, forward pass OK")
            passedTests += 1
        }

        // ---- TEST 11: Speed Benchmark (10 decode steps) ----
        totalTests += 1
        do {
            let benchCache = container.newCache()
            // Prefill with 5 tokens
            let prefillInput = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4), Int32(5)]).reshaped(1, 5)
            let prefillLogits = container.forward(prefillInput, cache: benchCache)
            var y = prefillLogits[0, -1].argMax()
            MLX.eval(y)

            // Warm up (2 tokens)
            for _ in 0..<2 {
                let d = container.forward(y.reshaped(1, 1), cache: benchCache)
                y = d[0, -1].argMax()
                MLX.eval(y)
            }

            // Benchmark 10 tokens
            let startBench = CFAbsoluteTimeGetCurrent()
            for _ in 0..<10 {
                let d = container.forward(y.reshaped(1, 1), cache: benchCache)
                y = d[0, -1].argMax()
                MLX.eval(y)
            }
            let benchElapsed = CFAbsoluteTimeGetCurrent() - startBench
            let tokPerSec = 10.0 / benchElapsed

            pass("T11 Speed: 10 tokens in \(String(format: "%.2fs", benchElapsed)) = \(String(format: "%.1f", tokPerSec)) tok/s")
            passedTests += 1
        }

        // ---- SUMMARY ----
        print("")
        print("=" * 60)
        print("SMOKE TEST RESULTS: \(passedTests)/\(totalTests) passed")
        if passedTests == totalTests {
            print("ALL TESTS PASSED")
        } else {
            print("SOME TESTS FAILED")
        }
        print("=" * 60)
    }
}

extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
