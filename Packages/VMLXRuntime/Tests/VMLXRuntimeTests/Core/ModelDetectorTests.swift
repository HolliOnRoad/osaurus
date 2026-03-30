import Testing
import Foundation
@testable import VMLXRuntime

@Suite("ModelDetector")
struct ModelDetectorTests {

    // MARK: - Helpers

    /// Create a temp model directory with optional config files.
    private func createModelDir(
        name: String = "test-model",
        jangConfig: [String: Any]? = nil,
        hfConfig: [String: Any]? = nil,
        weightIndex: [String: Any]? = nil,
        addPreprocessorConfig: Bool = false,
        safetensorsFiles: [String]? = nil
    ) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelDetectorTests-\(UUID().uuidString)")
            .appendingPathComponent(name)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        if let jang = jangConfig {
            let data = try JSONSerialization.data(withJSONObject: jang)
            try data.write(to: dir.appendingPathComponent("jang_config.json"))
        }

        if let hf = hfConfig {
            let data = try JSONSerialization.data(withJSONObject: hf)
            try data.write(to: dir.appendingPathComponent("config.json"))
        }

        if let idx = weightIndex {
            let data = try JSONSerialization.data(withJSONObject: idx)
            try data.write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        }

        if addPreprocessorConfig {
            let data = try JSONSerialization.data(withJSONObject: ["image_processor_type": "Qwen2VLImageProcessor"])
            try data.write(to: dir.appendingPathComponent("preprocessor_config.json"))
        }

        if let files = safetensorsFiles {
            for file in files {
                FileManager.default.createFile(
                    atPath: dir.appendingPathComponent(file).path,
                    contents: Data([0x00])
                )
            }
        }

        return dir
    }

    /// Standard JANG config dict for a Qwen3.5 hybrid model.
    private func qwen35JangConfig() -> [String: Any] {
        [
            "quantization": [
                "method": "jang-importance",
                "profile": "JANG_2S",
                "target_bits": 2.5,
                "actual_bits": 2.85,
                "block_size": 64,
                "calibration_method": "weights",
                "quantization_method": "mse",
                "scoring_method": "weight-magnitude",
                "bit_widths_used": [2, 4, 6],
                "quantization_scheme": "asymmetric",
                "quantization_backend": "mx.quantize"
            ] as [String: Any],
            "source_model": [
                "name": "Qwen3.5-4B",
                "dtype": "bfloat16",
                "parameters": "4025327616"
            ] as [String: Any],
            "architecture": [
                "type": "hybrid_ssm",
                "attention": "none",
                "has_vision": true,
                "has_ssm": true,
                "has_moe": false
            ] as [String: Any],
            "runtime": [
                "total_weight_bytes": 1422278016,
                "total_weight_gb": 1.32
            ] as [String: Any],
            "format": "jang",
            "format_version": "2.0"
        ]
    }

    // MARK: - Basic Detection

    @Test("Detect JANG model with full config")
    func detectJangModel() throws {
        let dir = try createModelDir(
            name: "Qwen3.5-4B-JANG_2S",
            jangConfig: qwen35JangConfig()
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)

        #expect(model.isJang)
        #expect(model.name == "Qwen3.5-4B-JANG_2S")
        #expect(model.sourceModel == "Qwen3.5-4B")
        #expect(model.jangProfile == "JANG_2S")
        #expect(model.family == "qwen3.5")

        // Architecture
        #expect(model.architectureType == "hybrid_ssm")
        #expect(model.attentionType == "none")
        #expect(model.hasVision)
        #expect(model.hasSSM)
        #expect(!model.hasMoE)
        #expect(model.isHybrid)

        // Quantization
        #expect(model.quantProfile == "JANG_2S")
        #expect(model.targetBits == 2.5)
        #expect(model.actualBits == 2.85)
        #expect(model.blockSize == 64)
        #expect(model.bitWidthsUsed == [2, 4, 6])

        // Runtime
        #expect(model.totalWeightBytes == 1422278016)
        #expect(model.totalWeightGB == 1.32)
    }

    @Test("Detect non-JANG model with only config.json")
    func detectHFModel() throws {
        let dir = try createModelDir(
            name: "Llama-3.3-70B",
            hfConfig: [
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "max_position_embeddings": 131072,
                "vocab_size": 128256,
                "num_hidden_layers": 80
            ] as [String: Any]
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)

        #expect(!model.isJang)
        #expect(model.jangProfile == nil)
        #expect(model.modelType == "llama")
        #expect(model.hfArchitectures == ["LlamaForCausalLM"])
        #expect(model.contextWindow == 131072)
        #expect(model.vocabSize == 128256)
        #expect(model.numLayers == 80)
        #expect(model.family == "llama")
    }

    @Test("Detect empty directory")
    func detectEmptyDir() throws {
        let dir = try createModelDir(name: "empty-model")
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(!model.isJang)
        #expect(model.name == "empty-model")
        #expect(model.family == "unknown")
    }

    // MARK: - Vision Detection

    @Test("Vision detected from jang_config")
    func visionFromJang() throws {
        let dir = try createModelDir(
            jangConfig: qwen35JangConfig()
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.hasVision)
    }

    @Test("Vision detected from config.json image_token_id")
    func visionFromImageToken() throws {
        let dir = try createModelDir(
            hfConfig: [
                "model_type": "qwen2_vl",
                "image_token_id": 151655,
                "video_token_id": 151656
            ] as [String: Any]
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.hasVision)
        #expect(model.imageTokenId == 151655)
        #expect(model.videoTokenId == 151656)
    }

    @Test("Vision detected from preprocessor_config.json existence")
    func visionFromPreprocessor() throws {
        let dir = try createModelDir(
            addPreprocessorConfig: true
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.hasVision)
        #expect(model.hasPreprocessorConfig)
    }

    @Test("Vision detected from vision_config in config.json")
    func visionFromVisionConfig() throws {
        let dir = try createModelDir(
            hfConfig: [
                "model_type": "qwen3_5",
                "vision_config": ["hidden_size": 1280] as [String: Any]
            ] as [String: Any]
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.hasVision)
    }

    // MARK: - MLA Detection

    @Test("MLA detected from jang_config attention type")
    func mlaFromJang() throws {
        var config = qwen35JangConfig()
        var arch = config["architecture"] as! [String: Any]
        arch["attention"] = "mla"
        arch["has_ssm"] = false
        arch["type"] = "moe"
        config["architecture"] = arch

        let dir = try createModelDir(jangConfig: config)
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.attentionType == "mla")
    }

    @Test("MLA fields parsed from config.json")
    func mlaFieldsFromHF() throws {
        let dir = try createModelDir(
            hfConfig: [
                "model_type": "deepseek_v3",
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64
            ] as [String: Any]
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.kvLoraRank == 512)
        #expect(model.qkNopeHeadDim == 128)
        #expect(model.qkRopeHeadDim == 64)
    }

    // MARK: - Hybrid Detection

    @Test("Hybrid detected from jang architecture")
    func hybridFromJang() throws {
        let dir = try createModelDir(jangConfig: qwen35JangConfig())
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.hasSSM)
        #expect(model.isHybrid)
    }

    @Test("Hybrid override pattern from config.json")
    func hybridOverridePattern() throws {
        let dir = try createModelDir(
            hfConfig: [
                "model_type": "nemotron_h",
                "hybrid_override_pattern": "MEMEM*MEMEM*MEMEM*"
            ] as [String: Any]
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.hybridOverridePattern == "MEMEM*MEMEM*MEMEM*")
    }

    @Test("Layer types from nested text_config")
    func layerTypesFromTextConfig() throws {
        let dir = try createModelDir(
            hfConfig: [
                "model_type": "qwen3_5",
                "text_config": [
                    "layer_types": ["linear_attention", "full_attention", "linear_attention"]
                ] as [String: Any]
            ] as [String: Any]
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.layerTypes == ["linear_attention", "full_attention", "linear_attention"])
    }

    // MARK: - MoE Detection

    @Test("MoE fields from config.json")
    func moeFromHF() throws {
        let dir = try createModelDir(
            hfConfig: [
                "model_type": "mixtral",
                "num_local_experts": 8,
                "num_experts_per_tok": 2
            ] as [String: Any]
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.numExperts == 8)
        #expect(model.numExpertsPerTok == 2)
    }

    @Test("MoE from jang architecture")
    func moeFromJang() throws {
        var config = qwen35JangConfig()
        var arch = config["architecture"] as! [String: Any]
        arch["has_moe"] = true
        arch["type"] = "hybrid_moe_ssm"
        config["architecture"] = arch

        let dir = try createModelDir(jangConfig: config)
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.hasMoE)
        #expect(model.architectureType == "hybrid_moe_ssm")
    }

    // MARK: - Weight Files

    @Test("Weight files from safetensors index")
    func weightFilesFromIndex() throws {
        let dir = try createModelDir(
            jangConfig: qwen35JangConfig(),
            weightIndex: [
                "metadata": [
                    "format": "jang",
                    "format_version": "2.0",
                    "total_size": 1422278016
                ] as [String: Any],
                "weight_map": [
                    "model.layers.0.weight": "model-00001-of-00003.safetensors",
                    "model.layers.1.weight": "model-00001-of-00003.safetensors",
                    "model.layers.2.weight": "model-00002-of-00003.safetensors",
                    "model.layers.3.weight": "model-00003-of-00003.safetensors"
                ] as [String: String]
            ]
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.numShards == 3)
        #expect(model.weightFiles.count == 3)
        #expect(model.weightFiles.contains("model-00001-of-00003.safetensors"))
    }

    @Test("Weight files discovered from filesystem")
    func weightFilesFromFilesystem() throws {
        let dir = try createModelDir(
            safetensorsFiles: [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors"
            ]
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.numShards == 2)
        #expect(model.weightFiles.count == 2)
    }

    @Test("Runtime overrides weight index total_size")
    func runtimeOverridesIndex() throws {
        let dir = try createModelDir(
            jangConfig: qwen35JangConfig(),
            weightIndex: [
                "metadata": [
                    "total_size": 999999
                ] as [String: Any],
                "weight_map": [:] as [String: String]
            ]
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        // jang_config runtime should take precedence
        #expect(model.totalWeightBytes == 1422278016)
    }

    // MARK: - Family Detection

    @Test("Family detected from source model name")
    func familyFromSourceModel() throws {
        let cases: [(String, String)] = [
            ("Qwen3.5-4B", "qwen3.5"),
            ("Llama-4-Maverick", "llama4"),
            ("Mistral-Large-2", "mistral"),
            ("DeepSeek-V3", "deepseek"),
            ("Nemotron-H-47B", "nemotron"),
            ("Gemma-3n-E4B", "gemma"),
            ("Phi-4-mini", "phi"),
            ("MiniMax-M2-40B", "minimax"),
        ]

        for (sourceName, expectedFamily) in cases {
            var config = qwen35JangConfig()
            var sm = config["source_model"] as! [String: Any]
            sm["name"] = sourceName
            config["source_model"] = sm

            let dir = try createModelDir(jangConfig: config)
            defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

            let model = try ModelDetector.detect(at: dir)
            #expect(model.family == expectedFamily,
                    "Expected family '\(expectedFamily)' for source '\(sourceName)', got '\(model.family)'")
        }
    }

    @Test("Family detected from HF model_type")
    func familyFromModelType() throws {
        let dir = try createModelDir(
            hfConfig: [
                "model_type": "nemotron_h"
            ] as [String: Any]
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.family == "nemotron")
    }

    // MARK: - Merged Detection

    @Test("Merges jang_config and config.json")
    func mergedDetection() throws {
        let dir = try createModelDir(
            jangConfig: qwen35JangConfig(),
            hfConfig: [
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "max_position_embeddings": 32768,
                "vocab_size": 151936,
                "num_hidden_layers": 36,
                "text_config": [
                    "layer_types": ["linear_attention", "full_attention"]
                ] as [String: Any]
            ] as [String: Any],
            addPreprocessorConfig: true
        )
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)

        // From jang_config
        #expect(model.isJang)
        #expect(model.hasSSM)
        #expect(model.hasVision)
        #expect(model.quantProfile == "JANG_2S")

        // From config.json
        #expect(model.modelType == "qwen3_5")
        #expect(model.hfArchitectures == ["Qwen3_5ForConditionalGeneration"])
        #expect(model.contextWindow == 32768)
        #expect(model.vocabSize == 151936)
        #expect(model.numLayers == 36)
        #expect(model.layerTypes == ["linear_attention", "full_attention"])

        // From filesystem
        #expect(model.hasPreprocessorConfig)
    }

    // MARK: - Display Name

    @Test("Display name includes profile for JANG models")
    func displayNameJang() throws {
        let dir = try createModelDir(jangConfig: qwen35JangConfig())
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.name == "Qwen3.5-4B-JANG_2S")
    }

    @Test("Display name falls back to directory name")
    func displayNameFallback() throws {
        let dir = try createModelDir(name: "my-custom-model")
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }

        let model = try ModelDetector.detect(at: dir)
        #expect(model.name == "my-custom-model")
    }

    // MARK: - Scan (smoke test)

    @Test("scanAvailableModels does not crash")
    func scanSmoke() {
        // Just verify it runs without crashing.
        // On CI / clean machines this will return empty or a few models.
        let models = ModelDetector.scanAvailableModels()
        // No assertion on count -- just verify it completes
        _ = models
    }
}
