import Testing
import Foundation
@testable import VMLXRuntime

@Suite("JangLoader")
struct JangLoaderTests {

    // MARK: - Helpers

    /// Create a temp model directory with a jang_config.json containing the given JSON dict.
    private func createTempModelDir(
        with configName: String = "jang_config.json",
        content: [String: Any]
    ) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("JangLoaderTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let configURL = dir.appendingPathComponent(configName)
        let data = try JSONSerialization.data(withJSONObject: content)
        try data.write(to: configURL)

        return dir
    }

    /// Build a real-format jang_config.json dictionary.
    private func realJangConfig(
        profile: String = "JANG_2S",
        targetBits: Double = 2.5,
        actualBits: Double = 2.85,
        blockSize: Int = 64,
        bitWidths: [Int] = [2, 4, 6],
        sourceName: String = "Qwen3.5-4B",
        archType: String = "hybrid_ssm",
        attention: String = "none",
        hasVision: Bool = true,
        hasSSM: Bool = true,
        hasMoE: Bool = false,
        totalBytes: Int = 1422278016,
        totalGB: Double = 1.32
    ) -> [String: Any] {
        [
            "quantization": [
                "method": "jang-importance",
                "profile": profile,
                "target_bits": targetBits,
                "actual_bits": actualBits,
                "block_size": blockSize,
                "calibration_method": "weights",
                "quantization_method": "mse",
                "scoring_method": "weight-magnitude",
                "bit_widths_used": bitWidths,
                "quantization_scheme": "asymmetric",
                "quantization_backend": "mx.quantize"
            ] as [String: Any],
            "source_model": [
                "name": sourceName,
                "dtype": "bfloat16",
                "parameters": "4025327616"
            ] as [String: Any],
            "architecture": [
                "type": archType,
                "attention": attention,
                "has_vision": hasVision,
                "has_ssm": hasSSM,
                "has_moe": hasMoE
            ] as [String: Any],
            "runtime": [
                "total_weight_bytes": totalBytes,
                "total_weight_gb": totalGB
            ] as [String: Any],
            "format": "jang",
            "format_version": "2.0"
        ]
    }

    // MARK: - Detection

    @Test("Detect JANG model")
    func detectJangModel() throws {
        let dir = try createTempModelDir(content: realJangConfig())
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(JangLoader.isJangModel(at: dir))
    }

    @Test("Non-JANG directory returns false")
    func notJangModel() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("JangLoaderTests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(!JangLoader.isJangModel(at: dir))
    }

    @Test("Finds alternative config names")
    func alternativeConfigNames() throws {
        for name in jangConfigFileNames {
            let dir = try createTempModelDir(with: name, content: realJangConfig())
            defer { try? FileManager.default.removeItem(at: dir) }

            #expect(JangLoader.isJangModel(at: dir))
            let configPath = JangLoader.findConfigPath(at: dir)
            #expect(configPath?.lastPathComponent == name)
        }
    }

    // MARK: - Config Parsing

    @Test("Parse real-format jang_config.json")
    func parseRealConfig() throws {
        let dir = try createTempModelDir(content: realJangConfig())
        defer { try? FileManager.default.removeItem(at: dir) }

        let config = try JangLoader.loadConfig(at: dir)

        // Format
        #expect(config.format == "jang")
        #expect(config.formatVersion == "2.0")
        #expect(config.isV2)

        // Quantization
        #expect(config.quantization.method == "jang-importance")
        #expect(config.quantization.profile == "JANG_2S")
        #expect(config.quantization.targetBits == 2.5)
        #expect(config.quantization.actualBits == 2.85)
        #expect(config.quantization.blockSize == 64)
        #expect(config.quantization.bitWidthsUsed == [2, 4, 6])
        #expect(config.quantization.quantizationScheme == "asymmetric")
        #expect(config.quantization.quantizationBackend == "mx.quantize")

        // Source model
        #expect(config.sourceModel.name == "Qwen3.5-4B")
        #expect(config.sourceModel.dtype == "bfloat16")
        #expect(config.sourceModel.parameters == "4025327616")
        #expect(config.sourceModel.parameterCount == 4025327616)

        // Architecture
        #expect(config.architecture.type == "hybrid_ssm")
        #expect(config.architecture.attention == "none")
        #expect(config.architecture.hasVision == true)
        #expect(config.architecture.hasSSM == true)
        #expect(config.architecture.hasMoE == false)

        // Runtime
        #expect(config.runtime.totalWeightBytes == 1422278016)
        #expect(config.runtime.totalWeightGB == 1.32)
    }

    @Test("Parse MoE model config")
    func parseMoEConfig() throws {
        let dir = try createTempModelDir(content: realJangConfig(
            profile: "JANG_4K",
            targetBits: 2.5,
            bitWidths: [3, 4, 5, 6, 8],
            sourceName: "Qwen3.5-122B-A10B",
            archType: "hybrid_moe_ssm",
            hasVision: true,
            hasSSM: true,
            hasMoE: true
        ))
        defer { try? FileManager.default.removeItem(at: dir) }

        let config = try JangLoader.loadConfig(at: dir)
        #expect(config.architecture.type == "hybrid_moe_ssm")
        #expect(config.architecture.hasMoE == true)
        #expect(config.architecture.hasSSM == true)
        #expect(config.quantization.profile == "JANG_4K")
        #expect(config.quantization.bitWidthsUsed == [3, 4, 5, 6, 8])
    }

    @Test("Parse MLA model config (Mistral)")
    func parseMLA() throws {
        let dir = try createTempModelDir(content: realJangConfig(
            archType: "moe",
            attention: "mla",
            hasVision: true,
            hasSSM: false,
            hasMoE: true
        ))
        defer { try? FileManager.default.removeItem(at: dir) }

        let config = try JangLoader.loadConfig(at: dir)
        #expect(config.architecture.attention == "mla")
        #expect(JangLoader.isMLA(config: config))
        #expect(!JangLoader.isHybridModel(config: config))
        #expect(JangLoader.isMoE(config: config))
    }

    @Test("Parse Nemotron config (pure MoE, GQA)")
    func parseNemotron() throws {
        let dir = try createTempModelDir(content: realJangConfig(
            archType: "moe",
            attention: "gqa",
            hasVision: false,
            hasSSM: false,
            hasMoE: true
        ))
        defer { try? FileManager.default.removeItem(at: dir) }

        let config = try JangLoader.loadConfig(at: dir)
        #expect(config.architecture.type == "moe")
        #expect(config.architecture.attention == "gqa")
        #expect(!JangLoader.isHybridModel(config: config))
        #expect(JangLoader.isMoE(config: config))
        #expect(!JangLoader.isVisionModel(config: config))
    }

    // MARK: - Architecture Introspection

    @Test("isHybridModel checks architecture.has_ssm")
    func hybridDetection() {
        let hybrid = JangConfig(architecture: JangArchitecture(hasSSM: true))
        #expect(JangLoader.isHybridModel(config: hybrid))

        let nonHybrid = JangConfig(architecture: JangArchitecture(hasSSM: false))
        #expect(!JangLoader.isHybridModel(config: nonHybrid))
    }

    @Test("isVisionModel checks architecture.has_vision")
    func visionDetection() {
        let vision = JangConfig(architecture: JangArchitecture(hasVision: true))
        #expect(JangLoader.isVisionModel(config: vision))

        let noVision = JangConfig(architecture: JangArchitecture(hasVision: false))
        #expect(!JangLoader.isVisionModel(config: noVision))
    }

    @Test("isMLA checks architecture.attention")
    func mlaDetection() {
        let mla = JangConfig(architecture: JangArchitecture(attention: "mla"))
        #expect(JangLoader.isMLA(config: mla))

        let gqa = JangConfig(architecture: JangArchitecture(attention: "gqa"))
        #expect(!JangLoader.isMLA(config: gqa))

        let none = JangConfig(architecture: JangArchitecture(attention: "none"))
        #expect(!JangLoader.isMLA(config: none))
    }

    @Test("isMoE checks architecture.has_moe")
    func moeDetection() {
        let moe = JangConfig(architecture: JangArchitecture(hasMoE: true))
        #expect(JangLoader.isMoE(config: moe))

        let noMoe = JangConfig(architecture: JangArchitecture(hasMoE: false))
        #expect(!JangLoader.isMoE(config: noMoe))
    }

    // MARK: - TurboQuant Integration

    @Test("buildTQConfig from JANG quantization profile")
    func buildTQConfig() {
        let config = JangConfig(
            quantization: JangQuantization(
                profile: "JANG_2S",
                bitWidthsUsed: [2, 4, 6]
            )
        )

        let tq = JangLoader.buildTQConfig(from: config)
        #expect(tq != nil)
        // min bits = 2 -> default
        #expect(tq?.defaultKeyBits == 2)
        #expect(tq?.defaultValueBits == 2)
        // max bits = 6 -> critical
        #expect(tq?.criticalKeyBits == 6)
        #expect(tq?.criticalValueBits == 6)
    }

    @Test("buildTQConfig with JANG_4M profile (4-bit)")
    func buildTQConfig4M() {
        let config = JangConfig(
            quantization: JangQuantization(
                profile: "JANG_4M",
                targetBits: 4.0,
                bitWidthsUsed: [4, 8]
            )
        )

        let tq = JangLoader.buildTQConfig(from: config)
        #expect(tq != nil)
        #expect(tq?.defaultKeyBits == 4)
        #expect(tq?.criticalKeyBits == 8)
    }

    @Test("buildTQConfig with empty bit widths returns nil")
    func buildTQConfigEmpty() {
        let config = JangConfig(
            quantization: JangQuantization(bitWidthsUsed: [])
        )
        #expect(JangLoader.buildTQConfig(from: config) == nil)
    }

    @Test("buildTQConfig with layer pattern and MLA")
    func buildTQConfigWithPatternAndMLA() {
        let config = JangConfig(
            quantization: JangQuantization(bitWidthsUsed: [2, 4, 6])
        )

        let pattern: [LayerType] = [.ssm, .ssm, .ssm, .attention]
        let tq = JangLoader.buildTQConfig(
            from: config,
            layerPattern: pattern,
            kvLoraRank: 512,
            qkNopeHeadDim: 128,
            qkRopeHeadDim: 64,
            vHeadDim: 128
        )

        #expect(tq != nil)
        #expect(tq?.layerPattern == [.ssm, .ssm, .ssm, .attention])
        #expect(tq?.mlaKeyDim == 192)  // 128 + 64
        #expect(tq?.mlaValueDim == 128)
        #expect(tq?.isMLA == true)
    }

    // MARK: - Errors

    @Test("Config not found error")
    func configNotFound() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("JangLoaderTests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(throws: JangLoaderError.self) {
            _ = try JangLoader.loadConfig(at: dir)
        }
    }

    @Test("Error descriptions are non-nil")
    func errorDescriptions() {
        let errors: [JangLoaderError] = [
            .configNotFound("/path"),
            .invalidConfig("bad"),
            .unsupportedVersion("0.5"),
            .loadFailed("error")
        ]
        for error in errors {
            #expect(error.errorDescription != nil)
        }
    }

    // MARK: - Quantization Profiles

    @Test("All JANG profiles parse correctly")
    func allProfiles() throws {
        let profiles: [(String, Float, [Int])] = [
            ("JANG_1L", 2.5, [2, 8]),
            ("JANG_2L", 2.0, [2, 6, 8]),
            ("JANG_2S", 2.5, [2, 4, 6]),
            ("JANG_4K", 2.5, [3, 4, 5, 6, 8]),
            ("JANG_4M", 4.0, [4, 8]),
            ("JANG_4S", 2.5, [4, 6]),
        ]

        for (profile, targetBits, bitWidths) in profiles {
            let dir = try createTempModelDir(content: realJangConfig(
                profile: profile,
                targetBits: Double(targetBits),
                bitWidths: bitWidths
            ))
            defer { try? FileManager.default.removeItem(at: dir) }

            let config = try JangLoader.loadConfig(at: dir)
            #expect(config.quantization.profile == profile)
            #expect(config.quantization.targetBits == targetBits)
            #expect(config.quantization.bitWidthsUsed == bitWidths)
        }
    }
}
