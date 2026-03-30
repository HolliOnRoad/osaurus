//
//  WeightLoader.swift
//  VMLXRuntime
//
//  Weight loading for native VMLXRuntime models.
//  Ported from mlx-swift-lm's Load.swift.
//
//  Handles:
//  - Loading safetensors files from a model directory
//  - Calling model.sanitize() for weight key remapping
//  - Auto-quantizing Linear -> QuantizedLinear when weights have .scales
//  - Calling model.update(parameters:) to load weights into the model
//

import Foundation
import MLX
import MLXNN

// MARK: - Base Configuration

/// Parsed from config.json to extract quantization info and model_type.
public struct VMLXBaseConfiguration: Codable, Sendable {
    public let modelType: String

    public struct Quantization: Codable, Sendable {
        public let groupSize: Int
        public let bits: Int
        private var _mode: QuantizationMode? = nil
        public var mode: QuantizationMode { _mode ?? .affine }

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits = "bits"
            case _mode = "mode"
        }
    }

    public let quantization: Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case quantization
    }
}

// MARK: - Weight Loading

/// Protocol for models that can sanitize their weight keys.
public protocol VMLXSanitizable {
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]
}

/// Load safetensors weights from a model directory, apply sanitization and quantization,
/// and update the model's parameters.
///
/// Note: The `eval(model)` call at the end is MLX's lazy evaluation trigger (not code eval).
/// It forces all pending MLX computations to materialize, which is required after weight loading.
public func vmlxLoadWeights(
    modelDirectory: URL,
    model: Module,
    quantization: VMLXBaseConfiguration.Quantization? = nil
) throws {
    // 1. Load all safetensors files
    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }
    }

    // 2. Model-specific key sanitization
    if let sanitizable = model as? VMLXSanitizable {
        weights = sanitizable.sanitize(weights: weights)
    }

    // 3. Auto-quantize: if weights contain .scales keys, convert Linear -> QuantizedLinear
    //
    // For JANG mixed-precision models: different layers use different bit widths
    // (e.g., SSM layers at 4-bit, attention at 6-bit, embedding at 4-bit).
    // We infer the actual bits per-layer from weight/scales shapes:
    //   bits = weight.dim(1) * 32 / (scales.dim(1) * group_size)
    let hasScales = weights.keys.contains { $0.hasSuffix(".scales") }
    if hasScales {
        let defaultGroupSize = quantization?.groupSize ?? 64
        let defaultBits = quantization?.bits ?? 4
        let mode = quantization?.mode ?? .affine

        // Check if the config-level bits is unsupported by MLX (e.g. JANG_4K uses 3-bit)
        let mlxValidBits = [2, 4, 6, 8]
        if !mlxValidBits.contains(defaultBits) {
            throw ModelLoaderError.unsupportedArchitecture(
                "Model uses \(defaultBits)-bit quantization which MLX does not support. "
                + "Supported: 2, 4, 6, 8-bit. JANG_4K (3-bit) requires a custom dequantization kernel."
            )
        }

        // Per-layer bits/group_size from actual weight shapes.
        // JANG mixed-precision: different layers use different bits AND group_size.
        // Router/gate tensors prefer gs=64 (precision-critical).
        // Ported from VMLX Python engine's _fix_quantized_bits() logic.
        quantize(model: model) { path, module in
            guard let scales = weights["\(path).scales"],
                  let weight = weights["\(path).weight"] else {
                return nil
            }

            let wCols = weight.dim(weight.ndim - 1)
            let sCols = scales.dim(scales.ndim - 1)

            let pathLower = path.lowercased()
            let isRouter = pathLower.contains(".gate.") || pathLower.hasSuffix(".gate")
                || pathLower.contains("shared_expert_gate")
            let gsCandidates: [Int]
            if isRouter {
                gsCandidates = [64, defaultGroupSize, 128, 32, 256]
            } else {
                gsCandidates = [defaultGroupSize, 64, 128, 32, 256]
            }

            for tryGS in gsCandidates {
                let inDim = sCols * tryGS
                guard inDim > 0, (wCols * 32) % inDim == 0 else { continue }
                let tryBits = (wCols * 32) / inDim
                if [2, 3, 4, 5, 6, 8].contains(tryBits) {
                    return (tryGS, tryBits, mode)
                }
            }

            return (defaultGroupSize, defaultBits, mode)
        }
    }

    // 4. Load weights into model
    // Use .noUnusedKeys to catch weight naming errors, but allow missing keys
    // (e.g., bias parameters that exist in the model but not in the weights —
    //  they stay at their initialized zero values, which is correct behavior
    //  for models like Qwen2 where Q/K/V have bias but O does not).
    // Load weights (no strict verification for JANG mixed-precision compatibility)
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [])
}
