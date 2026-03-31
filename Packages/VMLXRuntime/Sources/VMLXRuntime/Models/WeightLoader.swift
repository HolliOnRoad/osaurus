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

        // Quantize all layers with config-level bits/group_size.
        // The actual per-layer bits will be fixed by vmlxFixQuantizedBits() after
        // model.update() loads the packed weights.
        var quantizedCount = 0
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                quantizedCount += 1
                return (defaultGroupSize, defaultBits, mode)
            }
            return nil
        }
        NSLog("[WeightLoader] Quantized \(quantizedCount) modules, bits=\(defaultBits), gs=\(defaultGroupSize)")
    }

    // Load weights (no strict verification for JANG mixed-precision compatibility)
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [])

    // Fix per-layer bits/group_size for JANG mixed-precision models.
    // After model.update() loads packed weights, each QuantizedLinear/QuantizedEmbedding
    // may have wrong bits (e.g. bits=2 from config but weight is actually 8-bit).
    // We compute the correct bits from weight/scales shapes and mutate the module.
    // This requires our mlx-swift fork (jjang-ai/mlx-swift) which makes bits/groupSize
    // mutable (`var` instead of `let`).
    // Ported from VMLX Python engine's _fix_quantized_bits().
    let fixGroupSize = quantization?.groupSize ?? 64
    vmlxFixQuantizedBits(model: model, defaultGroupSize: fixGroupSize)
}

/// Fix per-layer bits and groupSize on all QuantizedLinear/QuantizedEmbedding modules.
/// Computes correct values from actual weight/scales shapes and mutates the module.
/// Requires jjang-ai/mlx-swift fork (mutable bits/groupSize).
/// Ported from VMLX Python engine's _fix_quantized_bits().
public func vmlxFixQuantizedBits(model: Module, defaultGroupSize: Int) {
    for (name, module) in model.namedModules() {
        // Handle QuantizedLinear (including QuantizedSwitchLinear via inheritance)
        if let ql = module as? QuantizedLinear {
            guard ql.weight.ndim >= 2 else { return }
            let wCols = ql.weight.dim(ql.weight.ndim - 1)
            let sCols = ql.scales.dim(ql.scales.ndim - 1)

            let nameLower = name.lowercased()
            let isRouter = (nameLower.contains(".gate") && !nameLower.contains("_proj"))
                || nameLower.contains("shared_expert_gate")
            let gsCandidates: [Int] = isRouter
                ? [64, defaultGroupSize, 128, 32, 256]
                : [defaultGroupSize, 64, 128, 32, 256]

            for tryGS in gsCandidates {
                let inDim = sCols * tryGS
                guard inDim > 0, (wCols * 32) % inDim == 0 else { continue }
                let tryBits = (wCols * 32) / inDim
                guard [2, 3, 4, 5, 6, 8].contains(tryBits) else { continue }
                if tryBits != ql.bits || tryGS != ql.groupSize {
                    print("[FixBits] \(name): bits \(ql.bits)→\(tryBits), gs \(ql.groupSize)→\(tryGS)")
                    ql.bits = tryBits
                    ql.groupSize = tryGS
                }
                break
            }
        }

        // Handle QuantizedEmbedding
        if let qe = module as? QuantizedEmbedding {
            guard qe.weight.ndim >= 2 else { return }
            let wCols = qe.weight.dim(qe.weight.ndim - 1)
            let sCols = qe.scales.dim(qe.scales.ndim - 1)

            for tryGS in [defaultGroupSize, 64, 128, 32, 256] {
                let inDim = sCols * tryGS
                guard inDim > 0, (wCols * 32) % inDim == 0 else { continue }
                let tryBits = (wCols * 32) / inDim
                guard [2, 3, 4, 5, 6, 8].contains(tryBits) else { continue }
                if tryBits != qe.bits { qe.bits = tryBits }
                if tryGS != qe.groupSize { qe.groupSize = tryGS }
                break
            }
        }
    }
}

/// Walk all modules in a model tree, calling visitor with (path, module).
private func _walkModules(_ module: Module, path: String = "", visitor: (String, Module) -> Void) {
    visitor(path, module)
    let mirror = Mirror(reflecting: module)
    for child in mirror.children {
        if let childModule = child.value as? Module {
            let childPath = path.isEmpty ? (child.label ?? "") : "\(path).\(child.label ?? "")"
            _walkModules(childModule, path: childPath, visitor: visitor)
        } else if let children = child.value as? [Module] {
            for (i, childModule) in children.enumerated() {
                let childPath = path.isEmpty ? "\(i)" : "\(path).\(i)"
                _walkModules(childModule, path: childPath, visitor: visitor)
            }
        }
    }
}
