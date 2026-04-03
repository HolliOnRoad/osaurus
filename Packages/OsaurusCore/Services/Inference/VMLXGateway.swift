//
//  VMLXGateway.swift
//  osaurus
//
//  Gateway/router that tracks running vmlx engine instances.
//  Each model gets its own Python subprocess on a unique port.
//  The gateway maps model names to their port for request routing.
//

import Foundation

/// Tracks a running vmlx engine instance.
struct VMLXInstance: Sendable {
    let modelName: String
    let modelPath: String
    let port: Int
    let processIdentifier: Int32
    let startedAt: Date
}

/// Actor that manages the mapping of model names to running engine instances.
/// Thread-safe via Swift actor isolation.
actor VMLXGateway {
    static let shared = VMLXGateway()

    /// Active instances keyed by model name
    private var instances: [String: VMLXInstance] = [:]

    // MARK: - Registration

    /// Register a newly launched engine instance.
    func register(_ instance: VMLXInstance) {
        instances[instance.modelName] = instance
    }

    /// Unregister an instance by model name.
    func unregister(model: String) {
        instances.removeValue(forKey: model)
    }

    /// Unregister all instances.
    func unregisterAll() {
        instances.removeAll()
    }

    // MARK: - Routing

    /// Get the port for a running model, or nil if not loaded.
    func port(for model: String) -> Int? {
        // Try exact match first
        if let instance = instances[model] {
            return instance.port
        }
        // Try case-insensitive match
        for (key, instance) in instances {
            if key.caseInsensitiveCompare(model) == .orderedSame {
                return instance.port
            }
        }
        // Try matching the last path component (e.g. "Llama-3.2-3B-Instruct-4bit"
        // matches "mlx-community/Llama-3.2-3B-Instruct-4bit")
        let modelSuffix = model.split(separator: "/").last.map(String.init) ?? model
        for (key, instance) in instances {
            let keySuffix = key.split(separator: "/").last.map(String.init) ?? key
            if keySuffix.caseInsensitiveCompare(modelSuffix) == .orderedSame {
                return instance.port
            }
        }
        return nil
    }

    /// Get the instance for a model.
    func instance(for model: String) -> VMLXInstance? {
        if let instance = instances[model] {
            return instance
        }
        // Fallback: case-insensitive
        for (key, instance) in instances {
            if key.caseInsensitiveCompare(model) == .orderedSame {
                return instance
            }
        }
        return nil
    }

    /// List all available (running) model names.
    func availableModels() -> [String] {
        Array(instances.keys)
    }

    /// List all running instances.
    func allInstances() -> [VMLXInstance] {
        Array(instances.values)
    }

    /// Check if any instance is running.
    func hasRunningInstances() -> Bool {
        !instances.isEmpty
    }

    /// Get the first running instance (for single-model mode).
    func firstInstance() -> VMLXInstance? {
        instances.values.first
    }

    /// Number of running instances.
    var count: Int {
        instances.count
    }
}
