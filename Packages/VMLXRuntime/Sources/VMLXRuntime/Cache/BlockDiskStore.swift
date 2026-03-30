import Foundation
@preconcurrency import MLX
import os

/// Block-level disk persistence for paged cache.
/// Stores individual cache blocks as safetensors files indexed by block hash.
public final class BlockDiskStore: @unchecked Sendable {

    public let cacheDir: URL
    public let maxSizeBytes: Int
    private let lock = OSAllocatedUnfairLock()

    // Stats (access under lock)
    private var _hits: Int = 0
    private var _misses: Int = 0
    private var _stores: Int = 0

    public var hits: Int { lock.withLock { _hits } }
    public var misses: Int { lock.withLock { _misses } }
    public var stores: Int { lock.withLock { _stores } }

    public init(cacheDir: URL, maxSizeGB: Float = 10.0) {
        self.cacheDir = cacheDir
        self.maxSizeBytes = Int(maxSizeGB * 1024 * 1024 * 1024)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
    }

    /// Store a cache block by its hash.
    public func store(blockHash: BlockHash, layers: [(keys: MLXArray, values: MLXArray)]) {
        let fileName = blockHash.hexString + ".safetensors"
        let fileURL = cacheDir.appendingPathComponent(fileName)

        // Build tensor dict
        var mutableArrays: [String: MLXArray] = [:]
        for (i, layer) in layers.enumerated() {
            mutableArrays["layer_\(i)_keys"] = layer.keys
            mutableArrays["layer_\(i)_values"] = layer.values
        }

        // Pre-materialize on calling thread (Metal safety)
        // MLXArray.eval() forces GPU computation, NOT code evaluation
        for (_, array) in mutableArrays { array.eval() }

        // Freeze as immutable for background write
        let arrays = mutableArrays
        let metadata = ["__num_layers__": "\(layers.count)"]

        lock.withLock { _stores += 1 }

        // Write safetensors file in background via GCD (avoids Swift concurrency
        // sendability checks since MLXArray is not Sendable)
        DispatchQueue.global(qos: .utility).async {
            do {
                try MLX.save(arrays: arrays, metadata: metadata, url: fileURL)
            } catch {
                // Write failed; fetch will treat as miss
            }
        }
    }

    /// Fetch a cache block by its hash.
    public func fetch(blockHash: BlockHash) -> [(keys: MLXArray, values: MLXArray)]? {
        let fileName = blockHash.hexString + ".safetensors"
        let fileURL = cacheDir.appendingPathComponent(fileName)

        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            lock.withLock { _misses += 1 }
            return nil
        }

        let tensors: [String: MLXArray]
        do {
            tensors = try loadArrays(url: fileURL)
        } catch {
            lock.withLock { _misses += 1 }
            return nil
        }

        // Reconstruct layers
        var layers: [(keys: MLXArray, values: MLXArray)] = []
        var i = 0
        while let keys = tensors["layer_\(i)_keys"],
              let values = tensors["layer_\(i)_values"] {
            layers.append((keys, values))
            i += 1
        }

        guard !layers.isEmpty else {
            lock.withLock { _misses += 1 }
            return nil
        }

        lock.withLock { _hits += 1 }
        return layers
    }

    /// Check if a block exists on disk.
    public func contains(blockHash: BlockHash) -> Bool {
        let fileName = blockHash.hexString + ".safetensors"
        return FileManager.default.fileExists(atPath: cacheDir.appendingPathComponent(fileName).path)
    }
}
