import Foundation
import os
import CryptoKit
import SQLite3
import MLX

/// L2 SSD cache with SQLite index and file-based storage.
/// Pre-serializes on calling thread (Metal safety), writes in background.
///
/// Tensor I/O uses safetensors via mlx-swift's `save(arrays:metadata:url:)` and
/// `loadArraysAndMetadata(url:)`. Pre-serialization happens on the calling thread
/// to avoid Metal cross-thread issues; the file write is dispatched to a background Task.
public final class DiskCache: @unchecked Sendable {

    public let cacheDir: URL
    public let maxSizeBytes: Int

    private let lock = OSAllocatedUnfairLock()
    private var db: OpaquePointer?  // SQLite handle
    private let dbPath: String

    /// SQLITE_TRANSIENT tells SQLite to copy the string immediately.
    private static let sqliteTransient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    // Stats (access only under lock)
    private var _hits: Int = 0
    private var _misses: Int = 0
    private var _stores: Int = 0

    public var hits: Int { lock.withLock { _hits } }
    public var misses: Int { lock.withLock { _misses } }
    public var stores: Int { lock.withLock { _stores } }

    public init(cacheDir: URL, maxSizeGB: Float = 10.0) {
        self.cacheDir = cacheDir
        self.maxSizeBytes = Int(maxSizeGB * 1024 * 1024 * 1024)
        self.dbPath = cacheDir.appendingPathComponent("cache_index.db").path

        // Create directory
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        // Open SQLite
        _openDatabase()
        _createTable()
    }

    deinit {
        if let db = db {
            sqlite3_close(db)
        }
    }

    // MARK: - Public API

    /// Check if tokens are cached on disk.
    public func contains(tokens: [Int]) -> Bool {
        let hash = Self.hashTokens(tokens)
        return lock.withLock {
            _lookupEntry(hash: hash) != nil
        }
    }

    /// Record a cache store (metadata only for now).
    /// Actual tensor I/O will be added with safetensors support.
    public func store(tokens: [Int], numTokens: Int, fileSize: Int = 0, metadata: String? = nil) -> Bool {
        let hash = Self.hashTokens(tokens)
        let fileName = "\(hash).safetensors"

        return lock.withLock {
            // Check if already cached
            if _lookupEntry(hash: hash) != nil {
                _updateAccess(hash: hash)
                return true
            }

            // Insert new entry
            let now = CFAbsoluteTimeGetCurrent()
            let sql = """
                INSERT INTO cache_entries (token_hash, file_name, num_tokens, file_size, \
                created_at, last_accessed, access_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """

            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return false }
            defer { sqlite3_finalize(stmt) }

            sqlite3_bind_text(stmt, 1, hash, -1, Self.sqliteTransient)
            sqlite3_bind_text(stmt, 2, fileName, -1, Self.sqliteTransient)
            sqlite3_bind_int(stmt, 3, Int32(numTokens))
            sqlite3_bind_int(stmt, 4, Int32(fileSize))
            sqlite3_bind_double(stmt, 5, now)
            sqlite3_bind_double(stmt, 6, now)
            if let meta = metadata {
                sqlite3_bind_text(stmt, 7, meta, -1, Self.sqliteTransient)
            } else {
                sqlite3_bind_null(stmt, 7)
            }

            guard sqlite3_step(stmt) == SQLITE_DONE else { return false }
            _stores += 1

            _evictIfNeeded()
            return true
        }
    }

    /// Fetch entry metadata. Returns (fileName, numTokens, metadata) or nil.
    public func fetch(tokens: [Int]) -> (fileName: String, numTokens: Int, metadata: String?)? {
        let hash = Self.hashTokens(tokens)
        return lock.withLock {
            guard let entry = _lookupEntry(hash: hash) else {
                _misses += 1
                return nil
            }
            _updateAccess(hash: hash)
            _hits += 1
            return entry
        }
    }

    /// Remove entry for tokens.
    public func remove(tokens: [Int]) {
        let hash = Self.hashTokens(tokens)
        lock.withLock {
            _deleteEntry(hash: hash)
        }
    }

    /// Total entries in index.
    public var entryCount: Int {
        lock.withLock { _entryCount() }
    }

    /// Total size of all cached files (from index metadata).
    public var totalSizeBytes: Int {
        lock.withLock { _totalSizeBytes() }
    }

    // MARK: - Tensor I/O (Safetensors)

    /// Serialize a HybridCache to a safetensors file on disk.
    ///
    /// **Metal safety**: The caller MUST invoke this on the thread that owns the
    /// MLXArrays (typically the inference thread). We pre-evaluate all tensors here
    /// so the background file write only touches CPU data.
    ///
    /// Key convention:
    /// - `layer_{i}_keys`, `layer_{i}_values` for attention layers
    /// - `layer_{i}_state_{j}` for SSM layers
    ///
    /// Metadata:
    /// - `__num_layers__` — total layer count
    /// - `__layer_{i}_type__` — "attention" or "ssm"
    /// - `__layer_{i}_offset__` — token offset (attention) or "0" (ssm)
    /// - `__layer_{i}_state_count__` — number of state tensors (ssm only)
    public func storeCache(tokens: [Int], cache: HybridCache) {
        let hash = Self.hashTokens(tokens)
        let fileURL = cacheDir.appendingPathComponent("\(hash).safetensors")

        // Pre-serialize: evaluate all arrays on the calling thread (Metal safety)
        var arrays: [String: MLXArray] = [:]
        var metadata: [String: String] = [:]
        metadata["__num_layers__"] = String(cache.layerCount)

        for (i, layer) in cache.layers.enumerated() {
            switch layer {
            case .attention(let kv):
                metadata["__layer_\(i)_type__"] = "attention"
                metadata["__layer_\(i)_offset__"] = String(kv.offset)
                // Force evaluation before background write
                MLX.eval(kv.keys, kv.values)
                arrays["layer_\(i)_keys"] = kv.keys
                arrays["layer_\(i)_values"] = kv.values

            case .compressedAttention(let ek, let ev, let offset):
                // Store TurboQuant-compressed attention to disk.
                // Saves 5x less disk space than float16 attention.
                metadata["__layer_\(i)_type__"] = "compressed_attention"
                metadata["__layer_\(i)_offset__"] = String(offset)
                metadata["__layer_\(i)_index_bits__"] = String(ek.indexBits)
                metadata["__layer_\(i)_seed__"] = String(ek.seed)
                metadata["__layer_\(i)_sink_count__"] = String(ek.sinkCount)
                metadata["__layer_\(i)_shape__"] = ek.shape.map(String.init).joined(separator: ",")
                metadata["__layer_\(i)_value_shape__"] = ev.shape.map(String.init).joined(separator: ",")
                MLX.eval(ek.indicesPacked, ek.qjlPacked, ek.residualNorms, ek.vectorNorms)
                MLX.eval(ev.indicesPacked, ev.vectorNorms)
                arrays["layer_\(i)_ek_indices"] = ek.indicesPacked
                arrays["layer_\(i)_ek_qjl"] = ek.qjlPacked
                arrays["layer_\(i)_ek_residual"] = ek.residualNorms
                arrays["layer_\(i)_ek_norms"] = ek.vectorNorms
                arrays["layer_\(i)_ev_indices"] = ev.indicesPacked
                arrays["layer_\(i)_ev_norms"] = ev.vectorNorms
                if let sinkK = ek.sinkData {
                    MLX.eval(sinkK)
                    arrays["layer_\(i)_ek_sink"] = sinkK
                }
                if let sinkV = ev.sinkData {
                    MLX.eval(sinkV)
                    arrays["layer_\(i)_ev_sink"] = sinkV
                }

            case .ssm(let ssm):
                metadata["__layer_\(i)_type__"] = "ssm"
                metadata["__layer_\(i)_offset__"] = "0"
                metadata["__layer_\(i)_state_count__"] = String(ssm.state.count)
                for (j, s) in ssm.state.enumerated() {
                    MLX.eval(s)
                    arrays["layer_\(i)_state_\(j)"] = s
                }
            }
        }

        // Compute file size estimate from evaluated arrays
        let estimatedSize = arrays.values.reduce(0) { $0 + $1.nbytes }

        // Record in SQLite index immediately (metadata-only, fast)
        _ = store(tokens: tokens, numTokens: tokens.count, fileSize: estimatedSize)

        // Write safetensors file in background (no Metal calls here)
        Task.detached { [arrays, metadata, fileURL] in
            do {
                try MLX.save(arrays: arrays, metadata: metadata, url: fileURL)
            } catch {
                // File write failed; the SQLite entry remains but the file is missing.
                // On next fetch, the missing-file check will treat it as a miss.
            }
        }
    }

    /// Load a HybridCache from a safetensors file on disk.
    ///
    /// Returns `nil` if the file is missing, corrupt, or the metadata is invalid.
    /// On success, updates the SQLite access timestamp (LRU tracking).
    public func fetchCache(tokens: [Int]) -> HybridCache? {
        let hash = Self.hashTokens(tokens)
        let fileURL = cacheDir.appendingPathComponent("\(hash).safetensors")

        // Quick check: does the SQLite index know about this?
        guard lock.withLock({ _lookupEntry(hash: hash) }) != nil else {
            lock.withLock { _misses += 1 }
            return nil
        }

        // Check the file actually exists
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            // Index entry without file — background write may have failed
            lock.withLock { _misses += 1 }
            return nil
        }

        // Load tensors and metadata
        let loaded: ([String: MLXArray], [String: String])
        do {
            loaded = try loadArraysAndMetadata(url: fileURL)
        } catch {
            lock.withLock { _misses += 1 }
            return nil
        }

        let (arrays, metadata) = loaded

        guard let numLayersStr = metadata["__num_layers__"],
              let numLayers = Int(numLayersStr), numLayers > 0 else {
            lock.withLock { _misses += 1 }
            return nil
        }

        // Reconstruct layers
        var layers: [LayerCacheEntry] = []
        layers.reserveCapacity(numLayers)

        for i in 0..<numLayers {
            guard let layerType = metadata["__layer_\(i)_type__"] else {
                lock.withLock { _misses += 1 }
                return nil
            }

            switch layerType {
            case "attention":
                guard let keys = arrays["layer_\(i)_keys"],
                      let values = arrays["layer_\(i)_values"] else {
                    lock.withLock { _misses += 1 }
                    return nil
                }
                let offset = Int(metadata["__layer_\(i)_offset__"] ?? "0") ?? 0
                layers.append(.attention(KVCacheLayer(keys: keys, values: values, offset: offset)))

            case "compressed_attention":
                // Reconstruct TurboQuant-compressed attention from disk
                guard let ekIndices = arrays["layer_\(i)_ek_indices"],
                      let ekQjl = arrays["layer_\(i)_ek_qjl"],
                      let ekResidual = arrays["layer_\(i)_ek_residual"],
                      let ekNorms = arrays["layer_\(i)_ek_norms"],
                      let evIndices = arrays["layer_\(i)_ev_indices"],
                      let evNorms = arrays["layer_\(i)_ev_norms"] else {
                    lock.withLock { _misses += 1 }
                    return nil
                }
                let offset = Int(metadata["__layer_\(i)_offset__"] ?? "0") ?? 0
                let indexBits = Int(metadata["__layer_\(i)_index_bits__"] ?? "3") ?? 3
                let seed = Int(metadata["__layer_\(i)_seed__"] ?? "42") ?? 42
                let shapeStr = metadata["__layer_\(i)_shape__"] ?? ""
                let shape = shapeStr.split(separator: ",").compactMap { Int($0) }
                let valShapeStr = metadata["__layer_\(i)_value_shape__"] ?? ""
                let valShape = valShapeStr.split(separator: ",").compactMap { Int($0) }

                // Load optional sink data (full-precision first N tokens)
                let ekSink = arrays["layer_\(i)_ek_sink"]
                let evSink = arrays["layer_\(i)_ev_sink"]

                let ek = EncodedKeys(
                    indicesPacked: ekIndices, qjlPacked: ekQjl,
                    residualNorms: ekResidual, vectorNorms: ekNorms,
                    shape: shape, indexBits: indexBits, seed: seed,
                    sinkData: ekSink
                )
                let ev = EncodedValues(
                    indicesPacked: evIndices, vectorNorms: evNorms,
                    shape: valShape, indexBits: indexBits, seed: seed,
                    sinkData: evSink
                )
                layers.append(.compressedAttention(ek, ev, offset))

            case "ssm":
                let stateCount = Int(metadata["__layer_\(i)_state_count__"] ?? "0") ?? 0
                var stateArrays: [MLXArray] = []
                stateArrays.reserveCapacity(stateCount)
                for j in 0..<stateCount {
                    guard let s = arrays["layer_\(i)_state_\(j)"] else {
                        lock.withLock { _misses += 1 }
                        return nil
                    }
                    stateArrays.append(s)
                }
                layers.append(.ssm(SSMStateLayer(state: stateArrays)))

            default:
                lock.withLock { _misses += 1 }
                return nil
            }
        }

        // Update access time in SQLite
        lock.withLock {
            _updateAccess(hash: hash)
            _hits += 1
        }

        return HybridCache(layers: layers)
    }

    // MARK: - Token Hashing

    /// SHA-256 of JSON-serialized token array. Matches Python VMLX format.
    public static func hashTokens(_ tokens: [Int]) -> String {
        // JSON format: compact, no spaces (matches Python json.dumps(separators=(",",":")))
        let json = "[" + tokens.map(String.init).joined(separator: ",") + "]"
        let hash = SHA256.hash(data: Data(json.utf8))
        return hash.map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Private SQLite Operations

    private func _openDatabase() {
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        guard sqlite3_open_v2(dbPath, &db, flags, nil) == SQLITE_OK else { return }
        // WAL mode for concurrent reads
        sqlite3_exec(db, "PRAGMA journal_mode=WAL", nil, nil, nil)
        sqlite3_exec(db, "PRAGMA synchronous=NORMAL", nil, nil, nil)
    }

    private func _createTable() {
        let sql = """
            CREATE TABLE IF NOT EXISTS cache_entries (
                token_hash TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                num_tokens INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 1,
                metadata TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed);
            """
        sqlite3_exec(db, sql, nil, nil, nil)
    }

    private func _lookupEntry(hash: String) -> (fileName: String, numTokens: Int, metadata: String?)? {
        var stmt: OpaquePointer?
        let sql = "SELECT file_name, num_tokens, metadata FROM cache_entries WHERE token_hash = ?"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return nil }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, hash, -1, Self.sqliteTransient)
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }

        let fileName = String(cString: sqlite3_column_text(stmt, 0))
        let numTokens = Int(sqlite3_column_int(stmt, 1))
        let metadata: String? = sqlite3_column_type(stmt, 2) != SQLITE_NULL
            ? String(cString: sqlite3_column_text(stmt, 2)) : nil
        return (fileName, numTokens, metadata)
    }

    private func _updateAccess(hash: String) {
        var stmt: OpaquePointer?
        let sql = "UPDATE cache_entries SET last_accessed = ?, access_count = access_count + 1 WHERE token_hash = ?"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_double(stmt, 1, CFAbsoluteTimeGetCurrent())
        sqlite3_bind_text(stmt, 2, hash, -1, Self.sqliteTransient)
        sqlite3_step(stmt)
    }

    private func _deleteEntry(hash: String) {
        var stmt: OpaquePointer?
        let sql = "DELETE FROM cache_entries WHERE token_hash = ?"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
        defer { sqlite3_finalize(stmt) }
        sqlite3_bind_text(stmt, 1, hash, -1, Self.sqliteTransient)
        sqlite3_step(stmt)

        // Also delete the file
        let filePath = cacheDir.appendingPathComponent("\(hash).safetensors").path
        try? FileManager.default.removeItem(atPath: filePath)
    }

    /// Internal entry count (must be called under lock).
    private func _entryCount() -> Int {
        var stmt: OpaquePointer?
        let sql = "SELECT COUNT(*) FROM cache_entries"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return 0 }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
        return Int(sqlite3_column_int(stmt, 0))
    }

    /// Internal total size (must be called under lock).
    private func _totalSizeBytes() -> Int {
        var stmt: OpaquePointer?
        let sql = "SELECT COALESCE(SUM(file_size), 0) FROM cache_entries"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return 0 }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return 0 }
        return Int(sqlite3_column_int64(stmt, 0))
    }

    /// LRU eviction when over max size. Must be called under lock.
    private func _evictIfNeeded() {
        while _totalSizeBytes() > maxSizeBytes {
            var stmt: OpaquePointer?
            let sql = "SELECT token_hash FROM cache_entries ORDER BY last_accessed ASC LIMIT 1"
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
            let rc = sqlite3_step(stmt)
            guard rc == SQLITE_ROW else {
                sqlite3_finalize(stmt)
                return
            }
            let hash = String(cString: sqlite3_column_text(stmt, 0))
            sqlite3_finalize(stmt)
            _deleteEntry(hash: hash)
        }
    }
}
