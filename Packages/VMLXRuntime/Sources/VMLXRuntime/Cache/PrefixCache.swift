import Foundation
import os

// MARK: - Prefix Cache (Trie-based)

/// A trie-based token prefix cache with LRU eviction.
///
/// Stores `HybridCache` entries keyed by token sequences. On fetch, returns one of
/// three possible results:
/// - **Exact match**: all requested tokens found in trie.
/// - **Shorter prefix**: the trie has a prefix of the requested tokens — returns the
///   cached entry plus the remaining (uncached) tokens.
/// - **Longer prefix**: the requested tokens are a prefix of a cached entry — truncates
///   the cached entry to the request length (only if `canTruncate` is true).
///
/// Thread-safe via `OSAllocatedUnfairLock`.
public final class PrefixCache: @unchecked Sendable {

    // MARK: - Trie Node

    private final class TrieNode {
        var children: [Int: TrieNode] = [:]
        var cache: HybridCache?
        var tokens: [Int]?  // Full token sequence stored at this leaf
    }

    // MARK: - State

    private let lock = OSAllocatedUnfairLock()
    private let root = TrieNode()
    private let maxEntries: Int

    /// LRU tracking: ordered list of token sequences, most recent at end.
    private var lruOrder: [[Int]] = []
    private var entryCount: Int = 0

    // MARK: - Stats

    public private(set) var hits: Int = 0
    public private(set) var misses: Int = 0

    // MARK: - Init

    public init(maxEntries: Int = 100) {
        self.maxEntries = maxEntries
    }

    /// Clear all cached entries.
    public func clear() {
        lock.withLock {
            root.children.removeAll()
            root.cache = nil
            lruOrder.removeAll()
            entryCount = 0
            hits = 0
            misses = 0
        }
    }

    // MARK: - Public API

    /// Fetch a cached entry for the given token sequence.
    ///
    /// Returns `(cache, remainingTokens)`:
    /// - If `cache` is non-nil, `remainingTokens` contains tokens not covered by the cache.
    /// - If `cache` is nil, `remainingTokens == tokens` (full miss).
    public func fetch(tokens: [Int]) -> (HybridCache?, [Int]) {
        lock.withLock {
            // Walk the trie following the token sequence
            var current = root
            var lastCacheNode: TrieNode?
            var lastCacheDepth: Int = 0

            for (i, token) in tokens.enumerated() {
                guard let next = current.children[token] else { break }
                current = next
                if current.cache != nil {
                    lastCacheNode = current
                    lastCacheDepth = i + 1
                }
            }

            // Case 1 & 2: Exact match or shorter prefix match
            if let node = lastCacheNode, let cache = node.cache {
                if lastCacheDepth == tokens.count {
                    // Exact match
                    _touchLRU(node.tokens!)
                    hits += 1
                    return (cache, [])
                } else {
                    // Shorter prefix: cache covers the first lastCacheDepth tokens
                    _touchLRU(node.tokens!)
                    hits += 1
                    return (cache, Array(tokens[lastCacheDepth...]))
                }
            }

            // Case 3: Longer prefix — a cached entry whose tokens START with ours.
            // We walked the entire `tokens` array without breaking, and some descendant
            // node holds a cache entry for a longer sequence.
            if let longerMatch = _findLongerMatch(tokens: tokens) {
                if longerMatch.canTruncate {
                    if let truncated = longerMatch.truncated(to: tokens.count) {
                        hits += 1
                        return (truncated, [])
                    }
                }
                // Can't truncate (hybrid model with SSM layers) — treat as miss
            }

            misses += 1
            return (nil, tokens)
        }
    }

    /// Store a cache entry for the given token sequence.
    public func store(tokens: [Int], cache: HybridCache) {
        lock.withLock {
            // Build the trie path
            var current = root
            for token in tokens {
                if current.children[token] == nil {
                    current.children[token] = TrieNode()
                }
                current = current.children[token]!
            }

            // Store at the leaf
            let isNew = current.cache == nil
            current.cache = cache
            current.tokens = tokens

            if isNew {
                entryCount += 1
                lruOrder.append(tokens)
                _evictIfNeeded()
            } else {
                _touchLRU(tokens)
            }
        }
    }

    /// Remove the cache entry for an exact token sequence.
    public func invalidate(tokens: [Int]) {
        lock.withLock {
            var current = root
            for token in tokens {
                guard let next = current.children[token] else { return }
                current = next
            }
            if current.cache != nil {
                current.cache = nil
                current.tokens = nil
                entryCount -= 1
                lruOrder.removeAll { $0 == tokens }
            }
        }
    }

    /// The number of cached entries.
    public var count: Int {
        lock.withLock { entryCount }
    }

    // MARK: - Private Helpers

    /// Walk the trie to the end of `tokens`, then search the subtree for any cache entry.
    private func _findLongerMatch(tokens: [Int]) -> HybridCache? {
        var current = root
        for token in tokens {
            guard let next = current.children[token] else { return nil }
            current = next
        }
        // `current` is at the end of our tokens — look for a cached descendant
        return _findFirstCache(in: current)
    }

    /// Depth-first search for the first cache entry in a subtree.
    private func _findFirstCache(in node: TrieNode) -> HybridCache? {
        if let cache = node.cache { return cache }
        for (_, child) in node.children {
            if let cache = _findFirstCache(in: child) { return cache }
        }
        return nil
    }

    /// Move `tokens` to the end of the LRU list (most recently used).
    private func _touchLRU(_ tokens: [Int]) {
        if let idx = lruOrder.firstIndex(of: tokens) {
            lruOrder.remove(at: idx)
            lruOrder.append(tokens)
        }
    }

    /// Evict the least recently used entries until we're at or below `maxEntries`.
    private func _evictIfNeeded() {
        while entryCount > maxEntries && !lruOrder.isEmpty {
            let oldest = lruOrder.removeFirst()
            _removeFromTrie(oldest)
            entryCount -= 1
        }
    }

    /// Remove a cache entry from the trie (does NOT prune empty intermediate nodes).
    private func _removeFromTrie(_ tokens: [Int]) {
        var current = root
        for token in tokens {
            guard let next = current.children[token] else { return }
            current = next
        }
        current.cache = nil
        current.tokens = nil
    }
}
