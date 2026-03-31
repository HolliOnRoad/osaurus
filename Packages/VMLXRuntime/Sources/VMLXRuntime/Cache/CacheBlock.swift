import Foundation
import CryptoKit
import MLX

/// Fixed-size block of KV cache data. Reference-counted for COW sharing.
public final class CacheBlock: @unchecked Sendable {
    public let blockId: Int
    public let blockSize: Int
    public var refCount: Int = 0
    public var blockHash: BlockHash?
    public var tokenCount: Int = 0
    var prevFreeBlock: CacheBlock?
    var nextFreeBlock: CacheBlock?
    /// Per-layer cache data stored in this block.
    /// Each entry is one model layer's slice for the block's token range.
    /// Uses LayerCacheEntry: .attention(KVCacheLayer) or .ssm(SSMStateLayer).
    /// For hybrid models, last block stores SSM as .ssm (cumulative);
    /// non-last blocks mark SSM positions as nil (skip).
    /// TurboQuant Phase 5 will add .compressedAttention case.
    public var cacheData: [LayerCacheEntry?]?
    public var lastAccess: Date = Date()

    public init(blockId: Int, blockSize: Int) {
        self.blockId = blockId
        self.blockSize = blockSize
    }

    public func isFull(blockSize: Int) -> Bool { tokenCount >= blockSize }
    public var isShared: Bool { refCount > 1 }
    public func touch() { lastAccess = Date() }

    public func reset() {
        refCount = 0; blockHash = nil; tokenCount = 0
        cacheData = nil; prevFreeBlock = nil; nextFreeBlock = nil
    }

    /// SHA-256 hash chain: each block's hash depends on parent + token content.
    public static func computeBlockHash(
        parentHash: BlockHash?, tokenIds: [Int], extraKeys: [MLXArray]? = nil
    ) -> BlockHash {
        var hasher = SHA256()
        if let parent = parentHash { hasher.update(data: parent.data) }
        tokenIds.withUnsafeBufferPointer { hasher.update(bufferPointer: UnsafeRawBufferPointer($0)) }
        if let extras = extraKeys {
            for array in extras {
                let shape = array.shape
                shape.withUnsafeBufferPointer { hasher.update(bufferPointer: UnsafeRawBufferPointer($0)) }
            }
        }
        return BlockHash(Data(hasher.finalize()))
    }
}

public struct BlockHash: Hashable, Sendable {
    public let data: Data
    public init(_ data: Data) { self.data = data }
    public var count: Int { data.count }
    public var hexString: String { data.map { String(format: "%02x", $0) }.joined() }
}
