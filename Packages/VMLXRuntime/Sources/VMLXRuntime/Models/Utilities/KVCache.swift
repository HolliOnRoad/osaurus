//
//  KVCache.swift
//  VMLXRuntime
//
//  Ported from mlx-swift-lm's MLXLMCommon/KVCache.swift
//  Only the types needed for our native model implementations.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - KVCache Protocol

/// Interface for Key/Value cache for LLMs.
public protocol VMLXKVCache: Evaluatable {
    var offset: Int { get }
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    var state: [MLXArray] { get set }
    var isTrimmable: Bool { get }
    @discardableResult func trim(_ n: Int) -> Int
    func copy() -> any VMLXKVCache
}

// MARK: - Base KV Cache

open class VMLXBaseKVCache: VMLXKVCache {
    public var offset: Int = 0

    public func innerState() -> [MLXArray] { [] }

    open func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("update(keys:values:) must be implemented by subclass")
    }

    open var state: [MLXArray] {
        get { [] }
        set {}
    }

    open var isTrimmable: Bool { false }

    @discardableResult
    open func trim(_ n: Int) -> Int { 0 }

    open func copy() -> any VMLXKVCache {
        fatalError("copy() must be implemented by subclass")
    }
}

// MARK: - Simple KV Cache

/// Standard KV cache for attention layers.
public class VMLXKVCacheSimple: VMLXBaseKVCache {
    internal var keys: MLXArray?
    internal var values: MLXArray?
    public var step = 256

    public override init() {
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += keys.dim(2)
        self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.values?[.ellipsis, previous ..< self.offset, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<self.offset, 0...]
        return (returnedKeys, returnedValues)
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset == keys.dim(2) {
                return [keys, values]
            } else {
                return [keys[.ellipsis, ..<offset, 0...], values[.ellipsis, ..<offset, 0...]]
            }
        }
        set {
            if newValue.count >= 2 {
                self.keys = newValue[0]
                self.values = newValue[1]
                self.offset = newValue[0].dim(2)
            }
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any VMLXKVCache {
        let new = VMLXKVCacheSimple()
        new.step = self.step
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        return new
    }
}

// MARK: - Arrays Cache (for SSM state)

/// Base cache for array-based state storage (SSM models).
public class VMLXArraysCache: VMLXBaseKVCache {
    private var cache: [MLXArray?]
    internal var leftPadding: MLXArray?

    public init(size: Int, leftPadding: [Int]? = nil) {
        self.cache = Array(repeating: nil, count: size)
        self.leftPadding = leftPadding.map { MLXArray($0) }
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        cache.compactMap { $0 }
    }

    public subscript(index: Int) -> MLXArray? {
        get { cache[index] }
        set { cache[index] = newValue }
    }

    public override var state: [MLXArray] {
        get { cache.compactMap { $0 } }
        set { cache = newValue.map { $0 as MLXArray? } }
    }

    public override func copy() -> any VMLXKVCache {
        let new = VMLXArraysCache(size: cache.count)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.offset = self.offset
        new.leftPadding = self.leftPadding
        return new
    }

    /// Create attention mask based on left padding.
    public func makeMask(N: Int) -> MLXArray? {
        if cache[0] == nil, let leftPadding = leftPadding {
            return MLXArray(0 ..< N) .>= leftPadding[0..., .newAxis]
        } else {
            return nil
        }
    }
}

// MARK: - Mamba Cache

/// Simple cache for Mamba-style state space models.
public class VMLXMambaCache: VMLXArraysCache {
    public init(leftPadding: [Int]? = nil) {
        super.init(size: 2, leftPadding: leftPadding)
    }

    public override func copy() -> any VMLXKVCache {
        let new = VMLXMambaCache()
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.offset = self.offset
        new.leftPadding = self.leftPadding
        return new
    }
}

// MARK: - Attention Mask Helpers

/// Create a causal attention mask.
public func vmlxCreateCausalMask(
    n: Int, offset: Int, windowSize: Int? = nil
) -> MLXArray {
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]
    var mask = linds .>= rinds
    if let windowSize {
        mask = mask & (linds .< rinds + windowSize)
    }
    return mask
}

/// Create an attention mask for scaled dot product attention.
/// Uses symbolic `.causal` mode when possible (avoids materializing full mask array).
/// Falls back to `.array(...)` only when cache offset is non-zero (resumed generation).
public func vmlxCreateAttentionMask(
    h: MLXArray, cache: VMLXKVCache?
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let t = h.dim(1)
    if t > 1 {
        let offset = cache?.offset ?? 0
        if offset == 0 {
            // Fresh prefill: symbolic causal mask (no array materialization)
            return .causal
        }
        // Resumed after cache: need explicit mask with offset
        return .array(vmlxCreateCausalMask(n: t, offset: offset))
    }
    return .none
}

/// Create an SSM mask for GatedDeltaNet layers.
public func vmlxCreateSSMMask(h: MLXArray, cache: VMLXMambaCache?) -> MLXArray? {
    if let cache {
        return cache.makeMask(N: h.dim(1))
    }
    return nil
}

// MARK: - Attention with Cache

/// Perform scaled dot product attention with automatic cache update.
public func vmlxAttentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: VMLXKVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )
    }
    let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
    return MLXFast.scaledDotProductAttention(
        queries: queries, keys: cachedKeys, values: cachedValues,
        scale: scale, mask: mask
    )
}
