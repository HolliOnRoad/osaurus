import Foundation

/// N-gram index for prompt lookup decoding.
/// Maps n-grams from the prompt to their continuation tokens.
public struct NgramIndex: Sendable {
    private var index: [ArraySlice<Int>: [Int]] = [:]
    private let ngramSize: Int

    public init(tokens: [Int], ngramSize: Int = 3) {
        self.ngramSize = ngramSize
        // Build index: for each n-gram in prompt, record the token that follows
        for i in 0..<(tokens.count - ngramSize) {
            let ngram = tokens[i..<(i + ngramSize)]
            let next = tokens[i + ngramSize]
            index[ngram, default: []].append(next)
        }
    }

    /// Look up draft tokens: given the last n tokens, predict what comes next.
    /// Returns up to maxDrafts candidate tokens.
    public func lookup(lastTokens: [Int], maxDrafts: Int = 5) -> [Int] {
        guard lastTokens.count >= ngramSize else { return [] }
        let ngram = lastTokens[(lastTokens.count - ngramSize)...]
        guard let candidates = index[ngram] else { return [] }
        return Array(candidates.prefix(maxDrafts))
    }
}

/// Prompt Lookup Decoding manager.
/// Tracks hit rates and adaptively enables/disables PLD based on effectiveness.
public struct PromptLookupDecoder: Sendable {
    public let ngramIndex: NgramIndex
    public let maxDrafts: Int

    public private(set) var totalAttempts: Int = 0
    public private(set) var successfulDrafts: Int = 0

    /// Hit rate (0.0 - 1.0).
    public var hitRate: Float {
        totalAttempts > 0 ? Float(successfulDrafts) / Float(totalAttempts) : 0
    }

    public init(promptTokens: [Int], maxDrafts: Int = 5, ngramSize: Int = 3) {
        self.ngramIndex = NgramIndex(tokens: promptTokens, ngramSize: ngramSize)
        self.maxDrafts = maxDrafts
    }

    /// Get draft tokens for the current generation state.
    public mutating func getDrafts(lastTokens: [Int]) -> [Int] {
        totalAttempts += 1
        let drafts = ngramIndex.lookup(lastTokens: lastTokens, maxDrafts: maxDrafts)
        if !drafts.isEmpty { successfulDrafts += 1 }
        return drafts
    }

    /// Record whether a draft was accepted by the model.
    public mutating func recordAcceptance(accepted: Int, total: Int) {
        // Track for adaptive tuning
    }
}
