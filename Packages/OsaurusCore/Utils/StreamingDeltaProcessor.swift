//
//  StreamingDeltaProcessor.swift
//  osaurus
//
//  Shared streaming delta processing pipeline used by both ChatView (chat mode)
//  and WorkSession (work mode). Handles delta buffering, <think> tag parsing,
//  adaptive flush tuning, and throttled UI sync.
//

import Foundation

/// Processes streaming LLM deltas into a ChatTurn with buffering,
/// thinking tag parsing, and throttled UI updates.
@MainActor
final class StreamingDeltaProcessor {

    // MARK: - State

    private var turn: ChatTurn
    private let onSync: (() -> Void)?

    /// Model-specific delta preprocessing (resolved once from modelId + options)
    private let modelId: String
    private let modelOptions: [String: ModelOptionValue]
    private var middleware: StreamingMiddleware?

    /// Delta buffering
    private var deltaBuffer = ""

    /// Fallback timer — safety net for push-based consumers (e.g. WorkSession
    /// delegate callbacks) where no more deltas may arrive to trigger an inline flush.
    private var flushTimer: Timer?
    private static let fallbackFlushInterval: TimeInterval = 0.1

    /// Thinking tag parsing
    private var isInsideThinking = false
    private var pendingTagBuffer = ""

    /// Adaptive flush tuning — tracked lengths avoid calling String.count on large buffers
    private var contentLength = 0
    private var thinkingLength = 0
    private var flushIntervalMs: Double = 50
    private var maxBufferSize: Int = 256
    private var longestFlushMs: Double = 0

    /// Sync batching — flush parses tags and appends to turn,
    /// sync triggers UI update at a slower cadence to prevent churn.
    private var hasPendingContent = false
    private var lastSyncTime = Date()
    private var lastFlushTime = Date()
    private var syncCount = 0

    // MARK: - Init

    init(
        turn: ChatTurn,
        modelId: String = "",
        modelOptions: [String: ModelOptionValue] = [:],
        onSync: (() -> Void)? = nil
    ) {
        self.turn = turn
        self.modelId = modelId
        self.modelOptions = modelOptions
        self.onSync = onSync
        self.middleware = StreamingMiddlewareResolver.resolve(for: modelId, modelOptions: modelOptions)
    }

    // MARK: - Public API

    /// Receive a streaming delta. Buffers it, checks flush conditions inline
    /// (O(1) integer comparisons), and flushes if thresholds are met.
    func receiveDelta(_ delta: String) {
        guard !delta.isEmpty else { return }

        let processed = middleware?.process(delta) ?? delta
        guard !processed.isEmpty else { return }
        deltaBuffer += processed

        let now = Date()
        let timeSinceFlush = now.timeIntervalSince(lastFlushTime) * 1000
        recomputeFlushTuning()

        if deltaBuffer.count >= maxBufferSize || timeSinceFlush >= flushIntervalMs {
            flush()
            syncIfNeeded(now: now)
        }

        // Fallback timer in case no more deltas arrive
        if flushTimer == nil, !deltaBuffer.isEmpty {
            flushTimer = Timer.scheduledTimer(
                withTimeInterval: Self.fallbackFlushInterval,
                repeats: false
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.flush()
                    self.syncToTurn()
                }
            }
        }
    }

    /// Force-flush all buffered deltas: parse thinking tags, route to turn.
    func flush() {
        invalidateTimer()
        guard !deltaBuffer.isEmpty else { return }

        let flushStart = Date()
        var textToProcess = pendingTagBuffer + deltaBuffer
        pendingTagBuffer = ""
        deltaBuffer = ""

        parseAndRoute(&textToProcess)

        lastFlushTime = Date()
        let flushMs = lastFlushTime.timeIntervalSince(flushStart) * 1000
        if flushMs > longestFlushMs { longestFlushMs = flushMs }
    }

    /// Finalize streaming: drain remaining buffers and partial tags, sync to UI.
    func finalize() {
        invalidateTimer()

        if !deltaBuffer.isEmpty || !pendingTagBuffer.isEmpty {
            let remaining = pendingTagBuffer + deltaBuffer
            pendingTagBuffer = ""
            deltaBuffer = ""
            if isInsideThinking {
                appendThinking(remaining)
            } else {
                appendContent(remaining)
            }
        }

        syncToTurn()
    }

    /// Reset for a new streaming session with a new turn.
    func reset(turn: ChatTurn) {
        invalidateTimer()
        self.turn = turn
        deltaBuffer = ""
        isInsideThinking = false
        pendingTagBuffer = ""
        contentLength = 0
        thinkingLength = 0
        flushIntervalMs = 50
        maxBufferSize = 256
        longestFlushMs = 0
        hasPendingContent = false
        lastSyncTime = Date()
        lastFlushTime = Date()
        syncCount = 0
        middleware = StreamingMiddlewareResolver.resolve(for: modelId, modelOptions: modelOptions)
    }

    // MARK: - Private

    private func invalidateTimer() {
        flushTimer?.invalidate()
        flushTimer = nil
    }

    private func appendContent(_ s: String) {
        guard !s.isEmpty else { return }
        turn.appendContent(s)
        contentLength += s.count
        hasPendingContent = true
    }

    private func appendThinking(_ s: String) {
        guard !s.isEmpty else { return }
        turn.appendThinking(s)
        thinkingLength += s.count
        hasPendingContent = true
    }

    private func syncToTurn() {
        guard hasPendingContent else { return }
        syncCount += 1
        turn.notifyContentChanged()
        hasPendingContent = false
        lastSyncTime = Date()
        onSync?()
    }

    private func syncIfNeeded(now: Date) {
        let totalChars = contentLength + thinkingLength
        // Sync to UI on every flush for per-token smoothness.
        // At high tok/s, each sync triggers a SwiftUI layout pass.
        // Back off on very long outputs to prevent layout thrash.
        let syncIntervalMs: Double =
            switch totalChars {
            case 0 ..< 5_000: 0    // Every token
            case 5_000 ..< 15_000: 16  // ~60fps
            default: 33               // ~30fps
            }

        let timeSinceSync = now.timeIntervalSince(lastSyncTime) * 1000
        if (syncCount == 0 && hasPendingContent)
            || (timeSinceSync >= syncIntervalMs && hasPendingContent)
        {
            syncToTurn()
        }
    }

    private func recomputeFlushTuning() {
        let totalChars = contentLength + thinkingLength

        // Flush every token for smooth per-token streaming on local inference.
        // At 80 tok/s = 12ms/token, we flush on every delta arrival.
        // Only back off for very long outputs where layout becomes expensive.
        switch totalChars {
        case 0 ..< 5_000:
            flushIntervalMs = 0; maxBufferSize = 1  // Every token
        case 5_000 ..< 15_000:
            flushIntervalMs = 16; maxBufferSize = 32  // ~60fps
        default:
            flushIntervalMs = 33; maxBufferSize = 64  // ~30fps
        }

        // Back off only if layout is genuinely slow (>50ms)
        if longestFlushMs > 50 {
            flushIntervalMs = min(100, max(flushIntervalMs, longestFlushMs * 1.2))
        }
    }

    // MARK: - Thinking Tag Parsing

    /// Partial tag prefixes for `<think>` and `</think>`, longest first.
    /// Only match 4+ char prefixes to avoid false positives on "<" or "<t" in content.
    private static let openPartials = ["<think", "<thin", "<thi"]
    private static let closePartials = ["</think", "</thin", "</thi"]

    private func parseAndRoute(_ text: inout String) {
        // GPT-OSS <|channel|> tags are handled by ChannelTagMiddleware
        // BEFORE reaching this parser. By the time text arrives here,
        // channel tags have been transformed to standard <think>/</ think>.
        while !text.isEmpty {
            if isInsideThinking {
                if let closeRange = text.range(of: "</think>") {
                    appendThinking(String(text[..<closeRange.lowerBound]))
                    text = String(text[closeRange.upperBound...])
                    isInsideThinking = false
                    syncToTurn()
                } else if let partial = Self.closePartials.first(where: { text.hasSuffix($0) }) {
                    appendThinking(String(text.dropLast(partial.count)))
                    pendingTagBuffer = String(text.suffix(partial.count))
                    text = ""
                } else {
                    appendThinking(text)
                    text = ""
                }
            } else {
                if let openRange = text.range(of: "<think>") {
                    appendContent(String(text[..<openRange.lowerBound]))
                    text = String(text[openRange.upperBound...])
                    isInsideThinking = true
                } else if let partial = Self.openPartials.first(where: { text.hasSuffix($0) }) {
                    appendContent(String(text.dropLast(partial.count)))
                    pendingTagBuffer = String(text.suffix(partial.count))
                    text = ""
                } else {
                    appendContent(text)
                    text = ""
                }
            }
        }
    }
}
