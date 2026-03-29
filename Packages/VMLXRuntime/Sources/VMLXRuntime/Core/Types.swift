import Foundation
import MLX

// MARK: - Sampling Parameters

/// Parameters controlling token sampling during inference.
public struct SamplingParams: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var repetitionPenalty: Float
    public var stop: [String]
    public var stopTokenIds: [Int]

    /// True when temperature is zero (deterministic / argmax sampling).
    public var isGreedy: Bool { temperature == 0 }

    public init(
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9,
        topK: Int = 0,
        minP: Float = 0.0,
        repetitionPenalty: Float = 1.0,
        stop: [String] = [],
        stopTokenIds: [Int] = []
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.stop = stop
        self.stopTokenIds = stopTokenIds
    }
}

// MARK: - Request Status

/// Lifecycle state of an inference request.
public enum RequestStatus: Int, Sendable, Comparable {
    case waiting = 0
    case running = 1
    case preempted = 2
    case finishedStopped = 3
    case finishedLengthCapped = 4
    case finishedAborted = 5

    /// True when the request has reached a terminal state.
    public var isFinished: Bool { rawValue >= 3 }

    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

// MARK: - Finish Reason

/// Why an inference request stopped generating tokens.
public enum FinishReason: String, Sendable {
    case stop
    case length
    case abort
    case toolCalls = "tool_calls"
}

// MARK: - Inference Request

/// A single inference request carrying prompt tokens, sampling config,
/// and mutable output state that accumulates during generation.
public struct InferenceRequest: @unchecked Sendable, Identifiable {
    public let id: String
    public let requestId: String
    public var promptTokenIds: [Int]
    public var samplingParams: SamplingParams
    public let arrivalTime: Date
    public var priority: Int
    public var status: RequestStatus

    // Output accumulation
    public var outputTokenIds: [Int]
    public var outputText: String
    public var finishReason: FinishReason?

    // Cache state
    public var promptCache: HybridCache?
    public var cachedTokens: Int
    public var remainingTokenIds: [Int]?
    public var blockTableIds: [Int]?
    public var sharedPrefixBlocks: Int

    // Multimodal fields
    public var pixelValues: MLXArray?
    public var imageGridTHW: [Int]?
    public var attentionMask: MLXArray?
    public var isMultimodal: Bool

    // Reasoning / thinking
    public var enableThinking: Bool
    public var reasoningEffort: String

    // MARK: - Computed Properties

    public var numPromptTokens: Int { promptTokenIds.count }
    public var numOutputTokens: Int { outputTokenIds.count }
    public var numTotalTokens: Int { numPromptTokens + numOutputTokens }
    public var isFinished: Bool { status.isFinished }

    // MARK: - Init

    public init(
        requestId: String,
        promptTokenIds: [Int],
        samplingParams: SamplingParams = SamplingParams(),
        priority: Int = 0,
        enableThinking: Bool = false,
        reasoningEffort: String = "medium",
        isMultimodal: Bool = false
    ) {
        self.id = requestId
        self.requestId = requestId
        self.promptTokenIds = promptTokenIds
        self.samplingParams = samplingParams
        self.arrivalTime = Date()
        self.priority = priority
        self.status = .waiting
        self.outputTokenIds = []
        self.outputText = ""
        self.finishReason = nil
        self.promptCache = nil
        self.cachedTokens = 0
        self.remainingTokenIds = nil
        self.blockTableIds = nil
        self.sharedPrefixBlocks = 0
        self.pixelValues = nil
        self.imageGridTHW = nil
        self.attentionMask = nil
        self.isMultimodal = isMultimodal
        self.enableThinking = enableThinking
        self.reasoningEffort = reasoningEffort
    }

    // MARK: - Mutations

    /// Append a newly generated token to the output sequence.
    public mutating func appendOutputToken(_ tokenId: Int) {
        outputTokenIds.append(tokenId)
    }

    /// Transition to a finished state with the given reason.
    public mutating func finish(reason: FinishReason) {
        switch reason {
        case .stop:
            status = .finishedStopped
        case .length:
            status = .finishedLengthCapped
        case .abort:
            status = .finishedAborted
        case .toolCalls:
            status = .finishedStopped
        }
        finishReason = reason
    }
}

// MARK: - Request Output

/// A snapshot of generation output, typically streamed back to the caller
/// after each decode step.
public struct RequestOutput: Sendable {
    public let requestId: String
    public var newTokenIds: [Int]
    public var newText: String
    public var outputTokenIds: [Int]
    public var outputText: String
    public var finishReason: FinishReason?
    public var numTotalTokens: Int

    public init(
        requestId: String,
        newTokenIds: [Int] = [],
        newText: String = "",
        outputTokenIds: [Int] = [],
        outputText: String = "",
        finishReason: FinishReason? = nil,
        numTotalTokens: Int = 0
    ) {
        self.requestId = requestId
        self.newTokenIds = newTokenIds
        self.newText = newText
        self.outputTokenIds = outputTokenIds
        self.outputText = outputText
        self.finishReason = finishReason
        self.numTotalTokens = numTotalTokens
    }
}

// MARK: - Cache Detail

/// Describes the type of cache hit for diagnostics and logging.
public enum CacheDetail: String, Sendable {
    case full
    case prefix
    case paged
    case disk
    case memory
    case tq = "+tq"
}
