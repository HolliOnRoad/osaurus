//
//  VMLXServiceBridge.swift
//  osaurus
//
//  Bridges VMLXRuntime's VMLXService into Osaurus's ToolCapableService protocol.
//  Handles type mapping between Osaurus ChatMessage/Tool/GenerationParameters and
//  VMLXRuntime's VMLXChatMessage/VMLXToolDefinition/SamplingParams.
//

import Foundation
import VMLXRuntime

// MARK: - Bridge Actor

/// Adapts VMLXService (from VMLXRuntime) to Osaurus's ToolCapableService protocol,
/// enabling it to participate in Osaurus's ModelServiceRouter alongside MLXService
/// and FoundationModelService.
actor VMLXServiceBridge: ToolCapableService {

    nonisolated let id: String = "vmlx"

    private let service: VMLXService

    init(service: VMLXService = .shared) {
        self.service = service
    }

    // MARK: - ModelService

    nonisolated func isAvailable() -> Bool {
        service.isAvailable()
    }

    nonisolated func handles(requestedModel: String?) -> Bool {
        service.handles(requestedModel: requestedModel)
    }

    func generateOneShot(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        requestedModel: String?
    ) async throws -> String {
        let vmlxMessages = messages.map { $0.toVMLX() }
        let params = parameters.toSamplingParams()
        return try await service.generateOneShot(
            messages: vmlxMessages,
            params: params,
            requestedModel: requestedModel
        )
    }

    func streamDeltas(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        requestedModel: String?,
        stopSequences: [String]
    ) async throws -> AsyncThrowingStream<String, Error> {
        let vmlxMessages = messages.map { $0.toVMLX() }
        var params = parameters.toSamplingParams()
        params.stop = stopSequences
        return try await service.streamDeltas(
            messages: vmlxMessages,
            params: params,
            requestedModel: requestedModel,
            stopSequences: stopSequences
        )
    }

    // MARK: - ToolCapableService

    func respondWithTools(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        stopSequences: [String],
        tools: [Tool],
        toolChoice: ToolChoiceOption?,
        requestedModel: String?
    ) async throws -> String {
        let vmlxMessages = messages.map { $0.toVMLX() }
        var params = parameters.toSamplingParams()
        params.stop = stopSequences
        let vmlxTools = tools.map { $0.toVMLX() }
        let vmlxChoice = toolChoice?.toVMLXString()
        return try await service.respondWithTools(
            messages: vmlxMessages,
            params: params,
            stopSequences: stopSequences,
            tools: vmlxTools,
            toolChoice: vmlxChoice,
            requestedModel: requestedModel
        )
    }

    func streamWithTools(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        stopSequences: [String],
        tools: [Tool],
        toolChoice: ToolChoiceOption?,
        requestedModel: String?
    ) async throws -> AsyncThrowingStream<String, Error> {
        let vmlxMessages = messages.map { $0.toVMLX() }
        var params = parameters.toSamplingParams()
        params.stop = stopSequences
        let vmlxTools = tools.map { $0.toVMLX() }
        let vmlxChoice = toolChoice?.toVMLXString()
        return try await service.streamWithTools(
            messages: vmlxMessages,
            params: params,
            stopSequences: stopSequences,
            tools: vmlxTools,
            toolChoice: vmlxChoice,
            requestedModel: requestedModel
        )
    }

    // MARK: - Model Management Passthrough

    func loadModel(name: String) async throws {
        try await service.loadModel(name: name)
    }

    func unloadModel() async {
        await service.unloadModel()
    }

    var isModelLoaded: Bool {
        get async { await service.isModelLoaded }
    }

    // MARK: - Static Model Discovery

    /// Return available VMLX model names by scanning well-known directories.
    /// Called from ChatEngine's installedModelsProvider to merge with MLXService models.
    nonisolated static func getAvailableModels() -> [String] {
        ModelDetector.scanAvailableModels().map(\.name)
    }
}

// MARK: - ChatMessage → VMLXChatMessage

extension ChatMessage {
    /// Convert Osaurus ChatMessage to VMLXRuntime's VMLXChatMessage.
    func toVMLX() -> VMLXChatMessage {
        // Map content parts (multimodal)
        let vmlxParts: [VMLXContentPart]? = contentParts?.map { part in
            switch part {
            case .text(let text):
                return .text(text)
            case .imageUrl(let url, let detail):
                return .imageURL(url: url, detail: detail)
            }
        }

        // Map tool calls
        let vmlxToolCalls: [VMLXToolCall]? = tool_calls?.map { tc in
            VMLXToolCall(
                id: tc.id,
                name: tc.function.name,
                arguments: tc.function.arguments
            )
        }

        return VMLXChatMessage(
            role: role,
            content: content,
            contentParts: vmlxParts,
            toolCalls: vmlxToolCalls,
            toolCallId: tool_call_id
        )
    }
}

// MARK: - VMLXChatMessage → ChatMessage

extension VMLXChatMessage {
    /// Convert VMLXRuntime's VMLXChatMessage back to Osaurus ChatMessage.
    func toOsaurus() -> ChatMessage {
        // Map tool calls back
        let osToolCalls: [ToolCall]? = toolCalls?.map { tc in
            ToolCall(
                id: tc.id,
                type: tc.type,
                function: ToolCallFunction(
                    name: tc.function.name,
                    arguments: tc.function.arguments
                )
            )
        }

        return ChatMessage(
            role: role,
            content: content,
            tool_calls: osToolCalls,
            tool_call_id: toolCallId
        )
    }
}

// MARK: - GenerationParameters → SamplingParams

extension GenerationParameters {
    /// Convert Osaurus GenerationParameters to VMLXRuntime's SamplingParams.
    func toSamplingParams() -> SamplingParams {
        SamplingParams(
            maxTokens: maxTokens,
            temperature: temperature ?? 0.7,
            topP: topPOverride ?? 0.9,
            repetitionPenalty: repetitionPenalty ?? 1.0
        )
    }
}

// MARK: - Tool → VMLXToolDefinition

extension Tool {
    /// Convert Osaurus Tool to VMLXRuntime's VMLXToolDefinition.
    func toVMLX() -> VMLXToolDefinition {
        VMLXToolDefinition(
            name: function.name,
            description: function.description
        )
    }
}

// MARK: - ToolChoiceOption → String

extension ToolChoiceOption {
    /// Convert Osaurus ToolChoiceOption to VMLXRuntime's string-based tool choice.
    func toVMLXString() -> String {
        switch self {
        case .auto:
            return "auto"
        case .none:
            return "none"
        case .function(let fn):
            return fn.function.name
        }
    }
}
