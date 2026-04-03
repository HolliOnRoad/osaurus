//
//  VMLXService.swift
//  osaurus
//
//  ToolCapableService implementation that bridges to the vmlx Python engine.
//  Replaces the old MLXService. Sends HTTP requests to the engine's
//  OpenAI-compatible API and parses SSE streaming responses.
//

import Foundation
import os.log

private let logger = Logger(subsystem: "ai.osaurus", category: "VMLXService")

actor VMLXService: ToolCapableService {

    static let shared = VMLXService()

    nonisolated var id: String { "vmlx" }

    // MARK: - Availability / Routing

    nonisolated func isAvailable() -> Bool {
        return !Self.getAvailableModels().isEmpty
    }

    nonisolated func handles(requestedModel: String?) -> Bool {
        let trimmed = (requestedModel ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        // Check if we have an installed model matching this name
        return ModelManager.findInstalledModel(named: trimmed) != nil
    }

    // MARK: - Static Discovery

    nonisolated static func getAvailableModels() -> [String] {
        return ModelManager.installedModelNames()
    }

    // MARK: - ModelService

    func streamDeltas(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        requestedModel: String?,
        stopSequences: [String]
    ) async throws -> AsyncThrowingStream<String, Error> {
        return try await streamWithTools(
            messages: messages,
            parameters: parameters,
            stopSequences: stopSequences,
            tools: [],
            toolChoice: nil,
            requestedModel: requestedModel
        )
    }

    func generateOneShot(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        requestedModel: String?
    ) async throws -> String {
        let stream = try await streamDeltas(
            messages: messages,
            parameters: parameters,
            requestedModel: requestedModel,
            stopSequences: []
        )
        var output = ""
        for try await delta in stream {
            if !StreamingToolHint.isSentinel(delta) {
                output += delta
            }
        }
        return output
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
        let stream = try await streamWithTools(
            messages: messages,
            parameters: parameters,
            stopSequences: stopSequences,
            tools: tools,
            toolChoice: toolChoice,
            requestedModel: requestedModel
        )
        var output = ""
        for try await delta in stream {
            if !StreamingToolHint.isSentinel(delta) {
                output += delta
            }
        }
        return output
    }

    func streamWithTools(
        messages: [ChatMessage],
        parameters: GenerationParameters,
        stopSequences: [String],
        tools: [Tool],
        toolChoice: ToolChoiceOption?,
        requestedModel: String?
    ) async throws -> AsyncThrowingStream<String, Error> {
        // Check if thinking display is enabled (controls UI display, not model generation)
        let showThinking: Bool = {
            if let val = parameters.modelOptions["disableThinking"]?.boolValue {
                return !val  // disableThinking=false means show thinking
            }
            return false  // Default: don't show thinking bubbles
        }()

        // Resolve model and port
        let resolved = try resolveModel(requestedModel)
        let modelName = resolved.name
        let port = try await ensureEngineRunning(for: requestedModel ?? modelName)

        // Build the HTTP request
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try buildRequestBody(
            messages: messages,
            model: modelName,
            parameters: parameters,
            stopSequences: stopSequences,
            tools: tools,
            toolChoice: toolChoice,
            stream: true
        )

        // Reset idle timer after stream completes (not before — prevents sleep during generation)
        let config = await MainActor.run { ServerConfigurationStore.load() ?? .default }

        let (bytes, response) = try await URLSession.shared.bytes(for: request)

        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw VMLXError.engineNotRunning(model: "HTTP \(statusCode) from engine")
        }

        return AsyncThrowingStream { continuation in
            let streamTask = Task {
                // Signal generation started for stats tracking
                InferenceProgressManager.shared.generationDidStartAsync()

                var hasEmittedThinkOpen = false
                do {
                    var accumulatedToolCalls: [AccumulatedToolCall] = []

                    // Batching: accumulate content deltas and yield in batches
                    // to reduce async hops to the MainActor consumer.
                    // At 80+ tok/s, per-token yields cause ~80 async suspensions/sec
                    // each triggering MainActor work. Batching to ~20 yields/sec is smoother.
                    var contentBatch = ""
                    var reasoningBatch = ""
                    var lastYieldTime = CFAbsoluteTimeGetCurrent()
                    let minYieldIntervalSec: Double = 0.03  // ~33 yields/sec max

                    for try await line in bytes.lines {
                        guard !line.isEmpty else { continue }

                        guard let chunk = VMLXSSEParser.parse(line: line) else { continue }
                        if chunk.isDone { break }

                        // Accumulate reasoning
                        if let reasoning = chunk.reasoningContent, !reasoning.isEmpty, showThinking {
                            reasoningBatch += reasoning
                        }
                        // Accumulate content
                        if let content = chunk.content, !content.isEmpty {
                            contentBatch += content
                        }

                        // Update inference stats (lightweight, no UI)
                        if let usage = chunk.usage {
                            InferenceProgressManager.shared.updateStatsAsync(
                                prompt: usage.promptTokens,
                                completion: usage.completionTokens,
                                cached: usage.cachedTokens,
                                detail: usage.cacheDetail
                            )
                        }

                        // Accumulate tool call deltas (incremental arguments)
                        if let toolDeltas = chunk.toolCalls {
                            for delta in toolDeltas {
                                if delta.index < accumulatedToolCalls.count {
                                    accumulatedToolCalls[delta.index].arguments += delta.arguments
                                } else {
                                    accumulatedToolCalls.append(
                                        AccumulatedToolCall(
                                            id: delta.id,
                                            name: delta.functionName,
                                            arguments: delta.arguments
                                        )
                                    )
                                }
                            }
                        }

                        // Yield batched content at controlled intervals
                        let now = CFAbsoluteTimeGetCurrent()
                        let shouldFlush = (now - lastYieldTime) >= minYieldIntervalSec
                            || chunk.finishReason != nil  // Always flush on finish

                        if shouldFlush && (!reasoningBatch.isEmpty || !contentBatch.isEmpty) {
                            if !reasoningBatch.isEmpty {
                                if !hasEmittedThinkOpen {
                                    continuation.yield("<think>")
                                    hasEmittedThinkOpen = true
                                }
                                continuation.yield(reasoningBatch)
                                reasoningBatch = ""
                            }
                            if !contentBatch.isEmpty {
                                if hasEmittedThinkOpen {
                                    continuation.yield("</think>")
                                    hasEmittedThinkOpen = false
                                }
                                continuation.yield(contentBatch)
                                contentBatch = ""
                            }
                            lastYieldTime = now
                        }

                        // When finish_reason is "tool_calls", flush remaining then emit tools
                        if chunk.finishReason == "tool_calls" && !accumulatedToolCalls.isEmpty {
                            for tc in accumulatedToolCalls {
                                continuation.yield(StreamingToolHint.encode(tc.name))
                                continuation.yield(StreamingToolHint.encodeArgs(tc.arguments))
                            }
                            let first = accumulatedToolCalls[0]
                            continuation.finish(
                                throwing: ServiceToolInvocation(
                                    toolName: first.name,
                                    jsonArguments: first.arguments,
                                    toolCallId: first.id
                                )
                            )
                            return
                        }
                    }
                    // Flush any remaining batched content
                    if !reasoningBatch.isEmpty {
                        if !hasEmittedThinkOpen { continuation.yield("<think>"); hasEmittedThinkOpen = true }
                        continuation.yield(reasoningBatch)
                    }
                    if !contentBatch.isEmpty {
                        if hasEmittedThinkOpen { continuation.yield("</think>"); hasEmittedThinkOpen = false }
                        continuation.yield(contentBatch)
                    }
                    // Close unclosed think tag
                    if hasEmittedThinkOpen {
                        continuation.yield("</think>")
                    }
                    // Reset idle timer now that generation is done
                    await VMLXProcessManager.shared.resetIdleTimer(for: modelName, config: config)
                    InferenceProgressManager.shared.generationDidFinishAsync()
                    continuation.finish()
                } catch {
                    if hasEmittedThinkOpen {
                        continuation.yield("</think>")
                    }
                    // Reset idle timer even on error (engine is still running)
                    await VMLXProcessManager.shared.resetIdleTimer(for: modelName, config: config)
                    InferenceProgressManager.shared.generationDidFinishAsync()
                    continuation.finish(throwing: error)
                }
            }
            // When the consumer cancels the stream (user presses stop),
            // cancel the inner task which closes the HTTP connection.
            // The engine detects client disconnect and aborts generation.
            continuation.onTermination = { @Sendable _ in
                streamTask.cancel()
            }
        }
    }

    /// Accumulated tool call data across incremental SSE chunks.
    private struct AccumulatedToolCall {
        let id: String
        let name: String
        var arguments: String
    }

    // MARK: - Request Body Builder

    private func buildRequestBody(
        messages: [ChatMessage],
        model: String,
        parameters: GenerationParameters,
        stopSequences: [String],
        tools: [Tool],
        toolChoice: ToolChoiceOption?,
        stream: Bool
    ) throws -> Data {
        var body: [String: Any] = [
            "model": model,
            "stream": stream,
            "messages": messages.map { msg -> [String: Any] in
                var m: [String: Any] = ["role": msg.role]

                // Multimodal: if contentParts has images, send as array
                if let parts = msg.contentParts,
                    parts.contains(where: {
                        if case .imageUrl = $0 { return true }; return false
                    })
                {
                    m["content"] = parts.map { part -> [String: Any] in
                        switch part {
                        case .text(let text):
                            return ["type": "text", "text": text]
                        case .imageUrl(let url, let detail):
                            var imgDict: [String: Any] = ["url": url]
                            if let d = detail { imgDict["detail"] = d }
                            return ["type": "image_url", "image_url": imgDict]
                        }
                    }
                } else if var content = msg.content {
                    // Strip <think>...</think> blocks from prior assistant messages
                    if msg.role == "assistant" {
                        content = Self.stripThinkingBlocks(content)
                    }
                    m["content"] = content
                }

                if let toolCallId = msg.tool_call_id {
                    m["tool_call_id"] = toolCallId
                }
                if let toolCalls = msg.tool_calls {
                    let encoder = JSONEncoder()
                    if let tcData = try? encoder.encode(toolCalls),
                        let tcArray = try? JSONSerialization.jsonObject(with: tcData)
                    {
                        m["tool_calls"] = tcArray
                    }
                }
                return m
            },
            "max_tokens": parameters.maxTokens,
        ]

        if let temp = parameters.temperature {
            body["temperature"] = temp
        }
        if let topP = parameters.topPOverride {
            body["top_p"] = topP
        }
        if let rep = parameters.repetitionPenalty {
            body["repetition_penalty"] = rep
        }
        if !stopSequences.isEmpty {
            body["stop"] = stopSequences
        }

        // Tools — encode via JSONEncoder to handle JSONValue properly
        if !tools.isEmpty {
            let encoder = JSONEncoder()
            body["tools"] = try tools.map { tool -> [String: Any] in
                let toolData = try encoder.encode(tool)
                guard let toolDict = try JSONSerialization.jsonObject(with: toolData) as? [String: Any] else {
                    return ["type": "function", "function": ["name": tool.function.name] as [String: Any]]
                }
                return toolDict
            }
        }

        // Tool choice
        if let choice = toolChoice {
            switch choice {
            case .auto:
                body["tool_choice"] = "auto"
            case .none:
                body["tool_choice"] = "none"
            case .function(let fn):
                body["tool_choice"] = ["type": "function", "function": ["name": fn.function.name]]
            }
        }

        // Session ID for prefix cache reuse across turns
        if let sessionId = parameters.sessionId {
            body["session_id"] = sessionId
        }
        if let cacheHint = parameters.cacheHint {
            body["cache_hint"] = cacheHint
        }

        // Never send enable_thinking — let the engine auto-detect per model.
        // The thinking toggle controls UI DISPLAY only (showThinking flag above).
        // Sending enable_thinking:true breaks Gemma 4 2-bit (all output goes to thinking channel).
        // Not sending it lets the engine's reasoning parser cleanly separate thinking from content.

        // Stream options for usage reporting
        if stream {
            body["stream_options"] = ["include_usage": true]
        }

        return try JSONSerialization.data(withJSONObject: body)
    }

    // MARK: - Model Resolution

    /// Resolve the requested model to a (displayName, modelPath) pair.
    private func resolveModel(_ requestedModel: String?) throws -> (name: String, path: String) {
        let trimmed = (requestedModel ?? "").trimmingCharacters(in: .whitespacesAndNewlines)

        if !trimmed.isEmpty, let found = ModelManager.findInstalledModel(named: trimmed) {
            return (name: found.name, path: found.id)
        }

        // Fall back to first installed model
        let models = ModelManager.discoverLocalModels()
        guard let first = models.first else {
            throw VMLXError.noModelLoaded
        }
        let name = URL(fileURLWithPath: first.id).lastPathComponent.lowercased()
        return (name, first.id)
    }

    /// Ensure an engine is running for the model, launching if needed.
    /// `requestedModel` is the raw model identifier from the chat view (may be filesystem path or display name).
    private func ensureEngineRunning(for requestedModel: String) async throws -> Int {
        let pyReady = await MainActor.run { PythonEnvironmentManager.shared.isReady }
        if !pyReady {
            throw PythonEnvError.pythonNotProvisioned
        }
        let resolved = try resolveModel(requestedModel)

        // Check if already running (match by path or name)
        if let port = await VMLXGateway.shared.port(for: resolved.name) {
            return port
        }
        if let port = await VMLXGateway.shared.port(for: resolved.path) {
            return port
        }

        let config = await MainActor.run { ServerConfigurationStore.load() ?? .default }

        // Load per-model options using the SAME key the UI uses.
        // The chat view saves under model.id (filesystem path for local models).
        // Try the requested model ID first (which is the picker item's id = model.id),
        // then fall back to the resolved path, then the display name.
        let modelOptions: [String: ModelOptionValue]? = await MainActor.run {
            ModelOptionsStore.shared.loadOptions(for: requestedModel)
                ?? ModelOptionsStore.shared.loadOptions(for: resolved.path)
                ?? ModelOptionsStore.shared.loadOptions(for: resolved.name)
        }

        // Check eviction policy — if strict single model, stop others first
        if config.modelEvictionPolicy == .strictSingleModel {
            await VMLXProcessManager.shared.stopAll()
        }

        return try await VMLXProcessManager.shared.launchEngine(
            model: resolved.name,
            modelPath: resolved.path,
            config: config,
            modelOptions: modelOptions
        )
    }

    // MARK: - Think Block Stripping

    /// Remove <think>...</think> blocks from text to prevent history contamination.
    /// Handles multiline blocks and multiple occurrences.
    private static func stripThinkingBlocks(_ text: String) -> String {
        // Remove complete <think>...</think> blocks (including multiline)
        var result = text
        while let startRange = result.range(of: "<think>"),
            let endRange = result.range(of: "</think>", range: startRange.upperBound ..< result.endIndex)
        {
            result.removeSubrange(startRange.lowerBound ..< endRange.upperBound)
        }
        // Also remove [THINK]...[/THINK] (Mistral format)
        while let startRange = result.range(of: "[THINK]"),
            let endRange = result.range(of: "[/THINK]", range: startRange.upperBound ..< result.endIndex)
        {
            result.removeSubrange(startRange.lowerBound ..< endRange.upperBound)
        }
        // Trim leading whitespace left by removed blocks
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

}
