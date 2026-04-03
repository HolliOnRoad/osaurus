//
//  VMLXProcessManager.swift
//  osaurus
//
//  Manages vmlx-engine Python subprocesses. Each model instance
//  runs as a separate process on a unique port. Handles:
//  - Process spawning with bundled Python
//  - Health check polling
//  - Idle sleep / wake
//  - Process monitoring and auto-restart
//

import Foundation
import os.log

private let logger = Logger(subsystem: "ai.osaurus", category: "VMLXProcessManager")

/// Thread-safe last-line tracker for stderr crash diagnostics (Flag 4)
actor LastLine {
    private var line: String = ""
    func set(_ value: String) { line = value }
    func get() -> String { line }
}

actor VMLXProcessManager {
    static let shared = VMLXProcessManager()

    /// Running processes keyed by model name
    private var processes: [String: Process] = [:]
    /// Models currently being launched (prevents duplicate concurrent spawns)
    private var launching: Set<String> = []
    /// Idle timers keyed by model name
    private var idleTimers: [String: Task<Void, Never>] = [:]
    /// Last stderr line per model (for crash diagnostics)
    private var lastStderrLines: [String: LastLine] = [:]
    /// Crash restart counts for backoff (issue 2: prevent infinite restart loop)
    private var restartCounts: [String: Int] = [:]
    private static let maxRestarts = 3
    /// Monitor tasks keyed by model name
    private var monitors: [String: Task<Void, Never>] = [:]

    // MARK: - Launch

    /// Launch a vmlx-engine instance for the given model.
    /// Returns the port the engine is listening on.
    func launchEngine(
        model: String,
        modelPath: String,
        config: ServerConfiguration,
        modelOptions: [String: ModelOptionValue]? = nil
    ) async throws -> Int {
        // If already running, return existing port
        if let existing = await VMLXGateway.shared.port(for: model) {
            logger.info("Engine already running for \(model) on port \(existing)")
            return existing
        }

        // Prevent duplicate concurrent launches for the same model
        guard !launching.contains(model) else {
            logger.info("Engine launch already in progress for \(model), waiting...")
            // Poll until the other launch completes and registers with gateway
            for _ in 0..<60 {
                try await Task.sleep(for: .seconds(2))
                if let port = await VMLXGateway.shared.port(for: model) {
                    return port
                }
            }
            throw VMLXError.engineStartTimeout
        }
        launching.insert(model)
        defer { launching.remove(model) }

        let port = try findFreePort()
        let args = VMLXEngineConfig.buildArgs(model: modelPath, port: port, config: config, modelOptions: modelOptions)

        let pythonPath = Self.bundledPythonPath()
        logger.info("Launching vmlx-engine: \(pythonPath) -m vmlx_engine.cli \(args.joined(separator: " "))")

        // Check for orphaned engine on this port before launching
        await Self.killOrphanedEngine(port: port)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonPath)
        // Flag 1: -s suppresses user site-packages for full isolation
        process.arguments = ["-s", "-m", "vmlx_engine.cli"] + args

        // Flag 1: Process env isolation — only bundled Python, no user packages
        var env: [String: String] = [:]
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = ""  // No external paths
        // Set PYTHONHOME to the bundled Python root
        let bundleDir = (pythonPath as NSString).deletingLastPathComponent
        let libDir = (bundleDir as NSString)
            .deletingLastPathComponent  // bin -> python
        env["PYTHONHOME"] = libDir
        // Inherit minimal system env for dyld/Metal
        env["HOME"] = ProcessInfo.processInfo.environment["HOME"]
        env["PATH"] = "/usr/bin:/bin"
        env["DYLD_FRAMEWORK_PATH"] = ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"]
        env["TMPDIR"] = ProcessInfo.processInfo.environment["TMPDIR"]
        env["METAL_DEVICE_WRAPPER_TYPE"] = ProcessInfo.processInfo.environment["METAL_DEVICE_WRAPPER_TYPE"]
        // Pass HuggingFace token for gated models (Llama 3, Gemma, etc.)
        if let hfToken = ProcessInfo.processInfo.environment["HF_TOKEN"] ?? ProcessInfo.processInfo.environment["HUGGING_FACE_HUB_TOKEN"] {
            env["HF_TOKEN"] = hfToken
        }
        process.environment = env

        // Flag 4: Capture stdout/stderr — stderr last line is surfaced on crash
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        // Track last stderr line for crash diagnostics
        let lastStderrLine = LastLine()
        Task.detached { [model] in
            for try await line in stdoutPipe.fileHandleForReading.bytes.lines {
                logger.debug("[vmlx:\(model)] \(line)")
            }
        }
        Task.detached { [model] in
            for try await line in stderrPipe.fileHandleForReading.bytes.lines {
                logger.info("[vmlx:\(model):err] \(line)")
                await lastStderrLine.set(line)
            }
        }
        lastStderrLines[model] = lastStderrLine

        try process.run()
        processes[model] = process

        // Flag 3: 120s timeout with 2s intervals for large model loading
        try await waitForHealth(port: port, timeout: 120, process: process, model: model)

        // Register with gateway
        let instance = VMLXInstance(
            modelName: model,
            modelPath: modelPath,
            port: port,
            processIdentifier: process.processIdentifier,
            startedAt: Date()
        )
        await VMLXGateway.shared.register(instance)

        // Start process monitor
        startMonitor(for: model, process: process, modelPath: modelPath, config: config)

        logger.info("vmlx-engine ready for \(model) on port \(port)")
        return port
    }

    // MARK: - Stop

    /// Stop the engine for a model.
    func stopEngine(model: String) async {
        cancelIdleTimer(for: model)
        monitors[model]?.cancel()
        monitors.removeValue(forKey: model)

        if let process = processes.removeValue(forKey: model) {
            let pid = process.processIdentifier
            process.terminate()  // SIGTERM
            // Give 1.5s for graceful shutdown (flush disk caches, etc.)
            try? await Task.sleep(for: .milliseconds(1500))
            if process.isRunning {
                // SIGKILL — force kill if SIGTERM didn't work
                kill(pid, SIGKILL)
            }
        }
        restartCounts.removeValue(forKey: model)
        await VMLXGateway.shared.unregister(model: model)
        logger.info("Stopped engine for \(model)")
    }

    /// Stop all running engines.
    func stopAll() async {
        let models = Array(processes.keys)
        for model in models {
            await stopEngine(model: model)
        }
    }

    // MARK: - Model Swap

    /// Swap from one model to another.
    func swapModel(
        from: String,
        to: String,
        modelPath: String,
        config: ServerConfiguration
    ) async throws -> Int {
        await stopEngine(model: from)
        return try await launchEngine(model: to, modelPath: modelPath, config: config)
    }

    // MARK: - Idle Sleep

    /// Send soft sleep to an engine (clears GPU caches, model stays loaded).
    func softSleep(model: String) async {
        guard let port = await VMLXGateway.shared.port(for: model) else { return }
        let url = URL(string: "http://127.0.0.1:\(port)/admin/soft-sleep")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        _ = try? await URLSession.shared.data(for: request)
        logger.info("Soft sleep sent to \(model)")
    }

    /// Send deep sleep to an engine (unloads model from memory).
    func deepSleep(model: String) async {
        guard let port = await VMLXGateway.shared.port(for: model) else { return }
        let url = URL(string: "http://127.0.0.1:\(port)/admin/deep-sleep")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        _ = try? await URLSession.shared.data(for: request)
        logger.info("Deep sleep sent to \(model)")
    }

    /// Reset the idle timer for a model. Called after each request completes.
    /// Uses the new checkbox-based soft/deep sleep settings.
    func resetIdleTimer(for model: String, config: ServerConfiguration) {
        cancelIdleTimer(for: model)

        // Soft sleep timer (clear caches, model stays loaded)
        if config.enableSoftSleep && config.softSleepMinutes > 0 {
            let minutes = config.softSleepMinutes
            idleTimers[model + ".soft"] = Task { [weak self] in
                try? await Task.sleep(for: .seconds(minutes * 60))
                guard !Task.isCancelled else { return }
                guard let self else { return }
                await self.softSleep(model: model)
                logger.info("Idle soft sleep triggered for \(model) after \(minutes)m")
            }
        }

        // Deep sleep timer (unload model from memory)
        if config.enableDeepSleep && config.deepSleepMinutes > 0 {
            let minutes = config.deepSleepMinutes
            idleTimers[model + ".deep"] = Task { [weak self] in
                try? await Task.sleep(for: .seconds(minutes * 60))
                guard !Task.isCancelled else { return }
                guard let self else { return }
                await self.deepSleep(model: model)
                logger.info("Idle deep sleep triggered for \(model) after \(minutes)m")
            }
        }
    }

    private func cancelIdleTimer(for model: String) {
        idleTimers[model]?.cancel()
        idleTimers.removeValue(forKey: model)
        idleTimers[model + ".soft"]?.cancel()
        idleTimers.removeValue(forKey: model + ".soft")
        idleTimers[model + ".deep"]?.cancel()
        idleTimers.removeValue(forKey: model + ".deep")
    }

    // MARK: - Health Check

    // Flag 3: 120s timeout, 2s polling intervals for large models
    // Checks process liveness during polling — surfaces crash immediately instead of 120s timeout
    private func waitForHealth(port: Int, timeout: TimeInterval, process: Process, model: String) async throws {
        let url = URL(string: "http://127.0.0.1:\(port)/health")!
        let deadline = Date().addingTimeInterval(timeout)

        while Date() < deadline {
            // Check if process crashed before health endpoint is up
            if !process.isRunning {
                let errMsg = await lastStderrLines[model]?.get() ?? "unknown error"
                throw VMLXError.engineCrashed(model: model, stderr: errMsg)
            }
            do {
                let (_, response) = try await URLSession.shared.data(from: url)
                if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                    return
                }
            } catch {
                // Engine not ready yet, keep polling
            }
            try await Task.sleep(for: .seconds(2))
        }
        // Final process check before timeout
        if !process.isRunning {
            let errMsg = await lastStderrLines[model]?.get() ?? "unknown error"
            throw VMLXError.engineCrashed(model: model, stderr: errMsg)
        }
        throw VMLXError.engineStartTimeout
    }

    // Flag 2: Kill orphaned vmlx-engine processes on a port (from prior app crash)
    private static func killOrphanedEngine(port: Int) async {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        proc.arguments = ["-ti", ":\(port)"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        do {
            try proc.run()
            proc.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            if !output.isEmpty {
                for pidStr in output.components(separatedBy: "\n") {
                    if let pid = Int32(pidStr.trimmingCharacters(in: .whitespacesAndNewlines)), pid > 0 {
                        logger.warning("Killing orphaned process \(pid) on port \(port)")
                        kill(pid, SIGTERM)
                    }
                }
                // Give orphan time to die
                try? await Task.sleep(for: .seconds(1))
            }
        } catch {
            // lsof failed — no orphan
        }
    }

    // MARK: - Process Monitor

    private func startMonitor(
        for model: String,
        process: Process,
        modelPath: String,
        config: ServerConfiguration
    ) {
        monitors[model] = Task { [weak self] in
            process.waitUntilExit()
            guard !Task.isCancelled else { return }
            guard let self else { return }

            let status = process.terminationStatus
            // Flag 4: Surface last stderr line on crash
            if let lastLine = await self.lastStderrLines[model] {
                let errMsg = await lastLine.get()
                if !errMsg.isEmpty {
                    logger.error("vmlx-engine for \(model) crashed: \(errMsg)")
                }
            }
            logger.warning("vmlx-engine for \(model) exited with status \(status)")

            // Clean up
            await VMLXGateway.shared.unregister(model: model)
            await self.cleanupProcess(model: model)

            // Auto-restart with backoff — give up after maxRestarts
            if status != 0 {
                let count = (await self.restartCounts[model] ?? 0) + 1
                await self.setRestartCount(model: model, count: count)
                if count > Self.maxRestarts {
                    logger.error("Engine for \(model) crashed \(count) times — giving up (OOM or bad model?)")
                } else {
                    let delay = min(Double(count) * 2.0, 10.0)  // 2s, 4s, 6s backoff
                    logger.info("Auto-restarting engine for \(model) (attempt \(count)/\(Self.maxRestarts), delay \(delay)s)")
                    try? await Task.sleep(for: .seconds(delay))
                    guard !Task.isCancelled else { return }
                    do {
                        _ = try await self.launchEngine(
                            model: model,
                            modelPath: modelPath,
                            config: config
                        )
                        // Reset count on successful restart
                        await self.setRestartCount(model: model, count: 0)
                    } catch {
                        logger.error("Failed to restart engine for \(model): \(error)")
                    }
                }
            }
        }
    }

    private func cleanupProcess(model: String) {
        processes.removeValue(forKey: model)
        cancelIdleTimer(for: model)
    }

    private func setRestartCount(model: String, count: Int) {
        restartCounts[model] = count
    }

    // MARK: - Port Allocation

    private func findFreePort() throws -> Int {
        let socket = socket(AF_INET, SOCK_STREAM, 0)
        guard socket >= 0 else { throw VMLXError.portAllocationFailed }
        defer { close(socket) }

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_addr.s_addr = INADDR_LOOPBACK.bigEndian
        addr.sin_port = 0  // Let OS assign a free port

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                bind(socket, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else { throw VMLXError.portAllocationFailed }

        var boundAddr = sockaddr_in()
        var addrLen = socklen_t(MemoryLayout<sockaddr_in>.size)
        let nameResult = withUnsafeMutablePointer(to: &boundAddr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                getsockname(socket, sockPtr, &addrLen)
            }
        }
        guard nameResult == 0 else { throw VMLXError.portAllocationFailed }

        return Int(boundAddr.sin_port.bigEndian)
    }

    // MARK: - Python Path

    /// Path to the bundled Python binary.
    /// Search order: app bundle → project Resources → vmlx repo bundled python → system python
    static func bundledPythonPath() -> String {
        // 1. Packaged app: Resources/bundled-python/python/bin/python3
        if let resourcePath = Bundle.main.resourcePath {
            let bundled = (resourcePath as NSString)
                .appendingPathComponent("bundled-python/python/bin/python3")
            if FileManager.default.fileExists(atPath: bundled) {
                return bundled
            }
        }
        // 2. Dev mode: project root Resources/bundled-python/
        let devPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Inference/
            .deletingLastPathComponent()  // Services/
            .deletingLastPathComponent()  // OsaurusCore/
            .deletingLastPathComponent()  // Packages/
            .appendingPathComponent("Resources/bundled-python/python/bin/python3")
            .path
        if FileManager.default.fileExists(atPath: devPath) {
            return devPath
        }
        // 3. Dev mode: reuse vmlx Electron app's bundled Python (has all deps)
        let vmlxBundled = NSString(string: NSHomeDirectory())
            .appendingPathComponent("mlx/vllm-mlx/panel/bundled-python/python/bin/python3")
        if FileManager.default.fileExists(atPath: vmlxBundled) {
            return vmlxBundled
        }
        // 4. Last resort: system Python (likely won't have vmlx_engine)
        return "/usr/bin/python3"
    }
}

// MARK: - Errors

enum VMLXError: Error, LocalizedError {
    case engineStartTimeout
    case engineCrashed(model: String, stderr: String)
    case portAllocationFailed
    case engineNotRunning(model: String)
    case noModelLoaded

    var errorDescription: String? {
        switch self {
        case .engineStartTimeout:
            return "vmlx engine failed to start within the timeout period"
        case .engineCrashed(let model, let stderr):
            return "vmlx engine for \(model) crashed during startup: \(stderr)"
        case .portAllocationFailed:
            return "Failed to allocate a free port for the engine"
        case .engineNotRunning(let model):
            return "No running engine for model: \(model)"
        case .noModelLoaded:
            return "No model is currently loaded"
        }
    }
}
