//
//  VMLXEngineConfig.swift
//  osaurus
//
//  Maps ServerConfiguration fields to CLI arguments for vmlx-engine.
//  Reference: vmlx_engine/cli.py serve_parser arguments.
//

import Foundation

enum VMLXEngineConfig {

    /// Build the CLI argument array for launching a vmlx-engine instance.
    ///
    /// Usage: `python3 -m vmlx_engine.cli serve <model> --port <port> [flags]`
    ///
    /// - Parameters:
    ///   - model: HuggingFace model name or local path
    ///   - port: Port to bind the engine on
    ///   - config: Current server configuration
    /// - Returns: Array of CLI arguments (not including `python3 -m vmlx_engine.cli`)
    static func buildArgs(model: String, port: Int, config: ServerConfiguration, modelOptions: [String: ModelOptionValue]? = nil) -> [String] {
        var args = ["serve", model, "--port", String(port)]

        // Host: always localhost (Swift gateway handles external exposure)
        args += ["--host", "127.0.0.1"]

        // Max tokens
        args += ["--max-tokens", String(config.maxTokens)]

        // Continuous batching
        if config.continuousBatching {
            args.append("--continuous-batching")
            args += ["--max-num-seqs", String(config.maxNumSeqs)]
            args += ["--stream-interval", String(config.streamInterval)]

            // Prefix cache
            if config.enablePrefixCache {
                args.append("--enable-prefix-cache")
                args += ["--prefix-cache-size", String(config.prefixCacheSize)]
            } else {
                args.append("--disable-prefix-cache")
            }

            // Memory-aware cache
            if let mb = config.cacheMemoryMB {
                args += ["--cache-memory-mb", String(mb)]
            } else {
                args += ["--cache-memory-percent", String(config.cacheMemoryPercent)]
            }

            if config.cacheTTLMinutes > 0 {
                args += ["--cache-ttl-minutes", String(config.cacheTTLMinutes)]
            }

            // Paged cache
            if config.usePagedCache {
                args.append("--use-paged-cache")
                args += ["--paged-cache-block-size", String(config.pagedCacheBlockSize)]
                args += ["--max-cache-blocks", String(config.maxCacheBlocks)]

                // Block-level disk cache (L2 for paged)
                if config.enableBlockDiskCache {
                    args.append("--enable-block-disk-cache")
                    args += ["--block-disk-cache-max-gb", String(config.blockDiskCacheMaxGB)]
                }
            }

            // KV cache quantization
            if config.kvCacheQuantization != "none" {
                args += ["--kv-cache-quantization", config.kvCacheQuantization]
                args += ["--kv-cache-group-size", String(config.kvCacheGroupSize)]
            }

            // Disk cache
            if config.enableDiskCache {
                args.append("--enable-disk-cache")
                args += ["--disk-cache-max-gb", String(config.diskCacheMaxGB)]
            }
        }

        // Tool call parser — per-model option overrides global config
        let toolParser: String = {
            if let perModel = modelOptions?["toolParser"]?.stringValue, !perModel.isEmpty {
                return perModel
            }
            return config.toolCallParser
        }().trimmingCharacters(in: .whitespacesAndNewlines)
        if !toolParser.isEmpty && toolParser != "none" {
            args.append("--enable-auto-tool-choice")
            args += ["--tool-call-parser", toolParser]
        }

        // Reasoning parser — per-model option overrides global config
        let reasoningParser: String = {
            if let perModel = modelOptions?["reasoningParser"]?.stringValue, !perModel.isEmpty {
                return perModel
            }
            return config.reasoningParser
        }().trimmingCharacters(in: .whitespacesAndNewlines)
        if !reasoningParser.isEmpty && reasoningParser != "none" {
            args += ["--reasoning-parser", reasoningParser]
        }

        // JIT compilation
        if config.enableJIT {
            args.append("--enable-jit")
        }

        // Default thinking mode
        if let thinking = config.defaultEnableThinking, !thinking.isEmpty {
            args += ["--default-enable-thinking", thinking]
        }

        // Speculative decoding
        if let specModel = config.speculativeModel,
           !specModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args += ["--speculative-model", specModel]
            args += ["--num-draft-tokens", String(config.numDraftTokens)]
        }

        // Prompt Lookup Decoding
        if config.enablePLD {
            args.append("--enable-pld")
        }

        // Default generation params
        if let temp = config.defaultTemperature {
            args += ["--default-temperature", String(temp)]
        }
        // defaultTopP takes priority, fall back to genTopP if non-default
        if let topP = config.defaultTopP {
            args += ["--default-top-p", String(topP)]
        } else if config.genTopP != 1.0 {
            args += ["--default-top-p", String(config.genTopP)]
        }

        return args
    }
}
