//
//  PrefixHash.swift
//  osaurus
//
//  Computes a stable hash from system prompt + tool names for prefix cache keying.
//  Extracted from the deleted ModelRuntime.swift.
//

import CryptoKit
import Foundation

enum PrefixHash {
    /// Compute a stable hash key from system prompt content and tool names.
    /// Used for prefix cache identification across turns.
    static func compute(systemContent: String, toolNames: [String]) -> String {
        let tools = toolNames.sorted().joined(separator: "\0")
        let combined = systemContent + "\0" + tools
        let digest = SHA256.hash(data: Data(combined.utf8))
        return digest.prefix(16).map { String(format: "%02x", $0) }.joined()
    }
}
