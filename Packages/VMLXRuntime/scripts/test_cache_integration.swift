#!/usr/bin/env swift
/// Standalone cache integration test script.
/// Run from the VMLXRuntime package directory:
///   swift run --package-path . test_cache_integration
///
/// Or just test detection (no GPU needed):
///   swift test --filter "ModelLoadingTests"

import Foundation

// Since we can't run MLX tests via swift test (metallib issue),
// document what needs to be verified via the Osaurus app:
//
// MANUAL TEST PLAN (run in Osaurus app):
//
// Test 1: Model Detection
//   ✅ Verified via swift test: detectJangModel passes
//   - Detects Qwen3.5-4B-JANG_2S correctly
//   - isJang=true, modelType=qwen3_5, isHybrid=true
//
// Test 2: Forward Pass + Cache Types
//   Run in Osaurus: load Qwen3.5-4B-JANG_4S, send "Hello"
//   Check /tmp/vmlx_debug.log for:
//   - "[Gen] Cache MISS: prefilling N tokens" (first turn)
//   - "[Gen] Stored cache: N tokens"
//
// Test 3: Multi-Turn Cache Hit
//   Continue conversation: send "How are you?"
//   Check log for:
//   - "[Gen] Cache HIT: 32 layers, N remaining tokens"
//   - "[Gen] Injected N SSM companion states from checkpoint"
//   This confirms: gen_prompt_len stripping, cache store/fetch, SSM companion
//
// Test 4: TurboQuant
//   Enable TurboQuant toggle in Settings → Advanced
//   Start new conversation, send "Hello"
//   Check log for:
//   - "[Gen] Stored cache:" (confirms TQ compression on store)
//   Continue conversation
//   - "[Gen] Cache HIT:" (confirms TQ decompression on fetch)
//
// Test 5: KV Quantization
//   Set KV Cache Bits to 4 in Settings
//   Start new conversation
//   Verify generation works (quantized KV cache transparent to user)
//
// Test 6: Disk Cache
//   Enable Disk Cache toggle in Settings
//   Generate response, then quit and relaunch
//   Send same message → should hit disk cache
//   Check log: "[Gen] Cache HIT: ... detail=disk"

print("Cache integration tests require GPU (Metal).")
print("Run model detection tests: swift test --filter ModelLoadingTests")
print("Run full tests in Osaurus app — see manual test plan above.")
