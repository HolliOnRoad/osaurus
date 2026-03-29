import Testing
import Foundation
@testable import VMLXRuntime

@Suite("SchedulerConfig")
struct SchedulerConfigTests {

    @Test("Default values")
    func defaults() {
        let config = SchedulerConfig()
        #expect(config.maxNumSeqs == 256)
        #expect(config.prefillBatchSize == 8)
        #expect(config.completionBatchSize == 32)
        #expect(config.enablePrefixCache == true)
        #expect(config.useMemoryAwareCache == true)
        #expect(config.kvCacheQuantization == "none")
        #expect(config.enableTurboQuant == false)
        #expect(config.policy == .fcfs)
    }

    @Test("Auto-detect scales by RAM")
    func autoDetect() {
        let config = SchedulerConfig.autoDetect()
        // Should produce valid config regardless of machine
        #expect(config.prefillStepSize >= 1024)
        #expect(config.maxNumSeqs >= 8)
    }

    @Test("toCacheCoordinatorConfig maps correctly")
    func toCacheConfig() {
        var config = SchedulerConfig()
        config.usePagedCache = true
        config.pagedCacheBlockSize = 128
        config.maxCacheBlocks = 2000
        config.enableDiskCache = true

        let cacheConfig = config.toCacheCoordinatorConfig()
        #expect(cacheConfig.usePagedCache == true)
        #expect(cacheConfig.pagedBlockSize == 128)
        #expect(cacheConfig.maxCacheBlocks == 2000)
        #expect(cacheConfig.enableDiskCache == true)
    }

    @Test("Custom config")
    func custom() {
        let config = SchedulerConfig(
            maxNumSeqs: 64,
            prefillBatchSize: 4,
            enableTurboQuant: true
        )
        #expect(config.maxNumSeqs == 64)
        #expect(config.prefillBatchSize == 4)
        #expect(config.enableTurboQuant == true)
    }
}
