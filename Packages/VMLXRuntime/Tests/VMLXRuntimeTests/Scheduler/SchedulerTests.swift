import Testing
import Foundation
@testable import VMLXRuntime

@Suite("Scheduler")
struct SchedulerTests {

    @Test("Add and schedule request")
    func addAndSchedule() {
        let scheduler = Scheduler(config: SchedulerConfig(maxNumSeqs: 10))
        let req = InferenceRequest(requestId: "r1", promptTokenIds: [1, 2, 3])
        scheduler.addRequest(req)

        let output = scheduler.schedule()
        #expect(output.scheduledRequestIds == ["r1"])
        #expect(output.hasWork)
        #expect(scheduler.runningCount == 1)
    }

    @Test("Cache lookup sets remaining tokens")
    func cacheLookup() {
        let scheduler = Scheduler()
        let req = InferenceRequest(requestId: "r1", promptTokenIds: [1, 2, 3, 4, 5])
        scheduler.addRequest(req)
        _ = scheduler.schedule()

        // No cache hit, so remainingTokenIds should be all prompt tokens
        let fetched = scheduler.getRequest("r1")
        #expect(fetched?.remainingTokenIds == [1, 2, 3, 4, 5])
        #expect(fetched?.cachedTokens == 0)
    }

    @Test("Record output appends tokens")
    func recordOutput() {
        let scheduler = Scheduler()
        scheduler.addRequest(InferenceRequest(requestId: "r1", promptTokenIds: [1]))
        _ = scheduler.schedule()

        scheduler.recordOutput(requestId: "r1", tokenId: 100, text: "hello")
        scheduler.recordOutput(requestId: "r1", tokenId: 101, text: " world")

        let req = scheduler.getRequest("r1")
        #expect(req?.outputTokenIds == [100, 101])
        #expect(req?.outputText == "hello world")
    }

    @Test("Finish request updates stats")
    func finishRequest() {
        let scheduler = Scheduler()
        scheduler.addRequest(InferenceRequest(requestId: "r1", promptTokenIds: [1]))
        _ = scheduler.schedule()
        scheduler.finishRequest("r1", reason: .stop)

        #expect(scheduler.totalRequestsProcessed == 1)
        #expect(scheduler.runningCount == 0)
    }

    @Test("Stop token detection")
    func stopTokens() {
        let scheduler = Scheduler()
        scheduler.configureForModel(isHybrid: false, stopTokenIds: [151643, 151645])

        #expect(scheduler.isStopToken(151643))
        #expect(scheduler.isStopToken(151645))
        #expect(!scheduler.isStopToken(999))
    }

    @Test("Hybrid model configuration")
    func hybridConfig() {
        let scheduler = Scheduler()
        scheduler.configureForModel(
            isHybrid: true,
            layerPattern: [.ssm, .ssm, .ssm, .attention],
            enableTQ: true
        )
        #expect(scheduler.isHybrid)
        #expect(scheduler.isTQActive)
        #expect(scheduler.layerPattern?.count == 4)
    }

    @Test("Abort removes request")
    func abort() {
        let scheduler = Scheduler()
        scheduler.addRequest(InferenceRequest(requestId: "r1", promptTokenIds: [1]))
        _ = scheduler.schedule()
        scheduler.abortRequest("r1")
        #expect(scheduler.runningCount == 0)
    }

    @Test("Multiple requests batched")
    func multipleBatched() {
        let scheduler = Scheduler(config: SchedulerConfig(maxNumSeqs: 5))
        for i in 0..<5 {
            scheduler.addRequest(InferenceRequest(requestId: "r\(i)", promptTokenIds: [i]))
        }
        let output = scheduler.schedule()
        #expect(output.scheduledRequestIds.count == 5)
    }

    @Test("Schedule respects max sequences")
    func respectsMax() {
        let scheduler = Scheduler(config: SchedulerConfig(maxNumSeqs: 2))
        for i in 0..<5 {
            scheduler.addRequest(InferenceRequest(requestId: "r\(i)", promptTokenIds: [i]))
        }
        let output = scheduler.schedule()
        #expect(output.scheduledRequestIds.count == 2)
        #expect(scheduler.waitingCount == 3)
    }

    @Test("Shutdown aborts all")
    func shutdown() {
        let scheduler = Scheduler()
        scheduler.addRequest(InferenceRequest(requestId: "r1", promptTokenIds: [1]))
        scheduler.addRequest(InferenceRequest(requestId: "r2", promptTokenIds: [2]))
        _ = scheduler.schedule()
        scheduler.shutdown()
        #expect(scheduler.runningCount == 0)
    }

    @Test("HasWork reflects pending state")
    func hasWork() {
        let scheduler = Scheduler()
        let o1 = scheduler.schedule()
        #expect(!o1.hasWork)  // Nothing pending

        scheduler.addRequest(InferenceRequest(requestId: "r1", promptTokenIds: [1]))
        let o2 = scheduler.schedule()
        #expect(o2.hasWork)  // Running request
    }
}
