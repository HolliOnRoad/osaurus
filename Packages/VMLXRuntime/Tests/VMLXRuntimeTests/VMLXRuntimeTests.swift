import Testing
@testable import VMLXRuntime

@Suite("VMLXRuntime")
struct VMLXRuntimeTests {
    @Test("Version is set")
    func versionIsSet() {
        #expect(VMLXRuntimeVersion.version == "0.1.0")
    }
}
