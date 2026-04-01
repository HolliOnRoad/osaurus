//
//  ModelPickerItemCache.swift
//  osaurus
//
//  Global cache for model picker items shared across all views.
//

import Foundation

@MainActor
final class ModelPickerItemCache: ObservableObject {
    static let shared = ModelPickerItemCache()

    @Published private(set) var items: [ModelPickerItem] = []
    @Published private(set) var isLoaded = false

    private var observersRegistered = false

    private init() {
        registerObservers()
    }

    private func registerObservers() {
        guard !observersRegistered else { return }
        observersRegistered = true
        for name: Notification.Name in [.localModelsChanged, .remoteProviderModelsChanged] {
            NotificationCenter.default.addObserver(
                forName: name,
                object: nil,
                queue: .main
            ) { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.invalidateCache()
                    await self?.buildModelPickerItems()
                }
            }
        }
    }

    @discardableResult
    func buildModelPickerItems() async -> [ModelPickerItem] {
        var options: [ModelPickerItem] = []

        if AppConfiguration.shared.foundationModelAvailable {
            options.append(.foundation())
        }

        let localModels = await Task.detached(priority: .userInitiated) {
            ModelManager.discoverLocalModels()
        }.value

        for model in localModels {
            options.append(.fromMLXModel(model))
        }

        // Add VMLX-detected models not already in the list (JANG, well-known dirs, HF cache)
        // Normalize IDs for dedup: lowercase, strip org prefix, replace separators
        func normalizeForDedup(_ id: String) -> String {
            let base = id.split(separator: "/").last.map(String.init) ?? id
            return base.lowercased()
                .replacingOccurrences(of: "-", with: "")
                .replacingOccurrences(of: "_", with: "")
                .replacingOccurrences(of: " ", with: "")
        }
        let existingNormalized = Set(options.map { normalizeForDedup($0.id) })
        let vmlxModels = VMLXServiceBridge.getAvailableModels()
        for name in vmlxModels {
            if !existingNormalized.contains(normalizeForDedup(name)) {
                options.append(.localDetected(name: name))
            }
        }

        let remoteModels = RemoteProviderManager.shared.cachedAvailableModels()

        for providerInfo in remoteModels {
            for modelId in providerInfo.models {
                options.append(
                    .fromRemoteModel(
                        modelId: modelId,
                        providerName: providerInfo.providerName,
                        providerId: providerInfo.providerId
                    )
                )
            }
        }

        items = options
        isLoaded = true
        return options
    }

    func prewarmModelCache() async {
        await buildModelPickerItems()
    }

    func prewarmLocalModelsOnly() {
        Task {
            let localModels = await Task.detached(priority: .userInitiated) {
                ModelManager.discoverLocalModels()
            }.value

            var options: [ModelPickerItem] = []
            if AppConfiguration.shared.foundationModelAvailable {
                options.append(.foundation())
            }
            for model in localModels {
                options.append(.fromMLXModel(model))
            }

            // Add VMLX-detected models (JANG, well-known dirs, HF cache)
            func normalizeForDedup(_ id: String) -> String {
                let base = id.split(separator: "/").last.map(String.init) ?? id
                return base.lowercased()
                    .replacingOccurrences(of: "-", with: "")
                    .replacingOccurrences(of: "_", with: "")
                    .replacingOccurrences(of: " ", with: "")
            }
            let existingNormalized = Set(options.map { normalizeForDedup($0.id) })
            let vmlxModels = VMLXServiceBridge.getAvailableModels()
            for name in vmlxModels {
                if !existingNormalized.contains(normalizeForDedup(name)) {
                    options.append(.localDetected(name: name))
                }
            }

            items = options
            isLoaded = true
        }
    }

    func invalidateCache() {
        isLoaded = false
        items = []
    }
}
