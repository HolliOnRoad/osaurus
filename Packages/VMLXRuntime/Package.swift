// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "VMLXRuntime",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "VMLXRuntime", targets: ["VMLXRuntime"]),
    ],
    dependencies: [
        .package(url: "https://github.com/jjang-ai/mlx-swift.git", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "VMLXRuntime",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "VMLXSmokeTest",
            dependencies: ["VMLXRuntime"],
            path: "Sources/VMLXSmokeTest"
        ),
        .testTarget(
            name: "VMLXRuntimeTests",
            dependencies: ["VMLXRuntime"]
        ),
    ]
)
