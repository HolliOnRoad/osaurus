// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "VMLXRuntime",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "VMLXRuntime", targets: ["VMLXRuntime"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.29.1"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.1.6"),
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
        .testTarget(
            name: "VMLXRuntimeTests",
            dependencies: ["VMLXRuntime"]
        ),
    ]
)
