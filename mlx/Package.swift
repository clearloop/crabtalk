// swift-tools-version: 5.9
import PackageDescription

// CrabllmMlx — Swift static library that sits behind the crabllm_mlx.h
// C ABI. Phase 5 pulls in `mlx-swift-lm` 2.31.3 for real model loading
// and generation. The target is pure Swift; C-compatible symbols are
// emitted via `@_cdecl`. The canonical header lives at
// `mlx/include/crabllm_mlx.h` and is consumed by `crates/mlx/build.rs`
// directly.
//
// Pinned to `2.31.3` because upstream main is a breaking 3.x. Revisit
// after we have coverage for the current API surface.
let package = Package(
    name: "CrabllmMlx",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "CrabllmMlx",
            type: .static,
            targets: ["CrabllmMlx"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            exact: "2.31.3"
        ),
    ],
    targets: [
        .target(
            name: "CrabllmMlx",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/CrabllmMlx",
            swiftSettings: [
                .unsafeFlags(["-warnings-as-errors"], .when(configuration: .release)),
            ]
        ),
    ]
)
