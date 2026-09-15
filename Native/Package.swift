// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Oritatami",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "Oritatami", targets: ["Oritatami"]),
    ],
    targets: [
        .executableTarget(
            name: "Oritatami",
            path: "Sources/Oritatami"
        ),
    ]
)
