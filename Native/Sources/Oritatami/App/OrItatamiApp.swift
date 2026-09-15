import SwiftUI

@main
struct OrItatamiApp: App {
    @StateObject private var backend = BackendManager()
    @StateObject private var api = APIClient()

    var body: some Scene {
        WindowGroup {
            Group {
                if backend.isReady {
                    ContentView()
                        .environmentObject(api)
                        .environmentObject(backend)
                } else {
                    LaunchScreen(backend: backend)
                }
            }
        }
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}

// MARK: - Launch screen shown while Python backend starts up

struct LaunchScreen: View {
    @ObservedObject var backend: BackendManager

    var body: some View {
        ZStack {
            Color(hex: "#0d1117").ignoresSafeArea()

            VStack(spacing: 24) {
                Image(systemName: "atom")
                    .font(.system(size: 64))
                    .foregroundColor(Color(hex: "#58a6ff"))

                Text("Oritatami")
                    .font(.system(size: 32, weight: .bold, design: .rounded))
                    .foregroundColor(.white)

                Text("タンパク質構造予測プラットフォーム")
                    .font(.system(size: 14))
                    .foregroundColor(Color(hex: "#8b949e"))

                if let err = backend.startupError {
                    VStack(spacing: 8) {
                        Text("起動エラー")
                            .font(.headline)
                            .foregroundColor(Color(hex: "#f85149"))
                        Text(err)
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(Color(hex: "#8b949e"))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                    }
                } else {
                    VStack(spacing: 8) {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .tint(Color(hex: "#58a6ff"))
                        Text(backend.startupMessage)
                            .font(.system(size: 12))
                            .foregroundColor(Color(hex: "#8b949e"))
                    }
                }
            }
        }
        .frame(width: 480, height: 360)
    }
}

// MARK: - Hex color helper

extension Color {
    init(hex: String) {
        let h = hex.trimmingCharacters(in: .init(charactersIn: "#"))
        var rgb: UInt64 = 0
        Scanner(string: h).scanHexInt64(&rgb)
        let r = Double((rgb >> 16) & 0xff) / 255
        let g = Double((rgb >>  8) & 0xff) / 255
        let b = Double((rgb >>  0) & 0xff) / 255
        self.init(red: r, green: g, blue: b)
    }
}
