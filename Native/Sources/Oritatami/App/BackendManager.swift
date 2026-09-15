import Foundation
import Combine

/// Manages the Python backend process lifecycle.
/// Starts `oritatami --serve` and polls the health endpoint until ready.
@MainActor
final class BackendManager: ObservableObject {
    @Published private(set) var isReady = false
    @Published private(set) var startupMessage = "バックエンドを起動中..."
    @Published private(set) var startupError: String? = nil

    static let port = 47823
    static let baseURL = URL(string: "http://127.0.0.1:\(port)")!

    private var process: Process?
    private var pollTask: Task<Void, Never>?

    init() {
        start()
    }

    private func start() {
        let proc = Process()
        // Try venv python first, then system python3
        let venvPath = findVenvPython()
        proc.executableURL = URL(fileURLWithPath: venvPath)
        proc.arguments = ["-m", "oritatami", "--serve", "--port", "\(Self.port)"]
        proc.environment = ProcessInfo.processInfo.environment

        // Suppress backend stdout/stderr (it logs to file)
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError  = FileHandle.nullDevice

        do {
            try proc.run()
        } catch {
            startupError = "プロセス起動失敗: \(error.localizedDescription)"
            return
        }
        self.process = proc

        // Poll health endpoint
        pollTask = Task {
            await pollHealth()
        }
    }

    private func pollHealth() async {
        let healthURL = Self.baseURL.appendingPathComponent("/api/health")
        var attempts = 0
        while !Task.isCancelled {
            do {
                let (_, resp) = try await URLSession.shared.data(from: healthURL)
                if let http = resp as? HTTPURLResponse, http.statusCode == 200 {
                    isReady = true
                    return
                }
            } catch {}
            attempts += 1
            if attempts > 60 {
                startupError = "バックエンドが60秒以内に起動しませんでした"
                return
            }
            startupMessage = "バックエンドを起動中 (\(attempts)s)..."
            try? await Task.sleep(nanoseconds: 1_000_000_000)
        }
    }

    /// Find the oritatami venv python, falling back to system python3.
    private func findVenvPython() -> String {
        // Common locations relative to the app bundle
        let candidates = [
            // Installed via pipx / user venv
            "\(NSHomeDirectory())/.local/bin/oritatami",
            "\(NSHomeDirectory())/.local/pipx/venvs/oritatami/bin/python",
            "\(NSHomeDirectory())/Library/Python/3.11/bin/oritatami",
            // Developer: venv next to the Native/ folder
            bundleRelative("../../.venv/bin/python"),
            bundleRelative("../../.venv/bin/python3"),
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]
        // If oritatami is directly on PATH use -m oritatami; otherwise use python -m
        for p in candidates {
            if FileManager.default.fileExists(atPath: p) { return p }
        }
        return "/usr/bin/python3"
    }

    private func bundleRelative(_ rel: String) -> String {
        let bundleDir = Bundle.main.bundlePath
        return (bundleDir as NSString).appendingPathComponent(rel)
    }

    deinit {
        pollTask?.cancel()
        process?.terminate()
    }
}
