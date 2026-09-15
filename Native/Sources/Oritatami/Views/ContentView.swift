import SwiftUI

/// Main window — 3-pane HSplitView: Workbench | Viewer | Assistant
struct ContentView: View {
    @EnvironmentObject var api: APIClient

    @State private var components: [ComponentInput] = []
    @State private var jobs: [Job] = []
    @State private var selectedJobId: String?
    @State private var structureURL: URL?      // PDB URL for Mol* viewer
    @State private var hoveredResidue: String?
    @State private var selectedResidue: ResidueInfo?

    // Viewer controls
    @State private var viewerStyle: String = "cartoon"
    @State private var viewerColorMode: String = "chain"
    @State private var isSpin = false

    var body: some View {
        HSplitView {
            // ─── Left: Workbench ───────────────────────────────────────
            WorkbenchView(
                components: $components,
                selectedJobId: $selectedJobId,
                structureURL: $structureURL,
                jobs: $jobs
            )
            .frame(minWidth: 280, idealWidth: 320, maxWidth: 420)

            // ─── Center: 3D Viewer ─────────────────────────────────────
            VStack(spacing: 0) {
                ViewerToolbar(
                    style: $viewerStyle,
                    colorMode: $viewerColorMode,
                    isSpin: $isSpin,
                    hasStructure: structureURL != nil
                )

                ZStack(alignment: .bottomLeading) {
                    ViewerView(
                        cifURL: $structureURL,
                        style: viewerStyle,
                        colorMode: viewerColorMode,
                        isSpin: isSpin,
                        onHover: { hoveredResidue = $0 },
                        onSelect: { selectedResidue = $0 }
                    )

                    if let label = hoveredResidue {
                        Text(label)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.white)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.black.opacity(0.75))
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                            .padding(12)
                    }
                }
            }
            .frame(minWidth: 400)

            // ─── Right: Assistant ──────────────────────────────────────
            AssistantView(
                components: $components,
                jobs: $jobs,
                selectedJobId: $selectedJobId,
                structureURL: $structureURL
            )
            .frame(minWidth: 280, idealWidth: 340, maxWidth: 420)
        }
        .background(Color(hex: "#0d1117"))
        .frame(minWidth: 1100, minHeight: 720)
        .onAppear { loadJobs() }
    }

    private func loadJobs() {
        Task {
            if let fetched = try? await api.listJobs() {
                jobs = fetched
            }
        }
    }
}

// MARK: - Viewer Toolbar

struct ViewerToolbar: View {
    @Binding var style: String
    @Binding var colorMode: String
    @Binding var isSpin: Bool
    var hasStructure: Bool

    private let styles = [("cartoon", "Cartoon"), ("ball+stick", "Ball+Stick"), ("surface", "Surface")]
    private let colors = [("chain", "鎖"), ("residue", "残基"), ("bfactor", "Bファクタ")]

    var body: some View {
        HStack(spacing: 12) {
            Picker("表示スタイル", selection: $style) {
                ForEach(styles, id: \.0) { Text($1).tag($0) }
            }
            .pickerStyle(.segmented)
            .frame(width: 220)
            .disabled(!hasStructure)

            Divider().frame(height: 18)

            Picker("色分け", selection: $colorMode) {
                ForEach(colors, id: \.0) { Text($1).tag($0) }
            }
            .pickerStyle(.segmented)
            .frame(width: 160)
            .disabled(!hasStructure)

            Divider().frame(height: 18)

            Toggle(isOn: $isSpin) {
                Label("回転", systemImage: "arrow.clockwise.circle")
                    .font(.system(size: 12))
            }
            .toggleStyle(.button)
            .disabled(!hasStructure)

            Spacer()

            if hasStructure {
                HStack(spacing: 4) {
                    Circle().fill(Color(hex: "#3fb950")).frame(width: 8, height: 8)
                    Text("構造ロード済み")
                        .font(.system(size: 11))
                        .foregroundColor(Color(hex: "#8b949e"))
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color(hex: "#161b22"))
        .overlay(
            Rectangle().frame(height: 1).foregroundColor(Color(hex: "#30363d")),
            alignment: .bottom
        )
    }
}
