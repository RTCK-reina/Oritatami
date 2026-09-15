import SwiftUI

/// Left panel — component editor + job submission.
struct WorkbenchView: View {
    @EnvironmentObject var api: APIClient
    @Binding var components: [ComponentInput]
    @Binding var selectedJobId: String?
    @Binding var structureURL: URL?
    @Binding var jobs: [Job]

    @State private var isSubmitting = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Label("ワークベンチ", systemImage: "wrench.and.screwdriver")
                    .font(.headline)
                    .foregroundColor(.white)
                Spacer()
                Button(action: addProtein) {
                    Image(systemName: "plus")
                        .foregroundColor(Color(hex: "#58a6ff"))
                }
                .buttonStyle(.plain)
                .help("コンポーネントを追加")
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color(hex: "#161b22"))

            Divider().overlay(Color(hex: "#30363d"))

            // Component list
            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach($components) { $comp in
                        ComponentRow(component: $comp, onDelete: { remove(comp) })
                    }
                }
                .padding(12)
            }

            Divider().overlay(Color(hex: "#30363d"))

            // Error banner
            if let err = errorMessage {
                Text(err)
                    .font(.system(size: 12))
                    .foregroundColor(Color(hex: "#f85149"))
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
            }

            // Submit button
            Button(action: submit) {
                HStack {
                    if isSubmitting {
                        ProgressView().progressViewStyle(.circular).tint(.white).scaleEffect(0.7)
                    } else {
                        Image(systemName: "play.fill")
                    }
                    Text(isSubmitting ? "予測中..." : "構造予測を実行")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(PrimaryButtonStyle())
            .disabled(components.isEmpty || isSubmitting)
            .padding(12)

            Divider().overlay(Color(hex: "#30363d"))

            // Job list
            JobListView(jobs: $jobs, selectedId: $selectedJobId, structureURL: $structureURL)
        }
        .background(Color(hex: "#0d1117"))
    }

    private func addProtein() {
        components.append(ComponentInput(type: .protein, sequence: ""))
    }

    private func remove(_ comp: ComponentInput) {
        components.removeAll { $0.id == comp.id }
    }

    private func submit() {
        errorMessage = nil
        isSubmitting = true

        // Build spec: map each ComponentInput to API ComponentSpec
        let specComponents = components.map { $0.toSpecComponent() }
        let spec = Spec(name: nil, components: specComponents)
        let req = PredictionRequest(spec: spec, title: nil, parentId: nil, origin: "user")

        Task {
            do {
                let job = try await api.submitPrediction(req)
                jobs.insert(job, at: 0)
                selectedJobId = job.id
            } catch {
                errorMessage = error.localizedDescription
            }
            isSubmitting = false
        }
    }
}

// MARK: - Component Row

struct ComponentRow: View {
    @Binding var component: ComponentInput
    var onDelete: () -> Void

    @State private var isExpanded = true

    var sequencePlaceholder: String {
        switch component.type {
        case .protein, .rna, .dna: return "アミノ酸/塩基配列を入力..."
        case .ligand:              return "SMILES文字列を入力..."
        case .ccd:                 return "CCDコードを入力 (例: ATP)"
        case .metal:               return "金属イオンコードを入力 (例: ZN)"
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header row
            HStack {
                Image(systemName: icon(for: component.type))
                    .foregroundColor(color(for: component.type))
                    .frame(width: 20)

                Picker("", selection: $component.type) {
                    ForEach(ComponentInput.ComponentType.allCases, id: \.self) { t in
                        Text(label(for: t)).tag(t)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(width: 100)

                Spacer()

                Button(action: { isExpanded.toggle() }) {
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption)
                        .foregroundColor(Color(hex: "#8b949e"))
                }
                .buttonStyle(.plain)

                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.caption)
                        .foregroundColor(Color(hex: "#f85149"))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)

            if isExpanded {
                Divider().overlay(Color(hex: "#30363d"))

                ZStack(alignment: .topLeading) {
                    if component.sequence.isEmpty {
                        Text(sequencePlaceholder)
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(Color(hex: "#30363d"))
                            .padding(.top, 8)
                            .padding(.leading, 5)
                            .allowsHitTesting(false)
                    }
                    TextEditor(text: $component.sequence)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(Color(hex: "#e6edf3"))
                        .scrollContentBackground(.hidden)
                        .background(Color(hex: "#0d1117"))
                        .frame(minHeight: 80)
                }
                .padding(8)

                if component.type == .protein || component.type == .rna || component.type == .dna {
                    HStack {
                        Text("\(component.sequence.count) 残基")
                            .font(.system(size: 11))
                            .foregroundColor(Color(hex: "#8b949e"))
                        Spacer()
                        if component.type == .protein && component.sequence.count < 60 && !component.sequence.isEmpty {
                            Text("⚠ 60残基未満")
                                .font(.system(size: 11))
                                .foregroundColor(Color(hex: "#e3b341"))
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.bottom, 8)
                }
            }
        }
        .background(Color(hex: "#161b22"))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(hex: "#30363d"), lineWidth: 1)
        )
    }

    private func icon(for t: ComponentInput.ComponentType) -> String {
        switch t {
        case .protein: return "bolt.circle"
        case .rna:     return "waveform"
        case .dna:     return "ladder.rungs"
        case .ligand:  return "molecule"
        case .ccd:     return "tag"
        case .metal:   return "atom"
        }
    }

    private func color(for t: ComponentInput.ComponentType) -> Color {
        switch t {
        case .protein: return Color(hex: "#58a6ff")
        case .rna:     return Color(hex: "#3fb950")
        case .dna:     return Color(hex: "#bc8cff")
        case .ligand:  return Color(hex: "#e3b341")
        case .ccd:     return Color(hex: "#ffa657")
        case .metal:   return Color(hex: "#ff7b72")
        }
    }

    private func label(for t: ComponentInput.ComponentType) -> String {
        switch t {
        case .protein: return "タンパク質"
        case .rna:     return "RNA"
        case .dna:     return "DNA"
        case .ligand:  return "リガンド"
        case .ccd:     return "CCD"
        case .metal:   return "金属イオン"
        }
    }
}

// MARK: - Job List

struct JobListView: View {
    @EnvironmentObject var api: APIClient
    @Binding var jobs: [Job]
    @Binding var selectedId: String?
    @Binding var structureURL: URL?

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("ジョブ履歴")
                    .font(.subheadline)
                    .foregroundColor(Color(hex: "#8b949e"))
                Spacer()
                Button(action: refresh) {
                    Image(systemName: "arrow.clockwise")
                        .font(.caption)
                        .foregroundColor(Color(hex: "#8b949e"))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)

            ScrollView {
                LazyVStack(spacing: 4) {
                    ForEach(jobs) { job in
                        JobRow(job: job, isSelected: job.id == selectedId) {
                            selectedId = job.id
                            if job.status == .succeeded {
                                structureURL = api.pdb(jobId: job.id)
                            }
                        }
                    }
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 8)
            }
            .frame(maxHeight: 220)
        }
    }

    private func refresh() {
        Task {
            if let fetched = try? await api.listJobs() {
                jobs = fetched
            }
        }
    }
}

struct JobRow: View {
    var job: Job
    var isSelected: Bool
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                VStack(alignment: .leading, spacing: 1) {
                    Text(job.title ?? String(job.id.prefix(8)))
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(.white)
                        .lineLimit(1)
                    if job.status == .running, let p = job.progress {
                        Text("\(Int(p * 100))%")
                            .font(.system(size: 10))
                            .foregroundColor(Color(hex: "#e3b341"))
                    }
                }
                Spacer()
                Text(statusLabel)
                    .font(.system(size: 11))
                    .foregroundColor(statusColor)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(isSelected ? Color(hex: "#1f6feb").opacity(0.3) : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
    }

    private var statusColor: Color {
        switch job.status {
        case .succeeded: return Color(hex: "#3fb950")
        case .failed:    return Color(hex: "#f85149")
        case .running:   return Color(hex: "#e3b341")
        case .queued:    return Color(hex: "#8b949e")
        }
    }

    private var statusLabel: String {
        switch job.status {
        case .succeeded: return "完了"
        case .failed:    return "失敗"
        case .running:   return "実行中"
        case .queued:    return "待機中"
        }
    }
}

// MARK: - Primary Button Style

struct PrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 14, weight: .semibold))
            .foregroundColor(.white)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(isEnabled
                          ? (configuration.isPressed ? Color(hex: "#1f6feb") : Color(hex: "#238636"))
                          : Color(hex: "#30363d"))
            )
    }
}
