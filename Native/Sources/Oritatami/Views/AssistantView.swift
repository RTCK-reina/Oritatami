import SwiftUI
import Combine

/// Right panel — AI assistant (Qwen) + job polling.
struct AssistantView: View {
    @EnvironmentObject var api: APIClient
    @Binding var components: [ComponentInput]
    @Binding var jobs: [Job]
    @Binding var selectedJobId: String?
    @Binding var structureURL: URL?

    @State private var history: [ChatMessage] = []
    @State private var input = ""
    @State private var isLoading = false
    @State private var safeguardAlert = false
    @State private var errorMessage: String?
    @State private var threadId: String?
    @State private var pollingTask: Task<Void, Never>?

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Label("AIアシスタント", systemImage: "sparkles")
                    .font(.headline)
                    .foregroundColor(.white)
                Spacer()
                Text("Qwen3")
                    .font(.system(size: 11))
                    .foregroundColor(Color(hex: "#8b949e"))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color(hex: "#30363d"))
                    .clipShape(Capsule())
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color(hex: "#161b22"))

            Divider().overlay(Color(hex: "#30363d"))

            // Message list
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        if history.isEmpty {
                            WelcomeMessage()
                        }
                        ForEach(history) { msg in
                            MessageBubble(message: msg, onApply: { proposal in
                                applyProposal(proposal)
                            })
                            .id(msg.id)
                        }
                        if isLoading {
                            TypingIndicator()
                                .id("typing")
                        }
                    }
                    .padding(12)
                }
                .onChange(of: history.count) {
                    withAnimation { proxy.scrollTo(history.last?.id) }
                }
                .onChange(of: isLoading) { _, loading in
                    if loading { withAnimation { proxy.scrollTo("typing") } }
                }
            }

            // Error / safeguard banner
            if safeguardAlert {
                SafeguardBanner { safeguardAlert = false }
            } else if let err = errorMessage {
                ErrorBanner(message: err) { errorMessage = nil }
            }

            Divider().overlay(Color(hex: "#30363d"))

            // Quick suggestions (only when empty)
            if history.isEmpty {
                QuickSuggestions { text in
                    input = text
                    send()
                }
            }

            // Input bar
            HStack(spacing: 8) {
                TextField("タンパク質の設計や解析について質問...", text: $input, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13))
                    .foregroundColor(.white)
                    .lineLimit(1...4)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(hex: "#161b22"))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color(hex: "#30363d"), lineWidth: 1)
                    )
                    .onSubmit { if !input.isEmpty { send() } }

                Button(action: send) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 28))
                        .foregroundColor(input.isEmpty || isLoading
                                         ? Color(hex: "#30363d")
                                         : Color(hex: "#58a6ff"))
                }
                .buttonStyle(.plain)
                .disabled(input.isEmpty || isLoading)
            }
            .padding(12)
            .background(Color(hex: "#0d1117"))
        }
        .background(Color(hex: "#0d1117"))
        .onAppear { startPolling() }
        .onDisappear { pollingTask?.cancel() }
    }

    // MARK: - Send

    private func send() {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        input = ""
        safeguardAlert = false
        errorMessage = nil
        history.append(ChatMessage(role: "user", content: text))
        isLoading = true

        // Build workbench spec from current components
        let workbenchSpec = Spec(
            name: nil,
            components: components.map { $0.toSpecComponent() }
        )

        let req = AssistantRequest(
            threadId: threadId,
            mode: "chat",
            message: text,
            workbench: workbenchSpec,
            jobId: selectedJobId,
            scanJobId: nil,
            focusChain: nil,
            count: 1
        )

        Task {
            do {
                let resp = try await api.ask(req)
                threadId = resp.threadId

                let proposals = resp.proposals.isEmpty ? nil : resp.proposals
                history.append(ChatMessage(
                    role: "assistant",
                    content: resp.reply,
                    proposals: proposals
                ))
            } catch APIError.safeguard(let msg) {
                safeguardAlert = true
                history.removeLast()    // remove the user message that triggered it
                _ = msg
            } catch {
                errorMessage = error.localizedDescription
            }
            isLoading = false
        }
    }

    // MARK: - Apply proposal

    private func applyProposal(_ proposal: Proposal) {
        guard let ap = proposal.apply else { return }
        switch ap.action {
        case "new_protein":
            // 新しいタンパク質デザイン → ワークベンチをこの1つに置き換え
            if let comp = ap.component {
                components = [comp.toComponentInput()]
            }
        case "add_component":
            // コンポーネントを追加 (リガンド・複合体パートナーなど)
            if let comp = ap.component {
                components.append(comp.toComponentInput())
            }
        case "mutate":
            // 変異の適用: 対象チェーンのコンポーネントを見つけてラベルを更新
            // (実際の配列変換はバックエンドで行われるため、ここではラベルを更新するのみ)
            if let muts = ap.mutations, let chain = ap.chain {
                let mutLabel = muts.joined(separator: "/")
                for i in components.indices {
                    if components[i].label == chain || (chain == "A" && i == 0) {
                        components[i].label = mutLabel.isEmpty ? components[i].label : mutLabel
                        break
                    }
                }
            }
        default:
            // フォールバック: apply.component があれば追加
            if let comp = ap.component {
                components.append(comp.toComponentInput())
            }
        }
    }

    // MARK: - Job polling (long-poll via changes endpoint + periodic fallback)

    private func startPolling() {
        pollingTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                await pollRunningJobs()
            }
        }
    }

    private func pollRunningJobs() async {
        let active = jobs.filter { $0.status == .running || $0.status == .queued }
        for job in active {
            guard let updated = try? await api.getJob(job.id) else { continue }
            if let idx = jobs.firstIndex(where: { $0.id == job.id }) {
                jobs[idx] = updated
            }
            if updated.status == .succeeded && selectedJobId == job.id {
                structureURL = api.pdb(jobId: job.id)
            }
        }
    }
}

// MARK: - MessageBubble

struct MessageBubble: View {
    var message: ChatMessage
    var onApply: (Proposal) -> Void

    var isUser: Bool { message.role == "user" }

    var body: some View {
        VStack(alignment: isUser ? .trailing : .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 8) {
                if isUser { Spacer(minLength: 40) }

                if !isUser {
                    Image(systemName: "sparkles")
                        .font(.system(size: 12))
                        .foregroundColor(Color(hex: "#58a6ff"))
                        .frame(width: 24, height: 24)
                        .background(Color(hex: "#1f2937"))
                        .clipShape(Circle())
                }

                Text(message.content)
                    .font(.system(size: 13))
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(isUser ? Color(hex: "#1f6feb") : Color(hex: "#161b22"))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .textSelection(.enabled)

                if !isUser { Spacer(minLength: 40) }
            }

            // Proposal apply buttons
            if let proposals = message.proposals, !proposals.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("提案されたコンポーネント:")
                        .font(.system(size: 11))
                        .foregroundColor(Color(hex: "#8b949e"))
                        .padding(.leading, 32)

                    ForEach(proposals) { proposal in
                        if proposal.isApplicable {
                            let actionIcon = proposal.apply?.action == "mutate" ? "dna" : "wrench.and.screwdriver"
                            let actionLabel: String = {
                                switch proposal.apply?.action {
                                case "mutate":
                                    let muts = proposal.apply?.mutations?.joined(separator: ", ") ?? ""
                                    return "変異を適用: \(muts)"
                                case "add_component":
                                    return "コンポーネントを追加"
                                default:
                                    return "ワークベンチに適用"
                                }
                            }()
                            Button(action: { onApply(proposal) }) {
                                HStack(spacing: 6) {
                                    Image(systemName: actionIcon)
                                        .font(.system(size: 11))
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(proposal.title ?? "提案を適用")
                                            .font(.system(size: 12, weight: .medium))
                                        Text(actionLabel)
                                            .font(.system(size: 11))
                                            .foregroundColor(Color(hex: "#8b949e"))
                                    }
                                    Spacer()
                                    Image(systemName: "arrow.right")
                                        .font(.system(size: 10))
                                }
                                .foregroundColor(Color(hex: "#58a6ff"))
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .background(Color(hex: "#1f2937"))
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color(hex: "#58a6ff").opacity(0.4), lineWidth: 1)
                                )
                            }
                            .buttonStyle(.plain)
                            .padding(.leading, 32)
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Sub-views

struct WelcomeMessage: View {
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "sparkles")
                .font(.system(size: 32))
                .foregroundColor(Color(hex: "#58a6ff"))
            Text("AIアシスタント")
                .font(.headline)
                .foregroundColor(.white)
            Text("タンパク質の設計、変異解析、構造の解釈など\nなんでも質問してください")
                .font(.system(size: 13))
                .foregroundColor(Color(hex: "#8b949e"))
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 32)
    }
}

struct TypingIndicator: View {
    @State private var dots = 0
    let timer = Timer.publish(every: 0.4, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "sparkles")
                .font(.system(size: 12))
                .foregroundColor(Color(hex: "#58a6ff"))
            Text(String(repeating: "●", count: dots + 1))
                .font(.system(size: 10))
                .foregroundColor(Color(hex: "#8b949e"))
                .onReceive(timer) { _ in dots = (dots + 1) % 3 }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(hex: "#161b22"))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

struct QuickSuggestions: View {
    var onTap: (String) -> Void

    private let suggestions = [
        "インスリンの構造を予測するシーケンスを教えて",
        "亜鉛を含む金属タンパク質を設計したい",
        "ATPと結合するドッキング複合体を作りたい",
        "この変異はタンパク質の安定性に影響しますか？",
    ]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(suggestions, id: \.self) { s in
                    Button(action: { onTap(s) }) {
                        Text(s)
                            .font(.system(size: 11))
                            .foregroundColor(Color(hex: "#58a6ff"))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(Color(hex: "#1f2937"))
                            .clipShape(Capsule())
                            .overlay(Capsule().stroke(Color(hex: "#30363d"), lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
        .background(Color(hex: "#0d1117"))
    }
}

struct SafeguardBanner: View {
    var onDismiss: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "shield.slash")
                .foregroundColor(Color(hex: "#e3b341"))
            Text("AIが生物学的安全フィルターを適用しました。質問を言い換えてみてください。")
                .font(.system(size: 12))
                .foregroundColor(Color(hex: "#e3b341"))
            Spacer()
            Button(action: onDismiss) {
                Image(systemName: "xmark")
                    .font(.caption)
                    .foregroundColor(Color(hex: "#8b949e"))
            }
            .buttonStyle(.plain)
        }
        .padding(10)
        .background(Color(hex: "#2d1b00"))
        .overlay(
            Rectangle().frame(height: 1).foregroundColor(Color(hex: "#e3b341").opacity(0.4)),
            alignment: .top
        )
    }
}

struct ErrorBanner: View {
    var message: String
    var onDismiss: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle")
                .foregroundColor(Color(hex: "#f85149"))
            Text(message)
                .font(.system(size: 12))
                .foregroundColor(Color(hex: "#f85149"))
                .lineLimit(2)
            Spacer()
            Button(action: onDismiss) {
                Image(systemName: "xmark")
                    .font(.caption)
                    .foregroundColor(Color(hex: "#8b949e"))
            }
            .buttonStyle(.plain)
        }
        .padding(10)
        .background(Color(hex: "#2d1117"))
        .overlay(
            Rectangle().frame(height: 1).foregroundColor(Color(hex: "#f85149").opacity(0.4)),
            alignment: .top
        )
    }
}
