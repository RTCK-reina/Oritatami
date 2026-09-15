import Foundation

// MARK: - Job

struct Job: Identifiable, Codable, Equatable {
    let id: String
    var status: JobStatus
    var title: String?
    var kind: String?
    var progress: Double?
    var error: String?
    var createdAt: Date?

    enum JobStatus: String, Codable {
        case queued, running, succeeded, failed
    }
}

// MARK: - Component (UI model — not encoded directly to API)

struct ComponentInput: Identifiable, Equatable {
    var id: String = UUID().uuidString
    var type: ComponentType
    var sequence: String = ""   // also carries SMILES (ligand) or CCD code (ccd/metal)
    var copies: Int?
    var label: String?

    enum ComponentType: String, CaseIterable {
        case protein, rna, dna, ligand, ccd, metal
    }

    /// Convert to API spec component (handles field routing by type)
    func toSpecComponent() -> ComponentSpec {
        let seq = sequence.isEmpty ? nil : sequence
        switch type {
        case .ligand:
            return ComponentSpec(type: type.rawValue, smiles: seq, copies: copies, label: label)
        case .ccd, .metal:
            return ComponentSpec(type: type.rawValue, ccd: seq, copies: copies, label: label)
        default:
            return ComponentSpec(type: type.rawValue, sequence: seq, copies: copies, label: label)
        }
    }
}

// MARK: - Component spec (API model — request & response)

struct ComponentSpec: Codable, Equatable {
    var type: String
    var sequence: String?
    var smiles: String?
    var ccd: String?
    var copies: Int?
    var label: String?

    /// Back-convert to UI model
    func toComponentInput() -> ComponentInput {
        ComponentInput(
            type: ComponentInput.ComponentType(rawValue: type) ?? .protein,
            sequence: sequence ?? smiles ?? ccd ?? "",
            copies: copies,
            label: label
        )
    }
}

// MARK: - Spec wrapper

struct Spec: Encodable {
    var name: String?
    var components: [ComponentSpec]
}

// MARK: - Prediction request

struct PredictionRequest: Encodable {
    var spec: Spec
    var title: String?
    var parentId: String?   // → parent_id via snake_case encoder
    var origin: String = "user"
}

// MARK: - ESM / mutation scan

struct MutationScanRequest: Encodable {
    var sequence: String
    var positions: [Int]?
}

struct MutationScanResult: Decodable {
    var scores: [MutationScore]
}

struct MutationScore: Decodable, Identifiable {
    var id: String { "\(position)\(mutant)" }
    var position: Int
    var wildtype: String
    var mutant: String
    var score: Double
}

// MARK: - AI proposals

/// apply フィールド — action ごとに component (追加系) か mutations (変異系) が入る
struct ProposalApply: Decodable {
    var action: String
    var component: ComponentSpec?   // new_protein / add_component
    var chain: String?              // mutate
    var mutations: [String]?        // mutate
}

struct Proposal: Decodable, Identifiable {
    var type: String?
    var title: String?
    var rationale: String?
    var status: String?
    var apply: ProposalApply?

    // Stable ID for ForEach
    var id: String { title ?? UUID().uuidString }

    /// 適用可能かどうか (apply が存在し、何らかのコンポーネントか変異を持つ)
    var isApplicable: Bool {
        guard let a = apply else { return false }
        return a.component != nil || (a.mutations?.isEmpty == false)
    }
}

// MARK: - Assistant

struct AssistantRequest: Encodable {
    var threadId: String?       // → thread_id
    var mode: String = "chat"
    var message: String
    var workbench: Spec?
    var jobId: String?          // → job_id
    var scanJobId: String?      // → scan_job_id
    var focusChain: String?     // → focus_chain
    var count: Int = 1
}

struct AssistantResponse: Decodable {
    var threadId: String        // ← thread_id
    var reply: String
    var proposals: [Proposal]
    var elapsedSec: Double?     // ← elapsed_sec
}

// MARK: - Chat display (local-only, not sent to API)

struct ChatMessage: Identifiable {
    var id: String = UUID().uuidString
    var role: String            // "user" | "assistant"
    var content: String
    var proposals: [Proposal]?
}

// MARK: - Health

struct HealthResponse: Decodable {
    var status: String
    var version: String?
}

// MARK: - API error

struct APIErrorBody: Decodable {
    var detail: String?
    var code: String?
}

enum APIError: Error, LocalizedError {
    case httpError(Int, String)
    case safeguard(String)
    case decoding(Error)
    case network(Error)

    var errorDescription: String? {
        switch self {
        case .httpError(let code, let msg): return "HTTP \(code): \(msg)"
        case .safeguard(let msg):           return "⚠️ \(msg)"
        case .decoding(let e):              return "デコードエラー: \(e)"
        case .network(let e):              return "ネットワークエラー: \(e)"
        }
    }
}
