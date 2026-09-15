import Foundation
import Combine

/// URLSession wrapper for the Oritatami FastAPI backend.
@MainActor
final class APIClient: ObservableObject {
    private let base = BackendManager.baseURL
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        d.dateDecodingStrategy = .iso8601
        return d
    }()
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        return e
    }()

    // MARK: - Generic helpers

    private func url(_ path: String) -> URL {
        URL(string: path, relativeTo: base) ?? base.appendingPathComponent(path)
    }

    func get<T: Decodable>(_ path: String) async throws -> T {
        let (data, resp) = try await URLSession.shared.data(from: url(path))
        try checkStatus(resp, data)
        return try decode(T.self, from: data)
    }

    func post<Body: Encodable, T: Decodable>(_ path: String, body: Body) async throws -> T {
        var req = URLRequest(url: url(path))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try encoder.encode(body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        try checkStatus(resp, data)
        return try decode(T.self, from: data)
    }

    private func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        do { return try decoder.decode(type, from: data) }
        catch { throw APIError.decoding(error) }
    }

    private func checkStatus(_ resp: URLResponse, _ data: Data) throws {
        guard let http = resp as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            if let body = try? decoder.decode(APIErrorBody.self, from: data) {
                if body.code == "safeguard" {
                    throw APIError.safeguard(body.detail ?? "安全フィルターが応答をブロックしました")
                }
                throw APIError.httpError(http.statusCode, body.detail ?? "Unknown error")
            }
            throw APIError.httpError(http.statusCode, String(data: data, encoding: .utf8) ?? "")
        }
    }

    // MARK: - Health

    func health() async throws -> HealthResponse {
        try await get("/api/health")
    }

    // MARK: - Jobs

    func listJobs() async throws -> [Job] {
        try await get("/api/jobs")
    }

    func getJob(_ id: String) async throws -> Job {
        try await get("/api/jobs/\(id)")
    }

    /// POST /api/jobs/predict — submit a structure prediction job
    func submitPrediction(_ req: PredictionRequest) async throws -> Job {
        try await post("/api/jobs/predict", body: req)
    }

    func cancelJob(_ id: String) async throws {
        var r = URLRequest(url: url("/api/jobs/\(id)/cancel"))
        r.httpMethod = "POST"
        let (data, resp) = try await URLSession.shared.data(for: r)
        try checkStatus(resp, data)
    }

    // MARK: - Structure files

    /// PDB structure URL for the Mol* viewer (primary format)
    func pdb(jobId: String) -> URL {
        var c = URLComponents(url: base, resolvingAgainstBaseURL: false)!
        c.path = "/api/jobs/\(jobId)/structure.pdb"
        c.queryItems = [URLQueryItem(name: "model", value: "0")]
        return c.url ?? base
    }

    /// Raw CIF file via the job file server
    func cifFile(jobId: String) -> URL {
        url("/api/jobs/\(jobId)/files/predictions/oritatami_model_0.cif")
    }

    // MARK: - Mutation scan

    func mutationScan(_ req: MutationScanRequest) async throws -> MutationScanResult {
        try await post("/api/jobs/scan", body: req)
    }

    // MARK: - Assistant

    /// POST /api/assistant/ask
    func ask(_ req: AssistantRequest) async throws -> AssistantResponse {
        try await post("/api/assistant/ask", body: req)
    }
}
