import { uiEvents } from './uiEvents';
import type {
    Accelerator,
    ChemInfo,
    CompareResult,
    Estimate,
    FullJob,
    Geometry,
    GpuState,
    SsdInfo,
    Health,
    ImportedStructure,
    JobSummary,
    LibraryItem,
    LlmCall,
    LlmStatus,
    Proposal,
    SearchRecord,
    Settings,
    SpecOut,
    StorageInfo,
    Thread,
    UniProtHit,
} from './types';

/** Whether the local API answered the last request. The UI shows a reconnect banner while offline. */
type ConnectionListener = (online: boolean) => void;
const connectionListeners = new Set<ConnectionListener>();
let online = true;

function setOnline(value: boolean) {
    if (value === online) return;
    online = value;
    connectionListeners.forEach(fn => fn(value));
}

export const connection = {
    get online() {
        return online;
    },
    subscribe(fn: ConnectionListener): () => void {
        connectionListeners.add(fn);
        return () => connectionListeners.delete(fn);
    },
};

export class ApiError extends Error {
    constructor(public status: number, message: string, public code?: string) {
        super(message);
    }
}

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
    let resp: Response;
    try {
        resp = await fetch(path, {
            method,
            headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
            body: body !== undefined ? JSON.stringify(body) : undefined,
            signal,
        });
    } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') throw e;
        setOnline(false);
        throw new ApiError(0, `バックエンドに接続できません (${e instanceof Error ? e.message : String(e)})`);
    }
    setOnline(true);
    const text = await resp.text();
    if (!resp.ok) {
        let detail = text;
        let code: string | undefined;
        try {
            const parsed: unknown = JSON.parse(text);
            if (parsed && typeof parsed === 'object') {
                const p = parsed as Record<string, unknown>;
                if ('detail' in p) {
                    const d = p.detail;
                    detail = typeof d === 'string' ? d : JSON.stringify(d);
                }
                if ('code' in p && typeof p.code === 'string') {
                    code = p.code;
                }
            }
        } catch {
            // non-JSON error body: keep raw text
        }
        throw new ApiError(resp.status, detail || `HTTP ${resp.status}`, code);
    }
    const type = resp.headers.get('content-type') ?? '';
    return (type.includes('application/json') ? JSON.parse(text) : text) as T;
}

/**
 * One press, one request.
 *
 * A prediction can take minutes, and during it the window stops repainting promptly — so a
 * button that looks stuck invites a second and a third press, and each one used to queue
 * another job / delete another row / start another download. Any *changing* request
 * (POST/PATCH/DELETE) is therefore keyed by method + path + body while it is in flight: an
 * identical one that arrives before the first has answered is not sent. It rides on the first
 * request's promise, so the caller still gets its result, and the UI is told so it can say
 * plainly that the second press was not needed.
 *
 * Reads are not deduplicated — polling and parallel loads of the same resource are normal.
 */
const inFlight = new Map<string, Promise<unknown>>();
/**
 * POSTs that are questions rather than actions — estimates, sequence and chemistry checks, a
 * structural superposition. Repeating one changes nothing, and the UI fires them as the
 * workbench is edited rather than because anybody pressed a button, so they join the request
 * already running without a word and are released as soon as it answers.
 */
const QUERY_PATHS = ['/api/estimate', '/api/sequence/', '/api/chem/', '/api/compare'];
/** An identical request is still refused for this long after the first one answered: a fast
 *  endpoint would otherwise let a double-click through, which is the case this exists for. */
const COOLDOWN_MS = 900;

function guarded<T>(method: string, path: string, body?: unknown): Promise<T> {
    const query = QUERY_PATHS.some(prefix => path.startsWith(prefix));
    const key = `${method} ${path} ${body === undefined ? '' : JSON.stringify(body)}`;
    const running = inFlight.get(key);
    if (running) {
        if (!query) uiEvents.emit('duplicateAction', path);
        return running as Promise<T>;
    }
    const p = request<T>(method, path, body);
    const release = () => {
        if (query) {
            if (inFlight.get(key) === p) inFlight.delete(key);
            return;
        }
        window.setTimeout(() => {
            if (inFlight.get(key) === p) inFlight.delete(key);
        }, COOLDOWN_MS);
    };
    p.then(release, release);
    inFlight.set(key, p);
    return p;
}

/** True while a changing request for this path prefix is on its way (for busy indicators). */
export function isBusy(pathPrefix: string): boolean {
    for (const key of inFlight.keys()) {
        if (key.split(' ')[1]?.startsWith(pathPrefix)) return true;
    }
    return false;
}

const get = <T>(p: string, signal?: AbortSignal) => request<T>('GET', p, undefined, signal);
const post = <T>(p: string, b?: unknown) => guarded<T>('POST', p, b ?? {});
const patch = <T>(p: string, b: unknown) => guarded<T>('PATCH', p, b);
const del = <T>(p: string) => guarded<T>('DELETE', p);
const q = encodeURIComponent;

export interface PdbWatcherConfig {
    enabled: boolean;
    max_per_poll: number;
    min_seq_len: number;
    max_seq_len: number;
    last_checked: string | null;
}

export type LeaderboardMetric =
    'mean_plddt' | 'core_plddt' | 'confidence_score' | 'iptm' | 'ptm' | 'complex_plddt';

export interface LeaderboardRow {
    rank: number;
    id: string;
    title: string | null;
    origin: string;
    created_at: number;
    finished_at: number | null;
    starred: boolean;
    parent_id: string | null;
    parent_title: string | null;
    autopilot_depth: number;
    mutations: string[];
    elapsed_sec: number | null;
    /** short label for how the prediction was computed (MSA route, sample count) */
    regime: string;
    regime_label: string;
    /** false when the parent ran under different conditions, so the delta is not a result */
    comparable_to_parent: boolean | null;
    /** on the score-vs-mutation-count front: nothing else is both better and simpler */
    pareto: boolean;
    lineage_root: string | null;
    mean_plddt: number | null;
    core_plddt: number | null;
    confidence_score: number | null;
    ptm: number | null;
    iptm: number | null;
    complex_plddt: number | null;
    affinity_ic50_um: number | null;
    affinity_delta_g: number | null;
    delta_mean_plddt: number | null;
    delta_core_plddt: number | null;
    delta_confidence_score: number | null;
    delta_iptm: number | null;
    delta_ptm: number | null;
    delta_complex_plddt: number | null;
}

export interface Leaderboard {
    metric: LeaderboardMetric;
    metrics: LeaderboardMetric[];
    count: number;
    returned: number;
    improved_count: number;
    best_id: string | null;
    conditions: { mixed: boolean; regimes: { label: string; short: string; msa: string; count: number }[] };
    pareto_count: number;
    rows: LeaderboardRow[];
}

export interface SearchHistory {
    predictions: number;
    pairs: number;
    skipped_regime: number;
    repeats: number;
    mean_delta: number | null;
    improved_rate: number | null;
    sigma: number | null;
    sigma_by_msa: Record<string, number>;
    distinct_experiments: number;
    configured_delta: number;
    suggested_delta: number | null;
    by_position: Record<string, { n: number; mean: number }>;
    worst_positions: { position: number; n: number; mean: number }[];
    repeated_substitutions: { code: string; n: number; mean: number; best: number }[];
}

/** When the whole queue is expected to be done — duration and wall-clock. */
export interface QueueEta {
    jobs: number;
    counted: number;
    /** jobs whose remaining time cannot be estimated; the total is a lower bound while > 0 */
    unknown: number;
    seconds: number;
    /** true when every job in the queue could be timed */
    complete: boolean;
    /** wall clock the queue empties — only when `complete` */
    finish_at: number | null;
    /** wall clock the queue is busy until at least, when something could not be timed */
    at_least_until: number | null;
    now: number;
    items: QueueEtaItem[];
}

export interface QueueEtaItem {
    id: string;
    title: string;
    status: string;
    kind: string;
    seconds: number | null;
    /** swap = history plus the predicted paging cost; baseline = history alone; unknown = no honest answer */
    basis?: 'swap' | 'baseline' | 'unknown';
    progress?: number | null;
    /** CPU seconds per wall second; falls with job size even when healthy, so not a health signal */
    efficiency?: number | null;
    /** share of CPU time spent computing rather than moving pages — this IS the health signal */
    user_share?: number | null;
    regime?: 'resident' | 'paging' | 'unknown';
    /** GB the run is expected to need beyond what can stay resident */
    overage_gb?: number;
    /** seconds attributed to paging that overage in and out */
    swap_seconds?: number;
    overrun?: boolean;
    note?: string;
}

export interface RiskReason { kind: string; label: string; severity: string }

export interface ProtectedPosition {
    position: number;
    residue: string;
    severity: string;
    reasons: RiskReason[];
    default_on: boolean;
}

export interface ProtectedSuggestion {
    chain: string;
    length: number;
    positions: ProtectedPosition[];
    default_on_count: number;
    sources: {
        uniprot: string | null;
        interface_residues: number;
        disordered_residues: number;
        conserved_residues: number;
        scan_job: string | null;
        root_job: string | null;
    };
    notes: string[];
}

export interface MemoryTrend {
    /** physical footprint of the app process right now, GB */
    current_gb: number | null;
    /** how many jobs have been sampled this session */
    samples: number;
    first_gb?: number;
    last_gb?: number;
    /** last minus first; positive means the app is keeping something between jobs */
    growth_gb: number | null;
    climbing: boolean;
    growth_warn_gb: number;
    series?: { job: number; gb: number }[];
    torch_mps: { driver_allocated_gb: number; in_use_gb: number } | null;
}

export interface FunctionFinding {
    kind: string;
    severity: string;
    text: string;
    chain?: string;
    mutation?: string;
    position?: number;
    reasons?: string[];
    detail?: Record<string, number>;
}

export interface FunctionRisk {
    job_id: string;
    level: 'ok' | 'warn' | 'danger';
    findings: FunctionFinding[];
    /** measured on first view for results predicted before the check existed */
    geometry: Geometry | { error: string } | null;
}

export interface AutopilotStatus {
    enabled: boolean;
    daily_budget: number;
    used_today: number;
    remaining_today: number | null;
    max_depth: number;
    queued: number;
    max_queued: number;
    max_variants_per_job: number;
    strategy?: string;
    selection?: string;
    climb_patience?: number;
    experiment?: string;
    min_esm_llr: number;
    noise_sigma?: number | null;
    noise_repeats?: number;
    suggested_delta?: number | null;
    delta_below_noise?: boolean;
    history_pairs?: number;
    history_mean_delta?: number | null;
    history_improved_rate?: number | null;
    protected_residues: string;
    min_disk_gb: number;
    disk_free_gb: number;
    accepting: boolean;
    blocked_reason: string | null;
}

export interface AutopilotAnalysis {
    mode: string;
    chain?: string;
    thread_id?: string | null;
    reply?: string;
    proposals?: Proposal[];
    reply_issues?: string[];
    elapsed_sec?: number;
    error?: string;
}

export interface AutopilotResult {
    job_id: string;
    started_at: number;
    finished_at?: number;
    elapsed_sec?: number;
    analyses: AutopilotAnalysis[];
}


export const api = {
    ping: () => get<{ ok: boolean; version: string }>('/api/ping'),
    health: () => get<Health>('/api/health'),
    settings: () => get<Settings>('/api/settings'),
    updateSettings: (s: Partial<Settings>) => patch<Settings>('/api/settings', s),

    llmStatus: () => get<LlmStatus>('/api/llm/status'),
    llmStart: () => post<{ running: boolean; started: boolean; error?: string; installing?: boolean;
        binary?: string | null; source?: string }>('/api/llm/start'),
    llmInstall: () => post<LlmStatus['install']>('/api/llm/install'),
    llmPull: (model: string) => post<LlmStatus['pull']>('/api/llm/pull', { model }),

    mutate: (sequence: string, mutations: string) =>
        post<{ sequence: string; mutations: string[] }>('/api/sequence/mutate', { sequence, mutations }),
    validateSequence: (sequence: string, type: string) =>
        post<{ sequence: string; length: number }>('/api/sequence/validate', { sequence, type }),

    chemDescribe: (smiles: string) => post<ChemInfo>('/api/chem/describe', { smiles }),
    pubchem: (name: string) =>
        get<{ name: string; cid: number; smiles: string; formula: string; describe: ChemInfo; svg: string }>(
            `/api/chem/pubchem?name=${q(name)}`,
        ),
    ccd: (code: string) =>
        get<{ ccd: string; name?: string; formula?: string; smiles?: string; svg?: string }>(`/api/chem/ccd/${q(code)}`),

    uniprotSearch: (text: string) => get<UniProtHit[]>(`/api/uniprot/search?q=${q(text)}`),
    uniprotEntry: (acc: string) =>
        get<{ accession: string; name: string; organism: string; sequence: string; features: unknown[]; function: string[] }>(
            `/api/uniprot/${q(acc)}`,
        ),
    pdbSearch: (text: string) => get<{ id: string; title: string }[]>(`/api/pdb/search?q=${q(text)}`),
    importPdb: (id: string, assembly = false) =>
        post<ImportedStructure>('/api/import/pdb', { id, assembly }),
    importAfdb: (id: string) => post<ImportedStructure>('/api/import/afdb', { id }),
    importUpload: (filename: string, content: string) =>
        post<ImportedStructure>('/api/import/upload', { filename, content }),

    submitPredict: (spec: SpecOut, title: string, parentId: string | null, origin = 'user') =>
        post<JobSummary>('/api/jobs/predict', { spec, title, parent_id: parentId, origin }),
    submitScan: (sequence: string, chain: string | null, label: string | null) =>
        post<JobSummary>('/api/jobs/scan', { sequence, chain, label }),
    submitRefine: (body: { sequence: string; label?: string; rounds?: number; fraction?: number; temperature?: number; fixed_positions?: number[]; origin?: string }) =>
        post<JobSummary>('/api/jobs/refine', body),
    jobs: (limit = 2000) => get<JobSummary[]>(`/api/jobs?limit=${limit}`),
    jobChanges: (rev: number, timeout: number, signal?: AbortSignal) =>
        get<{ rev: number; changed: boolean }>(`/api/jobs/changes?rev=${rev}&timeout=${timeout}`, signal),
    submitBatch: (spec: SpecOut, variants: { chain: string; mutations: string[] }[], parentId: string | null, origin = 'user') =>
        post<{ jobs: JobSummary[] }>('/api/jobs/predict/batch', { spec, variants, parent_id: parentId, origin }),
    retryJob: (id: string, body: { msa?: 'single'; accelerator?: Accelerator; diffusion_samples?: number; new_seed?: boolean } = {}) =>
        post<JobSummary>(`/api/jobs/${q(id)}/retry`, body),
    estimate: (spec: SpecOut) => post<Estimate>('/api/estimate', { spec }),
    exportJobToFolder: (id: string) => post<{ path: string; bytes: number }>(`/api/jobs/${q(id)}/export`),
    exportZipUrl: (id: string) => `/api/jobs/${q(id)}/export.zip`,
    structurePdbUrl: (id: string, model: number) => `/api/jobs/${q(id)}/structure.pdb?model=${model}`,
    saveDataUrl: (filename: string, dataUrl: string, reveal = true) =>
        post<{ path: string }>('/api/files/save', { filename, data_url: dataUrl, reveal }),
    notify: (title: string, message: string) => post<{ shown: boolean }>('/api/notify', { title, message }),
    storage: () => get<StorageInfo>('/api/storage'),
    cleanupStorage: (body: { intermediate: boolean; aligned_older_than_days: number | null; delete_failed_jobs: boolean }) =>
        post<{ freed_bytes: number; jobs_cleaned: number; aligned_removed: number; jobs_deleted: number; storage: StorageInfo }>(
            '/api/storage/cleanup', body),
    job: (id: string) => get<FullJob>(`/api/jobs/${q(id)}`),
    patchJob: (id: string, body: { title?: string; starred?: boolean }) => patch<JobSummary>(`/api/jobs/${q(id)}`, body),
    cancelJob: (id: string) => post<JobSummary>(`/api/jobs/${q(id)}/cancel`),
    reorderJob: (id: string, action: 'top' | 'up' | 'down' | 'bottom') =>
        post<JobSummary>(`/api/jobs/${q(id)}/reorder`, { action }),
    cancelQueued: (kind?: 'predict' | 'scan' | 'refine', includeRunning = false) =>
        post<{ cancelled: string[]; count: number }>('/api/jobs/queue/cancel_all',
            { kind: kind ?? null, include_running: includeRunning }),
    gpu: (fresh = false) => get<GpuState>(`/api/system/gpu${fresh ? '?fresh=true' : ''}`),
    setGpuLimit: (wiredLimitMb: number) =>
        post<GpuState>('/api/system/gpu', { wired_limit_mb: wiredLimitMb }),
    ssd: (fresh = false) => get<SsdInfo>(`/api/system/ssd${fresh ? '?fresh=true' : ''}`),
    memory: () => get<MemoryTrend>('/api/system/memory'),
    releaseMemory: () => post<{ freed_gb: number; esm_unloaded: boolean; footprint_gb: number | null }>(
        '/api/system/memory/release', {}),
    deleteJob: (id: string) => del<{ deleted: string }>(`/api/jobs/${q(id)}`),
    jobLog: (id: string) => get<string>(`/api/jobs/${q(id)}/log`),
    revealJob: (id: string) => post<{ opened: string }>(`/api/jobs/${q(id)}/reveal`),
    jobFileUrl: (id: string, rel: string) => `/api/jobs/${q(id)}/files/${rel.split('/').map(q).join('/')}`,

    compare: (fixed: { job_id: string; model: number }, moving: { job_id: string; model: number }) =>
        post<CompareResult>('/api/compare', { fixed, moving }),

    ask: (body: {
        thread_id: string | null;
        mode: string;
        message: string;
        workbench: SpecOut;
        job_id: string | null;
        scan_job_id: string | null;
        focus_chain: string | null;
        count: number;
        heavy?: boolean;
    }) => post<{ thread_id: string; reply: string; proposals: Proposal[]; reply_issues?: string[];
        corrected?: boolean; model?: string; elapsed_sec: number }>('/api/assistant/ask', body),
    searches: (limit = 100) => get<SearchRecord[]>(`/api/searches?limit=${limit}`),
    clearSearches: () => del<{ deleted: number }>('/api/searches'),
    llmCalls: (limit = 50, origin?: 'user' | 'autopilot') =>
        get<{ total: number; calls: LlmCall[] }>(`/api/llm/calls?limit=${limit}${origin ? `&origin=${origin}` : ''}`),
    exportLlmCalls: () => post<{ path: string; count: number; bytes: number }>('/api/llm/calls/export', {}),
    threads: () => get<Omit<Thread, 'messages'>[]>('/api/assistant/threads'),
    thread: (id: string) => get<Thread>(`/api/assistant/threads/${q(id)}`),
    autopilot: (jobId: string) => get<AutopilotResult>(`/api/jobs/${q(jobId)}/autopilot`),
    deleteThread: (id: string) => del<{ deleted: string }>(`/api/assistant/threads/${q(id)}`),

    leaderboard: (metric: LeaderboardMetric = 'mean_plddt', limit = 1000) =>
        get<Leaderboard>(`/api/leaderboard?metric=${q(metric)}&limit=${limit}`),
    autopilotStatus: () => get<AutopilotStatus>('/api/autopilot/status'),
    queueEta: () => get<QueueEta>('/api/jobs/eta'),
    functionRisk: (id: string) => get<FunctionRisk>(`/api/jobs/${q(id)}/function_risk`),
    protectedSuggest: (id: string, chain?: string) =>
        get<ProtectedSuggestion>(`/api/jobs/${q(id)}/protected_suggest${chain ? `?chain=${q(chain)}` : ''}`),
    searchHistory: () => get<SearchHistory>('/api/history'),

    pdbWatcher: {
        config: () => get<PdbWatcherConfig>('/api/pdb_watcher/config'),
        setConfig: (cfg: Partial<PdbWatcherConfig>) =>
            post<PdbWatcherConfig>('/api/pdb_watcher/config', cfg),
        pollNow: () => post<{ status: string }>('/api/pdb_watcher/poll_now', {}),
    },

    library: () => get<LibraryItem[]>('/api/library'),
    addLibrary: (type: string, name: string, data: Record<string, unknown>) =>
        post<LibraryItem>('/api/library', { type, name, data }),
    deleteLibrary: (id: string) => del<{ deleted: string }>(`/api/library/${q(id)}`),
};

export function errorMessage(e: unknown): string {
    if (e instanceof Error) return e.message;
    return String(e);
}

/**
 * Start a file download without leaving the app. A hidden iframe keeps error responses from
 * replacing the page; in the native window pywebview turns the attachment into a save panel.
 */
export function downloadFile(url: string): void {
    const frame = document.createElement('iframe');
    frame.style.display = 'none';
    frame.src = url;
    document.body.appendChild(frame);
    window.setTimeout(() => frame.remove(), 120_000);
}

export function formatBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let v = n / 1024;
    let i = 0;
    while (v >= 1024 && i < units.length - 1) {
        v /= 1024;
        i++;
    }
    return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}`;
}
