export type PolymerType = 'protein' | 'dna' | 'rna';
export type ComponentType = PolymerType | 'ligand';

export interface ComponentSource {
    db: string;
    id: string;
    version?: number;
}

/** A molecule on the workbench. Chains are assigned from order + copies when a job is built. */
export interface WBComponent {
    uid: string;
    type: ComponentType;
    label: string;
    copies: number;
    /** current sequence (polymers) */
    sequence?: string;
    /** sequence the mutations are counted from */
    baseSequence?: string;
    smiles?: string;
    ccd?: string;
    msa?: 'server' | 'single';
    cyclic?: boolean;
    source?: ComponentSource;
    origin?: 'user' | 'qwen';
}

export type Accelerator = 'auto' | 'mps' | 'cpu';

export interface PredictParams {
    diffusion_samples: number;
    recycling_steps: number;
    sampling_steps: number;
    use_potentials: boolean;
    seed: number | null;
    /** per-job device override (retry on CPU after running out of memory) */
    accelerator?: Accelerator;
}

export interface Workbench {
    name: string;
    components: WBComponent[];
    affinityBinderUid: string | null;
    params: PredictParams;
    parentJobId: string | null;
}

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';
export type JobKind = 'predict' | 'scan' | 'refine';

export interface JobLive {
    phase?: string;
    label?: string;
    progress?: number | null;
    started_at?: number;
    updated_at?: number;
    /** what this prediction was expected to take, frozen when it started (null = not computable) */
    estimate_sec?: number | null;
    memory?: LiveMemory;
}

/** Sampled every ~3 s by the supervisor while a prediction runs. Not persisted. */
export interface LiveMemory {
    ts: number;
    /** physical footprint of the whole Boltz process group right now — not resident size */
    footprint_gb: number;
    peak_gb: number;
    total_gb: number | null;
    swap_used_gb: number | null;
    free_disk_gb: number | null;
    /** CPU seconds the Boltz process group has consumed so far */
    cpu_sec?: number;
    /** fraction of that which is user time; a thrashing run is almost all system time */
    user_share?: number | null;
    /** CPU seconds per wall second since the previous sample — the run's real speed */
    efficiency?: number | null;
}

export interface Confidence {
    confidence_score?: number;
    ptm?: number;
    iptm?: number;
    ligand_iptm?: number;
    protein_iptm?: number;
    complex_plddt?: number;
    complex_iplddt?: number;
    complex_pae?: number | null;
    complex_ipae?: number | null;
    chains_ptm?: Record<string, number>;
    pair_chains_iptm?: Record<string, Record<string, number>>;
}

export interface ModelResult {
    index: number;
    file: string;
    confidence: Confidence;
    plddt: Record<string, number[]>;
    ligand_plddt: Record<string, number>;
}

export interface ChainRow {
    chain: string;
    type: ComponentType;
    label: string;
    length: number | null;
    sequence?: string | null;
    smiles?: string | null;
    ccd?: string | null;
    mutations: string[];
}

export interface Affinity {
    affinity_pred_value: number;
    affinity_probability_binary: number;
    binder?: string;
    ic50_um?: number | null;
    delta_g_kcal?: number | null;
}

export interface PaeSegment {
    chain: string;
    kind: 'polymer' | 'ligand';
    start: number;
    end: number;
}

export interface Interface {
    chains: [string, string];
    residues: Record<string, string[]>;
    contact_pairs: number;
}

export interface PredictResult {
    models: ModelResult[];
    chains: ChainRow[];
    affinity: Affinity | null;
    pae: { size: number; factor: number; matrix: number[][]; segments: PaeSegment[] | null } | null;
    interfaces: { cutoff: number; interfaces: Interface[] } | { error: string } | null;
    /** whether the coordinates are physically possible; null on results predicted before 0.2.1 */
    geometry?: Geometry | { error: string } | null;
    elapsed_sec: number;
    /** seconds spent per Boltz phase (0.2+) */
    timings?: Record<string, number>;
    token_estimate?: number;
    accelerator: string;
    msa: { server: boolean; reused_from: string | null; single_sequence_chains: string[] };
    normalized_spec: SpecOut;
}

export interface Geometry {
    model_index: number;
    atoms: number;
    /** atoms whose coordinates or pLDDT came back NaN/inf */
    nonfinite_atoms: number;
    nonfinite_examples: string[];
    other_models_nonfinite: { model_index: number; nonfinite_atoms: number }[];
    /** heavy-atom pairs overlapping beyond the van der Waals tolerance; null when NaN stopped the check */
    clashes: number | null;
    severe_clashes: number | null;
    /** clashes per 1000 atoms */
    clashscore: number | null;
    worst_clashes: { a: string; b: string; dist: number; overlap: number }[];
    chain_breaks: { chain: string; after: string; before: string; dist: number; limit: number }[];
    vdw_tolerance: number;
}

export interface ScanResult {
    model: string;
    alphabet: string;
    sequence: string;
    matrix: number[][];
    pseudo_perplexity: number;
    top_substitutions: { mutation: string; llr: number }[];
    position_tolerance: number[];
    elapsed_sec: number;
    chain: string | null;
    label: string | null;
}

export interface RefineResult {
    sequence: string;
    pseudo_perplexity: number;
    start_pseudo_perplexity: number;
    label: string | null;
    history: { round: number; sequence: string; pseudo_perplexity: number; changed: string[]; accepted?: boolean }[];
}

export interface SpecComponentOut {
    type: ComponentType;
    chains: string[];
    label: string;
    sequence?: string;
    smiles?: string;
    ccd?: string;
    msa?: 'server' | 'single';
    cyclic?: boolean;
    source?: ComponentSource;
    mutations?: string[];
    parent_sequence?: string;
}

export interface SpecOut {
    name?: string;
    components: SpecComponentOut[];
    affinity_binder?: string | null;
    params?: Partial<PredictParams>;
    reuse_msa?: boolean;
    workbench_name?: string;
}

export interface JobBase {
    id: string;
    kind: JobKind;
    status: JobStatus;
    title: string;
    created_at: number;
    started_at: number | null;
    finished_at: number | null;
    error: string | null;
    /** failure class: oom | msa | download | input | network | missing_dependency | error */
    error_kind?: string | null;
    phase: string | null;
    parent_id: string | null;
    origin: string;
    starred: boolean;
    live: JobLive;
    queue_position?: number;
    spec: SpecOut & Record<string, unknown>;
}

export interface JobSummaryResult {
    confidence?: Confidence;
    affinity?: Affinity | null;
    elapsed_sec?: number;
    model_count?: number;
    mean_plddt?: number | null;
    pseudo_perplexity?: number;
    start_pseudo_perplexity?: number;
    chain?: string | null;
}

export interface JobSummary extends JobBase {
    result: JobSummaryResult | null;
}

export interface PredictJob extends JobBase {
    kind: 'predict';
    result: PredictResult | null;
}
export interface ScanJob extends JobBase {
    kind: 'scan';
    result: ScanResult | null;
}
export interface RefineJob extends JobBase {
    kind: 'refine';
    result: RefineResult | null;
}
export type FullJob = PredictJob | ScanJob | RefineJob;

export interface ImportedStructure {
    name: string;
    note?: string | null;
    bytes?: number;
    url: string;
    format: 'mmcif' | 'pdb';
    title: string;
    source: ComponentSource;
    plddt_in_bfactor: boolean;
    summary: {
        title: string;
        chains: { chain: string; kind: PolymerType; sequence: string; entity: string; observed_residues: number }[];
        ligands: { chain: string; ccd: string; seqid: string; atoms: number; additive: boolean }[];
        warnings: string[];
    };
}

export interface CompareResult {
    rmsd: number;
    matched_ca: number;
    deviations: { fixed_chain: string; fixed_res: number; moving_chain: string; moving_res: number; distance: number }[];
    url: string;
    output: string;
}

export interface ChemInfo {
    smiles: string;
    canonical_smiles: string;
    formula: string;
    molecular_weight: number;
    heavy_atoms: number;
    fragments: number;
    hbd: number;
    hba: number;
    logp: number;
    rotatable_bonds: number;
    affinity_ok: boolean;
    svg?: string;
}

export type ProposalType = 'mutation_set' | 'add_ligand' | 'add_protein' | 'add_nucleic' | 'new_protein';

export interface ProposalApply {
    action: 'mutate' | 'add_component' | 'new_protein';
    chain?: string;
    mutations?: string[];
    component?: {
        type: ComponentType;
        label: string;
        sequence?: string;
        smiles?: string;
        ccd?: string;
        msa?: 'server' | 'single';
        source?: ComponentSource;
    };
}

export interface Proposal {
    type: ProposalType;
    title: string;
    rationale: string;
    issues: string[];
    status: 'ok' | 'warning' | 'invalid';
    apply: ProposalApply | null;
    chain?: string;
    mutations?: string[];
    rejected?: string[];
    /** mutations whose position was moved onto the residue the model actually named */
    repaired?: string[];
    esm?: { total_llr: number; per_mutation: { mutation: string; llr: number }[] } | null;
    chem?: ChemInfo;
    resolved?: Record<string, string | number | null | undefined>;
    stats?: { length: number; hydrophobic_fraction: number; most_common: string; pseudo_perplexity?: number };
    sequence?: string;
}

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    mode: string;
    created_at: number;
    proposals?: Proposal[];
    reply_issues?: string[];
    corrected?: boolean;
    elapsed_sec?: number;
    model?: string;
}

export interface Thread {
    id: string;
    title: string;
    created_at: number;
    updated_at: number;
    messages: ChatMessage[];
}

export interface Settings {
    ollama_url: string;
    llm_model: string;
    llm_model_heavy: string;
    llm_log_limit: number;
    llm_think: boolean;
    llm_temperature: number;
    boltz_bin: string;
    boltz_cache: string;
    accelerator: string;
    mps_strict: boolean;
    mps_memory_ratio: number;
    applecare: boolean;
    mpnn_enabled: boolean;
    mpnn_veto: number;
    msa_server_url: string;
    diffusion_samples: number;
    recycling_steps: number;
    sampling_steps: number;
    use_potentials: boolean;
    reuse_msa_for_variants: boolean;
    cleanup_intermediate: boolean;
    esm_model: string;
    esm_device: string;
    notify_on_finish: boolean;
    autopilot_enabled: boolean;
    autopilot_max_variants_per_job: number;
    autopilot_max_depth: number;
    autopilot_strategy: string;
    autopilot_selection: string;
    autopilot_protect_interfaces: boolean;
    autopilot_experiment: string;
    autopilot_climb_patience: number;
    autopilot_max_queued: number;
    autopilot_daily_budget: number;
    autopilot_min_disk_gb: number;
    autopilot_min_esm_llr: number;
    autopilot_explain: boolean;
    autopilot_proposals_per_call: number;
    autopilot_protected_residues: string;
    autopilot_protect_disordered: boolean;
    autopilot_disorder_plddt: number;
    autopilot_notify_improvement: boolean;
    autopilot_improvement_metric: string;
    autopilot_improvement_delta: number;
    job_auto_retry: boolean;
    job_max_retries: number;
}

export interface LlmStatus {
    server: boolean;
    model: string;
    model_available: boolean;
    heavy_model?: string;
    heavy_available?: boolean;
    models: { name: string; size: number; parameters?: string; quantization?: string }[];
    pull?: { active: boolean; model?: string; status?: string; completed?: number; total?: number; error?: string | null };
    /** where the Ollama binary in use came from, and how fetching one is going */
    binary?: string | null;
    source?: 'bundled' | 'downloaded' | 'system' | 'none';
    download_mb?: number;
    install?: { active: boolean; status?: string; completed?: number; total?: number; error?: string | null; path?: string | null };
}

export interface Health {
    version: string;
    home: string;
    boltz: { bin: string | null; version: string | null; weights: boolean; affinity_weights: boolean; ccd: boolean };
    torch: string | null;
    torch_probed: boolean;
    mps: boolean;
    llm: LlmStatus;
    esm: { loaded: boolean; model: string; device: string | null; cached: boolean };
    machine: { os: string; chip: string; memory_gb: number | null; python: string };
    disk_free_gb: number;
    exports_dir: string;
}

export interface Estimate {
    seconds: number;
    low: number;
    high: number;
    breakdown: { startup: number; msa: number; structure: number; affinity: number };
    basis: 'history' | 'default';
    samples: number;
    tokens: number;
    needs_msa_search: boolean;
    msa_reuse: boolean;
    queued_ahead: number;
    memory: {
        peak_gb: number; total_gb: number | null; level: 'ok' | 'caution' | 'danger';
        basis: 'history' | 'default'; samples: number;
        /** the estimate is above installed memory, so the run swaps from end to end */
        beyond_physical: boolean;
        /** whether the user told us the machine is covered; only changes the wording */
        applecare: boolean;
    };
}

/** NVMe SMART health for the boot drive. Null everywhere when smartmontools is not installed. */
export interface SsdWear {
    model: string | null;
    percentage_used: number;
    written_tb: number;
    read_tb: number;
    available_spare: number | null;
    available_spare_threshold: number | null;
    media_errors: number | null;
    power_on_hours: number | null;
    healthy: boolean;
    /** this drive's own TB-per-percent. Null until it has worn 1%. */
    tb_per_percent: number | null;
    /** 'delta' = measured across a real move in the counter. 'single' = one point, provisional. */
    tb_per_percent_basis: 'delta' | 'single' | 'none';
    tb_per_percent_span: number;
    readings: number;
}

/** What swapping costs the SSD, per day of running. A rate, not a total — see ssd.py. */
export interface SwapWriteForecast {
    mb_per_sec: number;
    tb_per_day: number;
    life_percent_per_day: number | null;
    life_basis: 'delta' | 'single' | 'none';
    /** what the user told us; the app cannot look it up */
    applecare: boolean;
    wear: SsdWear | null;
}

export interface SsdInfo extends SwapWriteForecast {
    available: boolean;
    smartctl: string | null;
}

export interface StorageInfo {
    home: string;
    jobs_bytes: number;
    imports_bytes: number;
    msa_cache_bytes: number;
    boltz_cache_bytes: number;
    disk_free_gb: number;
}

export interface UniProtHit {
    accession: string;
    entry_name: string;
    name: string;
    organism: string;
    length: number;
    genes: string[];
    reviewed: boolean;
}

export interface LibraryItem {
    id: string;
    type: ComponentType | 'workbench';
    name: string;
    data: Record<string, unknown>;
    created_at: number;
}

/** What the 3D viewer is showing */
export type ViewSource =
    | { kind: 'job'; jobId: string; model: number }
    | { kind: 'import'; item: ImportedStructure }
    | { kind: 'compare'; fixed: { jobId: string; model: number; title: string }; moving: { jobId: string; model: number; title: string }; result: CompareResult };

export interface SearchRecord {
    id: string;
    created_at: number;
    source: string;
    query: string;
    hits: number;
    top: { id: string; title?: string | null }[];
    picked: { id: string; title: string; at: number } | null;
}

export interface LlmCall {
    id: string;
    created_at: number;
    model: string;
    mode: string;
    origin: 'user' | 'autopilot';
    job_id: string | null;
    thread_id: string | null;
    reply: string;
    proposals: Proposal[];
    reply_issues: string[];
    corrected: boolean;
    elapsed_sec: number;
    error: string | null;
    messages?: { role: string; content: string }[];
}


/** What Metal will let one process hold, and the sysctl behind it (`iogpu.wired_limit_mb`). */
export interface GpuState {
    total_gb: number;
    min_mb: number;
    max_mb: number;
    headroom_gb: number;
    wired_limit_mb: number;
    metal_limit_gb: number | null;
    is_default: boolean;
    changed_to_mb?: number;
    restart_required?: boolean;
    persists_across_reboot?: boolean;
}
