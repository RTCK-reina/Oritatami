import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { api, connection, downloadFile, errorMessage } from './api';
import type { Example } from './examples';
import type {
    Accelerator,
    FullJob,
    Health,
    ImportedStructure,
    JobSummary,
    Proposal,
    SpecOut,
    ViewSource,
    WBComponent,
    Workbench,
} from './types';
import { applyMutationCodes, assignChains, buildSpec, emptyWorkbench, newUid, specToWorkbench } from './workbench';

export interface ToastAction {
    label: string;
    run: () => void;
}

export interface Toast {
    id: number;
    kind: 'info' | 'success' | 'error';
    text: string;
    action?: ToastAction;
}

const WB_KEY = 'oritatami.workbench.v1';

/** Whether two prediction inputs are the same molecules with the same settings. */
function sameInput(a: SpecOut, b: SpecOut): boolean {
    const comps = (s: SpecOut) => JSON.stringify(s.components.map(c => [c.type, c.sequence ?? '', c.smiles ?? '', c.ccd ?? '',
        c.chains.length, c.msa ?? '', !!c.cyclic]));
    const params = (s: SpecOut) => {
        const p = s.params ?? {};
        return JSON.stringify([p.diffusion_samples ?? 1, p.recycling_steps ?? 3, p.sampling_steps ?? 200, !!p.use_potentials,
            p.seed ?? null, p.accelerator ?? 'auto']);
    };
    return comps(a) === comps(b) && params(a) === params(b) && (a.affinity_binder ?? null) === (b.affinity_binder ?? null);
}
const TERMINAL = ['succeeded', 'failed', 'cancelled'];

function loadWorkbench(): Workbench {
    try {
        const raw = localStorage.getItem(WB_KEY);
        if (raw) {
            const wb = JSON.parse(raw) as Workbench;
            if (Array.isArray(wb.components)) return { ...emptyWorkbench(), ...wb, params: { ...emptyWorkbench().params, ...wb.params } };
        }
    } catch {
        // storage unavailable or corrupt: start empty (the workbench is a convenience cache)
    }
    return emptyWorkbench();
}

export interface AssistantRequest {
    nonce: number;
    mode: 'chat' | 'mutations' | 'complex' | 'design' | 'explain';
    message: string;
    jobId: string | null;
}

export interface Store {
    online: boolean;
    health: Health | null;
    refreshHealth: () => Promise<void>;
    workbench: Workbench;
    setWorkbench: (fn: (wb: Workbench) => Workbench) => void;
    addComponent: (c: Omit<WBComponent, 'uid' | 'copies'> & { copies?: number }) => void;
    updateComponent: (uid: string, patch: Partial<WBComponent>) => void;
    removeComponent: (uid: string) => void;
    jobs: JobSummary[];
    jobsLoaded: boolean;
    refreshJobs: () => Promise<void>;
    getJob: (id: string, force?: boolean) => Promise<FullJob>;
    jobCache: Record<string, FullJob>;
    selectedJobId: string | null;
    selectJob: (id: string | null) => void;
    /** select a job and, for a finished prediction, show it in 3D */
    openJob: (id: string) => void;
    view: ViewSource | null;
    setView: (v: ViewSource | null) => void;
    imports: ImportedStructure[];
    addImport: (i: ImportedStructure) => void;
    runPrediction: (opts?: { title?: string; origin?: string; workbench?: Workbench; force?: boolean }) => Promise<JobSummary | null>;
    runVariants: (variants: { chain: string; mutations: string[] }[], opts?: { origin?: string; workbench?: Workbench }) => Promise<void>;
    retryJob: (id: string, overrides?: { msa?: 'single'; accelerator?: Accelerator; diffusion_samples?: number; new_seed?: boolean }) => Promise<void>;
    exportJob: (id: string, how: 'download' | 'folder' | 'pdb', model?: number) => Promise<void>;
    scanComponent: (uid: string, wb?: Workbench) => Promise<void>;
    scanJobFor: (sequence: string | undefined) => JobSummary | undefined;
    loadJobIntoWorkbench: (jobId: string) => Promise<void>;
    applyProposal: (p: Proposal, run: boolean) => Promise<void>;
    compare: (fixedJobId: string, movingJobId: string) => Promise<void>;
    openExample: (ex: Example) => Promise<void>;
    toasts: Toast[];
    toast: (kind: Toast['kind'], text: string, action?: ToastAction) => void;
    dismissToast: (id: number) => void;
    threadId: string | null;
    setThreadId: (id: string | null) => void;
    selectedResidue: { chain: string; position: number } | null;
    setSelectedResidue: (r: { chain: string; position: number } | null) => void;
    assistantRequest: AssistantRequest | null;
    requestAssistant: (mode: AssistantRequest['mode'], message?: string, jobId?: string | null) => void;
    /** the assistant calls this once it has taken the request, so a remount does not send it again */
    clearAssistantRequest: (nonce: number) => void;
    /** incremented to ask the left column to show a tab */
    leftTabRequest: { tab: 'workbench' | 'jobs' | 'board'; nonce: number } | null;
    showLeftTab: (tab: 'workbench' | 'jobs' | 'board') => void;
}

const Ctx = createContext<Store | null>(null);

export function useStore(): Store {
    const s = useContext(Ctx);
    if (!s) throw new Error('StoreProvider missing');
    return s;
}

export function StoreProvider({ children }: { children: ReactNode }) {
    const [online, setOnline] = useState(connection.online);
    const [health, setHealth] = useState<Health | null>(null);
    const [workbench, setWorkbenchState] = useState<Workbench>(loadWorkbench);
    const [jobs, setJobs] = useState<JobSummary[]>([]);
    const [jobsLoaded, setJobsLoaded] = useState(false);
    const [jobCache, setJobCache] = useState<Record<string, FullJob>>({});
    const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
    const [view, setView] = useState<ViewSource | null>(null);
    const [imports, setImports] = useState<ImportedStructure[]>([]);
    const [toasts, setToasts] = useState<Toast[]>([]);
    // the open LLM conversation survives a reload / restart
    const [threadId, setThreadIdState] = useState<string | null>(() => {
        try {
            return localStorage.getItem('oritatami.thread');
        } catch {
            return null;
        }
    });
    const setThreadId = useCallback((id: string | null) => {
        setThreadIdState(id);
        try {
            if (id) localStorage.setItem('oritatami.thread', id);
            else localStorage.removeItem('oritatami.thread');
        } catch {
            // storage unavailable: the conversation just is not restored next time
        }
    }, []);
    const [selectedResidue, setSelectedResidue] = useState<{ chain: string; position: number } | null>(null);
    const [assistantRequest, setAssistantRequest] = useState<AssistantRequest | null>(null);
    const [leftTabRequest, setLeftTabRequest] = useState<Store['leftTabRequest']>(null);
    const watched = useRef<Set<string>>(new Set());
    const toastId = useRef(0);
    const prevStatus = useRef<Record<string, string>>({});
    const jobCacheRef = useRef(jobCache);
    jobCacheRef.current = jobCache;

    useEffect(() => connection.subscribe(setOnline), []);

    const toast = useCallback((kind: Toast['kind'], text: string, action?: ToastAction) => {
        toastId.current += 1;
        const id = toastId.current;
        setToasts(t => {
            // the same message repeating (e.g. a flaky network) replaces itself instead of stacking
            const rest = t.filter(x => x.text !== text);
            return [...rest.slice(-3), { id, kind, text, action }];
        });
        window.setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), kind === 'error' ? 12000 : action ? 9000 : 5000);
    }, []);
    const dismissToast = useCallback((id: number) => setToasts(t => t.filter(x => x.id !== id)), []);

    const setWorkbench = useCallback((fn: (wb: Workbench) => Workbench) => {
        setWorkbenchState(prev => {
            const next = fn(prev);
            try {
                localStorage.setItem(WB_KEY, JSON.stringify(next));
            } catch {
                // storage may be unavailable (private mode); the in-memory state still works
            }
            return next;
        });
    }, []);

    const addComponent: Store['addComponent'] = useCallback(c => {
        setWorkbench(wb => ({
            ...wb,
            components: [...wb.components, { copies: 1, ...c, uid: newUid(), baseSequence: c.baseSequence ?? c.sequence }],
        }));
    }, [setWorkbench]);

    const updateComponent = useCallback((uid: string, patch: Partial<WBComponent>) => {
        setWorkbench(wb => ({ ...wb, components: wb.components.map(c => (c.uid === uid ? { ...c, ...patch } : c)) }));
    }, [setWorkbench]);

    const removeComponent = useCallback((uid: string) => {
        setWorkbench(wb => ({
            ...wb,
            components: wb.components.filter(c => c.uid !== uid),
            affinityBinderUid: wb.affinityBinderUid === uid ? null : wb.affinityBinderUid,
        }));
    }, [setWorkbench]);

    const refreshHealth = useCallback(async () => {
        try {
            setHealth(await api.health());
        } catch {
            // connection problems surface through the offline banner; nothing else to report here
        }
    }, []);

    const getJob = useCallback(async (id: string, force = false) => {
        const cached = jobCacheRef.current[id];
        if (!force && cached && TERMINAL.includes(cached.status)) return cached;
        const job = await api.job(id);
        setJobCache(c => ({ ...c, [id]: job }));
        return job;
    }, []);

    const showLeftTab = useCallback((tab: 'workbench' | 'jobs' | 'board') => setLeftTabRequest({ tab, nonce: Date.now() }), []);

    const jobsRef = useRef(jobs);
    jobsRef.current = jobs;
    const openJob = useCallback((id: string) => {
        setSelectedJobId(id);
        const summary = jobsRef.current.find(j => j.id === id);
        if (summary?.kind === 'predict' && summary.status === 'succeeded') {
            setView(v => (v?.kind === 'job' && v.jobId === id ? v : { kind: 'job', jobId: id, model: 0 }));
        }
    }, []);

    const refreshJobs = useCallback(async () => {
        let list: JobSummary[];
        try {
            list = await api.jobs();
        } catch {
            return; // offline banner covers it; the sync loop retries
        }
        setJobs(list);
        setJobsLoaded(true);
        const firstLoad = Object.keys(prevStatus.current).length === 0;
        for (const j of list) {
            const before = prevStatus.current[j.id];
            prevStatus.current[j.id] = j.status;
            if (firstLoad || !before || before === j.status || !TERMINAL.includes(j.status)) continue;
            try {
                const full = await api.job(j.id);
                setJobCache(c => ({ ...c, [j.id]: full }));
            } catch {
                // the result view fetches it again on demand
            }
            const show = () => {
                setSelectedJobId(j.id);
                if (j.kind === 'predict') setView({ kind: 'job', jobId: j.id, model: 0 });
            };
            if (j.status === 'succeeded') toast('success', `完了: ${j.title}`, { label: '結果を見る', run: show });
            if (j.status === 'failed') toast('error', `失敗: ${j.title} — ${(j.error ?? '').split('\n')[0]}`, { label: '詳細', run: () => setSelectedJobId(j.id) });
            if (j.status !== 'cancelled' && !document.hasFocus()) {
                void api.notify(j.status === 'succeeded' ? '計算が終わりました' : '計算が失敗しました', j.title).catch(() => undefined);
            }
            if (watched.current.has(j.id)) {
                watched.current.delete(j.id);
                if (j.status === 'succeeded') show();
            }
        }
    }, [toast]);

    // Job sync: long-poll the server's change counter so updates arrive immediately without busy polling.
    useEffect(() => {
        let alive = true;
        const ctrl = new AbortController();
        (async () => {
            let rev = -1;
            let backoff = 500;
            while (alive) {
                try {
                    const r = await api.jobChanges(rev, rev < 0 ? 0 : 25, ctrl.signal);
                    backoff = 500;
                    if (r.rev !== rev) {
                        rev = r.rev;
                        await refreshJobs();
                        // coalesce bursts (e.g. ESM progress) into at most ~3 refreshes per second
                        await new Promise(res => window.setTimeout(res, 300));
                    }
                } catch (e) {
                    if (!alive || (e instanceof DOMException && e.name === 'AbortError')) return;
                    await new Promise(res => window.setTimeout(res, backoff));
                    backoff = Math.min(backoff * 2, 5000);
                    rev = -1; // the server may have restarted with a new counter
                }
            }
        })();
        return () => {
            alive = false;
            ctrl.abort();
        };
    }, [refreshJobs]);

    useEffect(() => {
        void refreshHealth();
        const t = window.setInterval(() => void refreshHealth(), 30000);
        return () => window.clearInterval(t);
    }, [refreshHealth]);
    // pick up a restarted backend quickly
    useEffect(() => {
        if (online) {
            void refreshHealth();
            return;
        }
        const t = window.setInterval(() => void api.ping().catch(() => undefined), 2000);
        return () => window.clearInterval(t);
    }, [online, refreshHealth]);

    const runPrediction: Store['runPrediction'] = useCallback(async (opts = {}) => {
        const wb = opts.workbench ?? workbench;
        if (!wb.components.length) {
            toast('error', '作業台が空です。「＋ タンパク質」などで分子を追加してください');
            return null;
        }
        const spec = buildSpec(wb);
        if (!opts.force && wb.parentJobId) {
            // e.g. ⌘↩ pressed again on an unchanged workbench: ask instead of spending minutes on a duplicate
            const parent = jobCacheRef.current[wb.parentJobId] ?? await api.job(wb.parentJobId).catch(() => null);
            if (parent?.kind === 'predict' && ['queued', 'running', 'succeeded'].includes(parent.status) && sameInput(parent.spec, spec)) {
                toast('info', `同じ入力の予測がすでにあります (${parent.title})。別のサンプルを得たいときは、そのまま予測できます`,
                    { label: 'それでも予測する', run: () => void runPredictionRef.current({ ...opts, force: true }) });
                return null;
            }
        }
        try {
            const job = await api.submitPredict(spec, opts.title ?? wb.name, wb.parentJobId, opts.origin ?? 'user');
            // The first prediction of a fresh workbench becomes its reference result, so later variants
            // (mutations, batches) are grouped under it and can be compared against it.
            setWorkbench(cur => (cur.parentJobId || cur.components !== wb.components ? cur : { ...cur, parentJobId: job.id }));
            watched.current.add(job.id);
            prevStatus.current[job.id] = job.status;
            setSelectedJobId(job.id);
            toast('info', `予測をキューに追加: ${job.title}`);
            await refreshJobs();
            return job;
        } catch (e) {
            toast('error', errorMessage(e));
            return null;
        }
    }, [workbench, toast, refreshJobs, setWorkbench]);

    const runPredictionRef = useRef(runPrediction);
    runPredictionRef.current = runPrediction;

    const runVariants: Store['runVariants'] = useCallback(async (variants, opts = {}) => {
        const wb = opts.workbench ?? workbench;
        if (!variants.length) return;
        try {
            const res = await api.submitBatch(buildSpec(wb), variants, wb.parentJobId, opts.origin ?? 'user');
            res.jobs.forEach(j => { prevStatus.current[j.id] = j.status; });
            toast('info', `${res.jobs.length} 個の変異体をキューに追加しました`, { label: 'ジョブを見る', run: () => showLeftTab('jobs') });
            await refreshJobs();
            showLeftTab('jobs');
        } catch (e) {
            toast('error', errorMessage(e));
        }
    }, [workbench, toast, refreshJobs, showLeftTab]);

    const retryJob: Store['retryJob'] = useCallback(async (id, overrides = {}) => {
        try {
            const job = await api.retryJob(id, overrides);
            watched.current.add(job.id);
            prevStatus.current[job.id] = job.status;
            setSelectedJobId(job.id);
            toast('info', `再実行をキューに追加: ${job.title}`);
            await refreshJobs();
        } catch (e) {
            toast('error', errorMessage(e));
        }
    }, [toast, refreshJobs]);

    const exportJob: Store['exportJob'] = useCallback(async (id, how, model = 0) => {
        try {
            if (how === 'download') {
                downloadFile(api.exportZipUrl(id));
            } else if (how === 'pdb') {
                downloadFile(api.structurePdbUrl(id, model));
            } else {
                const r = await api.exportJobToFolder(id);
                toast('success', `書き出しました: ${r.path}`);
            }
        } catch (e) {
            toast('error', errorMessage(e));
        }
    }, [toast]);

    const scanJobFor = useCallback((sequence: string | undefined) => {
        if (!sequence) return undefined;
        return jobs.find(j => j.kind === 'scan' && j.spec.sequence === sequence && j.status !== 'failed' && j.status !== 'cancelled');
    }, [jobs]);

    const scanComponent = useCallback(async (uid: string, wbOverride?: Workbench) => {
        const wb = wbOverride ?? workbench;
        const comp = wb.components.find(c => c.uid === uid);
        if (!comp?.sequence || comp.type !== 'protein') return;
        const existing = scanJobFor(comp.sequence);
        if (existing) {
            setSelectedJobId(existing.id);
            return;
        }
        const chains = assignChains(wb.components).get(uid) ?? [];
        try {
            const job = await api.submitScan(comp.sequence, chains[0] ?? null, comp.label);
            watched.current.add(job.id);
            prevStatus.current[job.id] = job.status;
            setSelectedJobId(job.id);
            await refreshJobs();
        } catch (e) {
            toast('error', errorMessage(e));
        }
    }, [workbench, scanJobFor, refreshJobs, toast]);

    const loadJobIntoWorkbench = useCallback(async (jobId: string) => {
        try {
            const job = await getJob(jobId);
            if (job.kind !== 'predict') throw new Error('予測ジョブではありません');
            setWorkbench(() => specToWorkbench(job.spec, job.id, job.title));
            showLeftTab('workbench');
            toast('info', `作業台に読み込みました: ${job.title}`);
        } catch (e) {
            toast('error', errorMessage(e));
        }
    }, [getJob, setWorkbench, toast, showLeftTab]);

    const applyProposal = useCallback(async (p: Proposal, run: boolean) => {
        if (!p.apply) return;
        let next: Workbench = workbench;
        try {
            if (p.apply.action === 'mutate') {
                const chains = assignChains(workbench.components);
                const target = workbench.components.find(c => (chains.get(c.uid) ?? []).includes(p.apply?.chain ?? ''));
                if (!target?.sequence) throw new Error(`チェーン ${p.apply.chain} が作業台にありません`);
                const sequence = applyMutationCodes(target.sequence, p.apply.mutations ?? []);
                next = {
                    ...workbench,
                    name: `${workbench.name.replace(/ \+ .*$/, '')} + ${(p.apply.mutations ?? []).join('/')}`,
                    components: workbench.components.map(c => (c.uid === target.uid ? { ...c, sequence, origin: 'qwen' } : c)),
                };
            } else if (p.apply.action === 'add_component' && p.apply.component) {
                const c = p.apply.component;
                next = {
                    ...workbench,
                    components: [...workbench.components, {
                        uid: newUid(), copies: 1, type: c.type, label: c.label, sequence: c.sequence, baseSequence: c.sequence,
                        smiles: c.smiles, ccd: c.ccd, msa: c.msa, source: c.source, origin: 'qwen',
                    }],
                };
            } else if (p.apply.action === 'new_protein' && p.apply.component) {
                const c = p.apply.component;
                next = {
                    ...emptyWorkbench(),
                    name: p.title,
                    components: [{ uid: newUid(), copies: 1, type: 'protein', label: c.label, sequence: c.sequence,
                        baseSequence: c.sequence, msa: c.msa ?? 'single', origin: 'qwen' }],
                };
            }
        } catch (e) {
            toast('error', errorMessage(e));
            return;
        }
        setWorkbench(() => next);
        if (run) {
            await runPrediction({ workbench: next, title: next.name, origin: 'qwen' });
        } else {
            toast('success', `作業台に適用: ${p.title}`);
            showLeftTab('workbench');
        }
    }, [workbench, setWorkbench, runPrediction, toast, showLeftTab]);

    const compare = useCallback(async (fixedJobId: string, movingJobId: string) => {
        try {
            const [a, b] = await Promise.all([getJob(fixedJobId), getJob(movingJobId)]);
            const result = await api.compare({ job_id: fixedJobId, model: 0 }, { job_id: movingJobId, model: 0 });
            setView({
                kind: 'compare',
                fixed: { jobId: fixedJobId, model: 0, title: a.title },
                moving: { jobId: movingJobId, model: 0, title: b.title },
                result,
            });
            toast('info', `重ね合わせ RMSD ${result.rmsd.toFixed(2)} Å (Cα ${result.matched_ca} 個)`);
        } catch (e) {
            toast('error', errorMessage(e));
        }
    }, [getJob, toast]);

    const requestAssistant = useCallback((mode: AssistantRequest['mode'], message = '', jobId: string | null = null) => {
        setAssistantRequest({ nonce: Date.now(), mode, message, jobId });
    }, []);

    const clearAssistantRequest = useCallback((nonce: number) => {
        setAssistantRequest(r => (r?.nonce === nonce ? null : r));
    }, []);

    const addImport = useCallback((i: ImportedStructure) => setImports(list => [i, ...list.filter(x => x.name !== i.name)]), []);

    const openExample = useCallback(async (ex: Example) => {
        const a = ex.action;
        try {
            if (a.kind === 'afdb' || a.kind === 'pdb') {
                const item = a.kind === 'afdb' ? await api.importAfdb(a.accession) : await api.importPdb(a.id);
                addImport(item);
                setView({ kind: 'import', item });
                toast('success', `${item.title} を表示しました`);
                return;
            }
            if (a.kind === 'assistant') {
                requestAssistant(a.mode, a.message, null);
                return;
            }
            const wb = await a.build();
            setWorkbench(() => wb);
            showLeftTab('workbench');
            if (a.then === 'predict') {
                await runPrediction({ workbench: wb, title: wb.name });
            } else if (a.then === 'scan') {
                await scanComponent(wb.components[0].uid, wb);
                toast('info', 'ESM-2 で変異スキャンを開始しました。結果のヒートマップをクリックすると変異を入れられます');
            }
        } catch (e) {
            toast('error', `例を開けませんでした: ${errorMessage(e)}`);
        }
    }, [addImport, requestAssistant, setWorkbench, showLeftTab, runPrediction, scanComponent, toast]);

    const value = useMemo<Store>(() => ({
        online, health, refreshHealth, workbench, setWorkbench, addComponent, updateComponent, removeComponent,
        jobs, jobsLoaded, refreshJobs, getJob, jobCache, selectedJobId, selectJob: setSelectedJobId, openJob, view, setView,
        imports, addImport, runPrediction, runVariants, retryJob, exportJob, scanComponent, scanJobFor, loadJobIntoWorkbench,
        applyProposal, compare, openExample, toasts, toast, dismissToast, threadId, setThreadId, selectedResidue,
        setSelectedResidue, assistantRequest, requestAssistant, clearAssistantRequest, leftTabRequest, showLeftTab,
    }), [online, health, refreshHealth, workbench, setWorkbench, addComponent, updateComponent, removeComponent, jobs,
        jobsLoaded, refreshJobs, getJob, jobCache, selectedJobId, openJob, view, imports, addImport, runPrediction,
        runVariants, retryJob, exportJob, scanComponent, scanJobFor, loadJobIntoWorkbench, applyProposal, compare,
        openExample, toasts, toast, dismissToast, threadId, setThreadId, selectedResidue, assistantRequest, requestAssistant,
        clearAssistantRequest, leftTabRequest, showLeftTab]);

    return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
