import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, errorMessage, type AutopilotStatus, type QueueEta as QueueEtaData } from './api';
import { AssistantPanel } from './components/AssistantPanel';
import { AutopilotDialog } from './components/AutopilotDialog';
import { CommandPalette, type Command } from './components/CommandPalette';
import { ErrorBoundary } from './components/ErrorBoundary';
import { HelpDialog } from './components/HelpDialog';
import { JobsPanel } from './components/JobsPanel';
import { LeaderboardPanel } from './components/LeaderboardPanel';
import { ResultsPanel } from './components/ResultsPanel';
import { SettingsDialog } from './components/SettingsDialog';
import { Splitter } from './components/Splitter';
import { Icon, Kbd, Spinner, Tabs } from './components/ui';
import { ViewerPanel } from './components/ViewerPanel';
import { WorkbenchPanel } from './components/WorkbenchPanel';
import { EXAMPLES } from './examples';
import { useHotkeys, usePersistentState, useTheme, useWindowWidth, type ThemePref } from './hooks';
import { StoreProvider, useStore } from './store';
import { uiEvents } from './uiEvents';
import { t } from './i18n';

const DEFAULTS = { left: 380, right: 400, results: 320 };
/** The 3D viewer stops being usable below this; side panels fold away rather than crush it. */
const MIN_CENTER = 520;
const RAIL = 28;

/** The loop runs unattended for hours and its switch lived four clicks deep in the settings
 *  dialog, so whether it was on had to be inferred from the job list. */
function AutopilotSwitch() {
    const { toast } = useStore();
    const [st, setSt] = useState<AutopilotStatus | null>(null);
    const [busy, setBusy] = useState(false);

    const load = useCallback(() => {
        api.autopilotStatus().then(setSt).catch(() => setSt(null));
    }, []);
    useEffect(() => {
        load();
        const t = window.setInterval(load, 10000);
        const off = uiEvents.on('autopilotChanged', load);
        return () => { window.clearInterval(t); off(); };
    }, [load]);

    const flip = useCallback(async (next: boolean) => {
        setBusy(true);
        try {
            await api.updateSettings({ autopilot_enabled: next });
            toast('info', next ? t('自律ループを動かします') : t('自律ループを止めました (実行中のジョブは最後まで走ります)'));
            load();
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setBusy(false);
        }
    }, [load, toast]);
    const enabled = st?.enabled ?? false;
    useEffect(() => uiEvents.on('toggleAutopilot', () => void flip(!enabled)), [flip, enabled]);

    if (!st) return null;
    const on = st.enabled;
    const blocked = on && !st.accepting;
    const title = [
        on ? t('自律ループ: 動作中') : t('自律ループ: 停止中'),
        st.experiment ? `${t('実験名')} ${st.experiment}` : null,
        `${st.strategy === 'climb' ? t('山登り') : t('ランダムウォーク')} ${t('· 待機')} ${st.queued}/${st.max_queued} ${t('· 24時間で')} ${st.used_today} ${t('件')}`,
        st.blocked_reason,
    ].filter(Boolean).join('\n');

    return (
        <button type="button" className={`pill autopilot-pill ${on ? (blocked ? 'warn' : 'ok') : ''}`}
            aria-pressed={on} disabled={busy} title={`${title}${t('\nクリックで自律ループの設定 (起点・進め方・禁止リスト)')}`}
            onClick={() => uiEvents.emit('openAutopilot')}>
            {busy ? <Spinner size={9} /> : <span className={`dot ${on ? 'on' : ''}`} />}
            {t('自律')}{on && st.queued > 0 ? ` ${st.queued}` : ''}
        </button>
    );
}

/** Duration in the app's usual shape: '1 時間 20 分', '45 秒'. */
function human(seconds: number): string {
    if (seconds < 60) return `${Math.round(seconds)} ${t('秒')}`;
    const m = Math.round(seconds / 60);
    if (m < 60) return `${m} ${t('分')}`;
    const h = Math.floor(m / 60);
    return `${h} ${t('時間')}${m % 60 ? ` ${m % 60} ${t('分')}` : ''}`;
}

function clock(at: number): string {
    const d = new Date(at * 1000);
    const hm = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
    const today = new Date();
    if (d.toDateString() === today.toDateString()) return hm;
    return `${d.getMonth() + 1}/${d.getDate()} ${hm}`;
}

/**
 * How long the queue still has to run, and when it should be done.
 *
 * Per-job estimates already existed on the workbench, but the question while a run is going is
 * whether it is worth waiting for — which is a clock time, not a duration, and covers the whole
 * queue rather than the job being submitted.
 */
function QueueEta() {
    const { jobs, showLeftTab } = useStore();
    const [eta, setEta] = useState<QueueEtaData | null>(null);
    const active = jobs.filter(j => j.status === 'running' || j.status === 'queued').length;

    useEffect(() => {
        let alive = true;
        const load = () => api.queueEta().then(v => { if (alive) setEta(v); }).catch(() => { if (alive) setEta(null); });
        load();
        const t = window.setInterval(load, 20000);
        return () => { alive = false; window.clearInterval(t); };
    }, [active]);

    if (!eta || !eta.jobs) return null;
    // Two different statements, and the difference matters when you are deciding whether to
    // leave the machine: "done by 17:14" and "busy until at least 17:14, with one job nobody
    // can time" are not the same promise.
    const until = eta.complete ? eta.finish_at : eta.at_least_until;
    const clockText = until ? clock(until) : null;
    const running = eta.items.find(i => i.status === 'running');
    const stalled = running && running.basis === 'unknown';
    // Not the CPU efficiency: measured here it falls with job size even when everything is
    // fine (0.55 at 76 tokens, 0.10 at 608) because the GPU does the work. The share of CPU
    // spent computing rather than moving pages is what separates the regimes — 0.64-0.86
    // healthy against 0.016 while paging.
    const slow = running?.regime === 'paging';
    const label = eta.complete
        ? `${t('残り')} ${human(eta.seconds)}`
        : eta.counted > 0 ? `${t('残り')} ${human(eta.seconds)} ${t('以上')}` : t('残り 不明');
    return (
        <button type="button" className={`pill pill-eta${stalled || slow ? ' pill-warn' : ''}`}
            onClick={() => showLeftTab('jobs')}
            title={[
                `${t('待機・実行中')} ${eta.jobs} ${t('件')}`,
                eta.counted ? `${t('見積もれた分の合計 約')} ${human(eta.seconds)}` : null,
                clockText ? (eta.complete ? `${t('終了見込み')} ${clockText}` : `${t('少なくとも')} ${clockText} ${t('までは塞がります')}`) : null,
                eta.unknown ? `${eta.unknown} ${t('件は残り時間を推定できません')}` : null,
                running?.note || null,
                typeof running?.progress === 'number' ? `${t('実行中のジョブ: 経過')} ${(running.progress * 100).toFixed(0)}%` : null,
                t('所要時間は過去の実績から。メモリに収まらない分はページング時間として上乗せしています'),
            ].filter(Boolean).join('\n')}>
            <Icon name="rotate" size={11} /> {label}{clockText ? ` · ${clockText}` : ''}
        </button>
    );
}

function TopBar({ themePref, onTheme }: { themePref: ThemePref; onTheme: (t: ThemePref) => void }) {
    const { health, jobs, openJob, showLeftTab } = useStore();
    const running = jobs.filter(j => j.status === 'running');
    const queued = jobs.filter(j => j.status === 'queued').length;
    const boltzOk = !!health?.boltz.bin;
    const llmOk = !!health?.llm.server && !!health.llm.model_available;
    const nextTheme: ThemePref = themePref === 'system' ? 'light' : themePref === 'light' ? 'dark' : 'system';
    return (
        <header className="topbar">
            <div className="brand">
                <svg viewBox="0 0 32 32" width="22" height="22" aria-hidden><path d="M4 23 Q9 5 16 16 T28 9" stroke="var(--accent)" strokeWidth="3.4" fill="none" strokeLinecap="round" /></svg>
                <span>Oritatami</span>
            </div>
            <button type="button" className="palette-trigger" onClick={() => uiEvents.emit('openPalette')} title={t('コマンド検索')}>
                <Icon name="search" size={13} /> <span>{t('やりたいことを検索…')}</span> <Kbd combo="mod+k" />
            </button>
            <span className="spacer" />
            {running.slice(0, 2).map(j => (
                <button type="button" key={j.id} className="pill pill-run" title={`${j.title}\n${j.live.label ?? ''}`} onClick={() => openJob(j.id)}>
                    <Spinner size={9} /> {j.title} · {typeof j.live.progress === 'number' ? `${Math.round(j.live.progress * 100)}%` : (j.live.label ?? j.phase)}
                </button>
            ))}
            {running.length > 2 && (
                <button type="button" className="pill pill-run" title={running.slice(2).map(j => j.title).join('\n')}
                    onClick={() => showLeftTab('jobs')}>{t('ほか')} {running.length - 2} {t('件')}</button>
            )}
            {queued > 0 && <span className="pill" title={t('待機中のジョブ')}>{t('待機')} {queued}</span>}
            <QueueEta />
            <AutopilotSwitch />
            <button type="button" className={`pill ${boltzOk ? 'ok' : 'bad'}`} onClick={() => uiEvents.emit('openSettings')}
                title={boltzOk ? `Boltz-2 ${health?.boltz.version ?? ''} (${health?.mps ? 'GPU' : 'CPU'})` : t('Boltz-2 が見つかりません — クリックで準備状況')}>
                Boltz-2 {health ? (health.mps ? 'GPU' : health.torch_probed ? 'CPU' : '…') : ''}
            </button>
            <button type="button" className={`pill ${health?.esm.loaded ? 'ok' : ''}`} onClick={() => uiEvents.emit('openSettings')} title={health?.esm.model}>ESM-2</button>
            <button type="button" className={`pill ${llmOk ? 'ok' : 'bad'}`} onClick={() => uiEvents.emit('openSettings')}
                title={health?.llm.server ? (health.llm.model_available ? health.llm.model : `${health.llm.model} ${t('が未ダウンロード')}`) : t('llama-server 未起動 — クリックで準備状況')}>LLM</button>
            <button type="button" className="icon-btn" onClick={() => onTheme(nextTheme)} aria-label={t('テーマ切り替え')}
                title={`${t('テーマ:')} ${themePref === 'system' ? t('システムに合わせる') : themePref === 'light' ? t('ライト') : t('ダーク')} ${t('(クリックで切り替え)')}`}>
                <Icon name={themePref === 'light' ? 'sun' : themePref === 'dark' ? 'moon' : 'layers'} />
            </button>
            <button type="button" className="icon-btn" onClick={() => uiEvents.emit('openHelp')} aria-label={t('使い方')} title={t('使い方と用語 (?)')}><Icon name="help" /></button>
            <button type="button" className="icon-btn" onClick={() => uiEvents.emit('openSettings')} aria-label={t('設定')} title={t('設定')}><Icon name="settings" /></button>
        </header>
    );
}

function ConnectionBanner() {
    const { online } = useStore();
    const [since, setSince] = useState<number | null>(null);
    useEffect(() => setSince(online ? null : Date.now()), [online]);
    if (online) return null;
    return (
        <div className="offline-banner" role="alert">
            <Spinner size={12} /> {t('バックエンドに接続できません。自動で再接続しています…')}
            {since && Date.now() - since > 15000 && <span className="small"> {t('長く続く場合はアプリを起動し直してください (作業台の内容は保存されています)。')}</span>}
        </div>
    );
}

function Toasts() {
    const { toasts, dismissToast } = useStore();
    return (
        <div className="toasts" aria-live="polite">
            {toasts.map(x => (
                <div key={x.id} className={`toast toast-${x.kind}`} role={x.kind === 'error' ? 'alert' : 'status'}>
                    <span className="grow">{x.text}</span>
                    {x.action && <button type="button" className="link" onClick={() => { x.action?.run(); dismissToast(x.id); }}>{x.action.label}</button>}
                    <button type="button" className="icon-btn" aria-label={t('閉じる')} onClick={() => dismissToast(x.id)}><Icon name="close" size={12} /></button>
                </div>
            ))}
        </div>
    );
}

function Layout() {
    const store = useStore();
    const { jobs, leftTabRequest, runPrediction, workbench, requestAssistant, openExample, openJob, showLeftTab } = store;
    const [themePref, setThemePref] = usePersistentState<ThemePref>('oritatami.theme', 'system');
    const theme = useTheme(themePref);
    const [left, setLeft] = usePersistentState<'workbench' | 'jobs' | 'board'>('oritatami.leftTab', 'workbench');
    const [sizes, setSizes] = usePersistentState('oritatami.layout.v2', DEFAULTS);
    const [collapsed, setCollapsed] = usePersistentState('oritatami.collapsed', { left: false, right: false });
    const [fullscreen, setFullscreen] = useState(false);
    const [dialog, setDialog] = useState<null | 'settings' | 'help' | 'palette' | 'autopilot'>(null);

    useEffect(() => {
        if (leftTabRequest) {
            setLeft(leftTabRequest.tab);
            setCollapsed(c => ({ ...c, left: false }));
            setFullscreen(false);
        }
    }, [leftTabRequest, setLeft, setCollapsed]);
    // A press that was swallowed because the same action was already running says so, once:
    // silence would look like the click missed, and one toast per press would be its own noise.
    const lastDuplicate = useRef(0);
    useEffect(() => uiEvents.on('duplicateAction', label => {
        const now = Date.now();
        if (now - lastDuplicate.current < 2500) return;
        lastDuplicate.current = now;
        store.toast('info', label
            ? `${t('「')}${label}${t('」はすでに実行中です。もう一度押す必要はありません')}`
            : t('同じ操作がすでに実行中です。もう一度押す必要はありません'));
    }), [store]);
    useEffect(() => uiEvents.on('openSettings', () => setDialog('settings')), []);
    useEffect(() => uiEvents.on('openHelp', () => setDialog('help')), []);
    useEffect(() => uiEvents.on('openPalette', () => setDialog('palette')), []);
    useEffect(() => uiEvents.on('openAutopilot', () => setDialog('autopilot')), []);
    useEffect(() => uiEvents.on('toggleViewerFullscreen', () => setFullscreen(f => !f)), []);
    // a request to the assistant should make the assistant visible
    useEffect(() => {
        if (store.assistantRequest) {
            setCollapsed(c => ({ ...c, right: false }));
            setFullscreen(false);
        }
    }, [store.assistantRequest, setCollapsed]);

    // dialogs owned by panels (e.g. "add molecule") are not in `dialog`; look at the DOM so shortcuts stay out of their way
    const noDialog = () => dialog === null && !document.querySelector('.modal-backdrop');

    /** Move to the next/previous finished prediction and show it. */
    const stepJob = (delta: number) => {
        const list = jobs.filter(j => j.kind === 'predict' && j.status === 'succeeded');
        if (!list.length) return;
        const current = store.view?.kind === 'job' ? store.view.jobId : null;
        const at = list.findIndex(j => j.id === current);
        // jobs are newest-first, so "next" (j / ]) means the older neighbour
        const next = at < 0 ? 0 : Math.min(list.length - 1, Math.max(0, at + delta));
        if (list[next] && list[next].id !== current) {
            showLeftTab('jobs');
            openJob(list[next].id);
        }
    };
    useHotkeys([
        { combo: 'mod+k', handler: () => setDialog(d => (d === 'palette' ? null : 'palette')), allowInInputs: true },
        { combo: 'mod+enter', handler: e => {
            // inside the LLM composer, mod+enter sends the message instead (handled there)
            if ((e.target as HTMLElement | null)?.closest('.composer')) return;
            if (noDialog()) void runPrediction();
        }, allowInInputs: true },
        { combo: 'mod+,', handler: () => setDialog('settings'), allowInInputs: true },
        { combo: 'mod+i', handler: () => { if (noDialog()) { showLeftTab('workbench'); uiEvents.emit('openAdd', 'uniprot'); } }, allowInInputs: true },
        { combo: '?', handler: () => noDialog() && setDialog('help') },
        { combo: 'f', handler: () => noDialog() && setFullscreen(f => !f) },
        { combo: 'r', handler: () => noDialog() && uiEvents.emit('viewerAction', 'reset') },
        { combo: 'escape', handler: () => { if (noDialog() && fullscreen) setFullscreen(false); } },
        { combo: 'mod+1', handler: () => showLeftTab('workbench'), allowInInputs: true },
        { combo: 'mod+2', handler: () => showLeftTab('jobs'), allowInInputs: true },
        { combo: 'mod+3', handler: () => showLeftTab('board'), allowInInputs: true },
        { combo: 'mod+b', handler: () => setCollapsed(c => ({ ...c, left: !c.left })), allowInInputs: true },
        { combo: 'mod+shift+b', handler: () => setCollapsed(c => ({ ...c, right: !c.right })), allowInInputs: true },
        { combo: 'mod+shift+a', handler: () => uiEvents.emit('toggleAutopilot'), allowInInputs: true },
        // ── step through results without the mouse ──────────────────────────────
        { combo: 'j', handler: () => noDialog() && stepJob(1) },
        { combo: 'k', handler: () => noDialog() && stepJob(-1) },
        { combo: ']', handler: () => noDialog() && stepJob(1) },
        { combo: '[', handler: () => noDialog() && stepJob(-1) },
        { combo: 'mod+shift+arrowdown', handler: () => stepJob(1), allowInInputs: true },
        { combo: 'mod+shift+arrowup', handler: () => stepJob(-1), allowInInputs: true },
        // ── viewer ─────────────────────────────────────────────────────────────
        { combo: 's', handler: () => noDialog() && uiEvents.emit('viewerAction', 'spin') },
        { combo: 'p', handler: () => noDialog() && uiEvents.emit('viewerAction', 'screenshot') },
        { combo: 'o', handler: () => noDialog() && uiEvents.emit('viewerAction', 'pocket') },
        { combo: 'v', handler: () => noDialog() && uiEvents.emit('viewerStyle', 'next') },
        { combo: 'shift+v', handler: () => noDialog() && uiEvents.emit('viewerStyle', 'prev') },
        { combo: 'c', handler: () => noDialog() && uiEvents.emit('viewerColor', 'next') },
        { combo: 'shift+c', handler: () => noDialog() && uiEvents.emit('viewerColor', 'prev') },
        // ── ask the LLM ───────────────────────────────────────────────────────────
        { combo: '/', handler: () => {
            if (!noDialog()) return;
            setCollapsed(c => ({ ...c, right: false }));
            setFullscreen(false);
            window.setTimeout(() => uiEvents.emit('focusAssistant'), 0);
        } },
    ]);

    const commands = useMemo<Command[]>(() => {
        const list: Command[] = [
            { id: 'predict', section: t('作業台'), label: t('構造を予測する'), shortcut: 'mod+enter', keywords: t('predict run boltz 実行 計算'), run: () => void runPrediction(), disabled: !workbench.components.length },
            { id: 'add-protein', section: t('作業台'), label: t('タンパク質を UniProt から追加'), shortcut: 'mod+i', keywords: t('add protein uniprot 検索'), run: () => { showLeftTab('workbench'); uiEvents.emit('openAdd', 'uniprot'); } },
            { id: 'add-paste', section: t('作業台'), label: t('配列を貼り付けて追加 (FASTA)'), keywords: 'paste fasta sequence dna rna', run: () => { showLeftTab('workbench'); uiEvents.emit('openAdd', 'paste'); } },
            { id: 'add-ligand', section: t('作業台'), label: t('リガンド・薬・補因子を追加'), keywords: t('ligand drug smiles ccd pubchem 化合物'), run: () => { showLeftTab('workbench'); uiEvents.emit('openAdd', 'ligand'); } },
            { id: 'add-pdb', section: t('作業台'), label: t('PDB / AlphaFold DB から取り込む'), keywords: 'pdb afdb alphafold import', run: () => { showLeftTab('workbench'); uiEvents.emit('openAdd', 'pdb'); } },
            { id: 'add-file', section: t('作業台'), label: t('構造ファイル (mmCIF / PDB) を開く'), keywords: 'file cif pdb open upload', run: () => { showLeftTab('workbench'); uiEvents.emit('openAdd', 'file'); } },
            { id: 'add-library', section: t('作業台'), label: t('ライブラリから開く'), keywords: t('library saved 保存'), run: () => { showLeftTab('workbench'); uiEvents.emit('openAdd', 'library'); } },
            { id: 'add-history', section: t('作業台'), label: t('検索履歴からもう一度追加する'), keywords: t('history search recent 履歴 前に調べた'), run: () => { showLeftTab('workbench'); uiEvents.emit('openAdd', 'history'); } },
            { id: 'autopilot-plan', section: t('アプリ'), label: t('自律ループの設定 (起点・進め方・禁止リスト)'), keywords: t('autopilot loop 自動 探索 ループ 保護 禁止'), run: () => uiEvents.emit('openAutopilot') },
            { id: 'autopilot-toggle', section: t('アプリ'), label: t('自律ループをその場で動かす / 止める'), shortcut: 'mod+shift+a', keywords: t('autopilot loop 自動 停止 toggle'), run: () => uiEvents.emit('toggleAutopilot') },
            { id: 'ask-focus', section: 'LLM', label: t('LLM に質問を書く'), shortcut: '/', keywords: t('ask chat qwen 質問 入力'), run: () => { setCollapsed(c => ({ ...c, right: false })); setFullscreen(false); window.setTimeout(() => uiEvents.emit('focusAssistant'), 0); } },
            { id: 'ask-mutations', section: 'LLM', label: t('LLM に変異を提案させる'), keywords: t('ai mutation llm qwen 提案'), run: () => requestAssistant('mutations') },
            { id: 'ask-complex', section: 'LLM', label: t('LLM に結合相手を提案させる'), keywords: 'ai complex partner ligand qwen', run: () => requestAssistant('complex') },
            { id: 'ask-design', section: 'LLM', label: t('LLM に新しい配列を設計させる'), keywords: 'ai design de novo qwen', run: () => requestAssistant('design') },
            { id: 'view-reset', section: t('3D ビューア'), label: t('視点をリセット'), shortcut: 'r', keywords: 'camera reset', run: () => uiEvents.emit('viewerAction', 'reset') },
            { id: 'view-spin', section: t('3D ビューア'), label: t('自動回転の切り替え'), shortcut: 's', keywords: 'spin rotate', run: () => uiEvents.emit('viewerAction', 'spin') },
            { id: 'view-style', section: t('3D ビューア'), label: t('表示のしかたを次へ'), shortcut: 'v', keywords: t('style cartoon surface representation リボン 表面'), run: () => uiEvents.emit('viewerStyle', 'next') },
            { id: 'view-color', section: t('3D ビューア'), label: t('色分けを次へ'), shortcut: 'c', keywords: t('color plddt chain rainbow 色'), run: () => uiEvents.emit('viewerColor', 'next') },
            { id: 'view-pocket', section: t('3D ビューア'), label: t('ポケット表示の切り替え'), shortcut: 'o', keywords: 'pocket ligand site', run: () => uiEvents.emit('viewerAction', 'pocket') },
            { id: 'view-shot', section: t('3D ビューア'), label: t('画像を撮る'), shortcut: 'p', keywords: t('screenshot image png 画像 保存'), run: () => uiEvents.emit('viewerAction', 'screenshot') },
            { id: 'view-full', section: t('3D ビューア'), label: t('ビューアを大きく表示 / 戻す'), shortcut: 'f', keywords: 'fullscreen maximize', run: () => setFullscreen(f => !f) },
            { id: 'tab-workbench', section: t('表示'), label: t('作業台を表示'), shortcut: 'mod+1', run: () => showLeftTab('workbench') },
            { id: 'tab-jobs', section: t('表示'), label: t('ジョブ一覧を表示'), shortcut: 'mod+2', keywords: t('jobs history 履歴 一覧 進捗 残り 終了'), run: () => showLeftTab('jobs') },
            { id: 'next-job', section: t('表示'), label: t('次の予測結果へ'), shortcut: 'j', keywords: t('next result 次 結果 移動'), run: () => stepJob(1) },
            { id: 'prev-job', section: t('表示'), label: t('前の予測結果へ'), shortcut: 'k', keywords: t('previous result 前 結果 移動'), run: () => stepJob(-1) },
            { id: 'tab-board', section: t('表示'), label: t('成果 (スコア順・系統) を表示'), shortcut: 'mod+3', keywords: t('leaderboard ranking lineage score plddt 順位 系統 リーダーボード'), run: () => showLeftTab('board') },
            { id: 'toggle-left', section: t('表示'), label: t('左パネルの表示 / 非表示'), shortcut: 'mod+b', run: () => setCollapsed(c => ({ ...c, left: !c.left })) },
            { id: 'toggle-right', section: t('表示'), label: t('LLM パネルの表示 / 非表示'), shortcut: 'mod+shift+b', run: () => setCollapsed(c => ({ ...c, right: !c.right })) },
            { id: 'theme-light', section: t('表示'), label: t('ライトテーマにする'), keywords: t('theme light 明るい'), run: () => setThemePref('light') },
            { id: 'theme-dark', section: t('表示'), label: t('ダークテーマにする'), keywords: t('theme dark 暗い'), run: () => setThemePref('dark') },
            { id: 'theme-system', section: t('表示'), label: t('テーマをシステムに合わせる'), keywords: 'theme system auto', run: () => setThemePref('system') },
            { id: 'layout-reset', section: t('表示'), label: t('パネルの配置を元に戻す'), keywords: 'layout reset', run: () => { setSizes(DEFAULTS); setCollapsed({ left: false, right: false }); } },
            { id: 'settings', section: t('アプリ'), label: t('設定・準備状況'), shortcut: 'mod+,', keywords: t('settings preferences ollama llama model storage 設定 環境設定 準備'), run: () => setDialog('settings') },
            { id: 'help', section: t('アプリ'), label: t('使い方と用語'), shortcut: '?', keywords: t('help glossary plddt ptm iptm pae ヘルプ 使い方 用語 わからない'), run: () => setDialog('help') },
        ];
        EXAMPLES.forEach(ex => list.push({ id: `ex-${ex.id}`, section: t('例を開く'), label: `${ex.title} — ${ex.summary}`, keywords: `example ${ex.tags.join(' ')}`, run: () => void openExample(ex) }));
        jobs.slice(0, 40).forEach(j => list.push({ id: `job-${j.id}`, section: t('ジョブを開く'), label: j.title, keywords: `${j.kind} ${j.status}`, run: () => { showLeftTab('jobs'); openJob(j.id); } }));
        return list;
    }, [runPrediction, workbench.components.length, showLeftTab, requestAssistant, setCollapsed, setThemePref, setSizes, openExample, jobs, openJob]);

    const active = jobs.filter(j => j.status === 'running' || j.status === 'queued').length;
    // Fixed side panels used to squeeze the 3D viewer down to a couple of hundred pixels in a
    // narrow window. Fold them to their rails instead — LLM first, then the workbench — and
    // bring them back untouched when there is room again. Opening a folded rail by hand wins
    // (`opened`), until the next resize decides again.
    const width = useWindowWidth();
    const [opened, setOpened] = useState({ left: false, right: false });
    useEffect(() => { setOpened({ left: false, right: false }); }, [width]);
    const wantLeft = !collapsed.left && !fullscreen;
    const wantRight = !collapsed.right && !fullscreen;
    const needed = (l: boolean, r: boolean) =>
        (l ? sizes.left + 5 : RAIL) + MIN_CENTER + (r ? sizes.right + 5 : RAIL);
    let showLeft = wantLeft;
    let showRight = wantRight;
    if (!fullscreen) {
        if (showRight && !opened.right && needed(showLeft, showRight) > width) showRight = false;
        if (showLeft && !opened.left && needed(showLeft, showRight) > width) showLeft = false;
        // an explicitly opened panel folds the other side rather than crushing the viewer
        if (opened.left && showRight && needed(true, true) > width) showRight = false;
        if (opened.right && showLeft && needed(true, true) > width) showLeft = false;
    }
    const folded = { left: wantLeft && !showLeft, right: wantRight && !showRight };
    /** Rendered width: a hand-opened panel in a narrow window gives ground before the viewer does. */
    const fit = (want: number, otherShown: boolean, otherWidth: number) =>
        Math.max(260, Math.min(want, width - (otherShown ? otherWidth + 5 : RAIL) - MIN_CENTER - 5));
    const leftW = showLeft ? fit(sizes.left, showRight, sizes.right) : 0;
    const rightW = showRight ? fit(sizes.right, showLeft, leftW) : 0;
    const unfold = (side: 'left' | 'right') => {
        setCollapsed(c => ({ ...c, [side]: false }));
        setOpened(o => ({ ...o, [side]: true }));
    };
    const columns = [
        showLeft ? `${leftW}px 5px` : fullscreen ? '' : `${RAIL}px`,
        'minmax(0, 1fr)',
        showRight ? `5px ${rightW}px` : fullscreen ? '' : `${RAIL}px`,
    ].filter(Boolean).join(' ');

    return (
        <div className={`app ${fullscreen ? 'is-fullscreen' : ''}`}>
            <TopBar themePref={themePref} onTheme={setThemePref} />
            <ConnectionBanner />
            <main className="layout" style={{ gridTemplateColumns: columns }}>
                {/* Panels stay mounted while hidden so in-flight work (an LLM request, an open dialog) survives. */}
                <aside className="col-left" aria-label={t('作業台とジョブ')} hidden={!showLeft}>
                    <div className="col-tabs">
                        <Tabs value={left} onChange={setLeft} tabs={[
                            { id: 'workbench', label: t('作業台'), title: '⌘1' },
                            { id: 'jobs', label: t('ジョブ'), badge: active || undefined, title: '⌘2' },
                            { id: 'board', label: t('成果'), title: t('⌘3 — スコア順と系統') },
                        ]} />
                        <button type="button" className="icon-btn collapse-btn" onClick={() => setCollapsed(c => ({ ...c, left: true }))} aria-label={t('左パネルを隠す')} title={t('隠す (⌘B)')}><Icon name="chevron-left" /></button>
                    </div>
                    <div className="col-body">
                        <ErrorBoundary area={left === 'workbench' ? t('作業台') : left === 'jobs' ? t('ジョブ一覧') : t('成果')}>
                            {left === 'workbench' ? <WorkbenchPanel />
                                : left === 'jobs' ? <JobsPanel />
                                    : <LeaderboardPanel />}
                        </ErrorBoundary>
                    </div>
                </aside>
                {showLeft && (
                    <Splitter orientation="vertical" label={t('左パネルの幅')} value={leftW} min={300}
                        max={Math.max(300, Math.min(620, width - (showRight ? rightW + 5 : RAIL) - MIN_CENTER - 5))}
                        onChange={v => setSizes(s => ({ ...s, left: v }))} onReset={() => setSizes(s => ({ ...s, left: DEFAULTS.left }))} />
                )}
                {!showLeft && !fullscreen && (
                    <button type="button" className={`rail ${folded.left ? 'rail-folded' : ''}`} onClick={() => unfold('left')}
                        aria-label={t('左パネルを表示')}
                        title={folded.left ? t('ウィンドウが狭いので畳んでいます。広げると自動で戻ります (クリックでも開きます)') : t('作業台とジョブを表示 (⌘B)')}>
                        <Icon name="chevron-right" />
                        <span className="rail-label">{t('作業台・ジョブ')}{active ? ` (${active})` : ''}</span>
                    </button>
                )}
                <section className="col-center" style={{ gridTemplateRows: fullscreen ? 'minmax(0, 1fr)' : `minmax(0, 1fr) 5px ${sizes.results}px` }}>
                    <ErrorBoundary area={t('3D ビューア')}>
                        <ViewerPanel theme={theme} fullscreen={fullscreen} onToggleFullscreen={() => setFullscreen(f => !f)} />
                    </ErrorBoundary>
                    {!fullscreen && (
                        <Splitter orientation="horizontal" label={t('結果パネルの高さ')} value={sizes.results} min={140} max={Math.max(200, window.innerHeight - 260)} direction={-1}
                            onChange={v => setSizes(s => ({ ...s, results: v }))} onReset={() => setSizes(s => ({ ...s, results: DEFAULTS.results }))} />
                    )}
                    <div className="results-slot" hidden={fullscreen}>
                        <ErrorBoundary area={t('結果')}>
                            <ResultsPanel />
                        </ErrorBoundary>
                    </div>
                </section>
                {showRight && (
                    <Splitter orientation="vertical" label={t('LLM パネルの幅')} value={rightW} min={300} direction={-1}
                        max={Math.max(300, Math.min(640, width - (showLeft ? leftW + 5 : RAIL) - MIN_CENTER - 5))}
                        onChange={v => setSizes(s => ({ ...s, right: v }))} onReset={() => setSizes(s => ({ ...s, right: DEFAULTS.right }))} />
                )}
                <aside className="col-right" aria-label={t('LLM アシスタント')} hidden={!showRight}>
                    <ErrorBoundary area={t('LLM アシスタント')}>
                        <AssistantPanel onCollapse={() => setCollapsed(c => ({ ...c, right: true }))} />
                    </ErrorBoundary>
                </aside>
                {!showRight && !fullscreen && (
                    <button type="button" className={`rail rail-right ${folded.right ? 'rail-folded' : ''}`} onClick={() => unfold('right')}
                        aria-label={t('LLM パネルを表示')}
                        title={folded.right ? t('ウィンドウが狭いので畳んでいます。広げると自動で戻ります (クリックでも開きます)') : t('LLM を表示 (⌘⇧B)')}>
                        <Icon name="chevron-left" />
                        <span className="rail-label">LLM</span>
                    </button>
                )}
            </main>
            <Toasts />
            {dialog === 'settings' && <SettingsDialog onClose={() => setDialog(null)} />}
            {dialog === 'help' && <HelpDialog onClose={() => setDialog(null)} />}
            {dialog === 'palette' && <CommandPalette commands={commands} onClose={() => setDialog(null)} />}
            {dialog === 'autopilot' && <AutopilotDialog onClose={() => setDialog(null)} />}
        </div>
    );
}

export function App() {
    return (
        <StoreProvider>
            <Layout />
        </StoreProvider>
    );
}
