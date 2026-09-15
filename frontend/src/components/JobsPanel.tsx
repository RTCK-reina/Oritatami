import { useEffect, useMemo, useState } from 'react';
import { api, errorMessage, type QueueEta as QueueEtaData } from '../api';
import { useStore } from '../store';
import type { JobSummary, LiveMemory } from '../types';
import { fmt, formatDuration } from '../workbench';
import { Button, Empty, Icon, Menu, Spinner, StatusDot } from './ui';

type Filter = 'all' | 'active' | 'predict' | 'scan' | 'refine' | 'starred' | 'auto' | 'manual';
const KIND_LABEL: Record<string, string> = { predict: '予測', scan: 'スキャン', refine: 'ESM改良' };
const FILTERS: { id: Filter; label: string }[] = [
    { id: 'all', label: 'すべて' }, { id: 'active', label: '実行中' }, { id: 'predict', label: '予測' },
    { id: 'scan', label: 'スキャン' }, { id: 'refine', label: 'ESM改良' },
    { id: 'auto', label: '自律ループ' }, { id: 'manual', label: '手動' }, { id: 'starred', label: '★' },
];

/** Which of these did the loop start? Without this the list mixes a night's worth of
 *  machine-made variants with the handful a person queued, and neither can be found. */
function isAuto(j: JobSummary): boolean {
    return j.origin.startsWith('autopilot');
}

function timeLabel(ts: number): string {
    const d = new Date(ts * 1000);
    const today = new Date();
    const hm = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
    return d.toDateString() === today.toDateString() ? hm : `${d.getMonth() + 1}/${d.getDate()} ${hm}`;
}

export function JobsPanel() {
    const { jobs, jobsLoaded, selectedJobId, openJob, refreshJobs, compare, retryJob, exportJob, toast } = useStore();
    const [filter, setFilter] = useState<Filter>('all');
    // Cancelling the whole queue is one click away from a decision the user cannot undo without
    // re-running everything, so the first click only arms it.
    const [confirmCancelAll, setConfirmCancelAll] = useState(false);
    // Same reasoning for the two bulk buttons: one stops work that is already running, the
    // other throws rows away. Neither is undoable by pressing it again.
    const [armed, setArmed] = useState<'stop' | 'clear' | null>(null);
    const arm = (which: 'stop' | 'clear') => {
        setArmed(which);
        window.setTimeout(() => setArmed(a => (a === which ? null : a)), 4000);
    };
    const [query, setQuery] = useState('');
    const [compareFrom, setCompareFrom] = useState<string | null>(null);
    const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
    const [renaming, setRenaming] = useState<string | null>(null);

    const byId = useMemo(() => new Map(jobs.map(j => [j.id, j])), [jobs]);
    const shown = useMemo(() => {
        const q = query.trim().toLowerCase();
        return jobs.filter(j => {
            if (filter === 'active' && !(j.status === 'running' || j.status === 'queued')) return false;
            if (filter === 'starred' && !j.starred) return false;
            if (filter === 'auto' && !isAuto(j)) return false;
            if (filter === 'manual' && (isAuto(j) || j.origin === 'qwen')) return false;
            if (['predict', 'scan', 'refine'].includes(filter) && j.kind !== filter) return false;
            if (!q) return true;
            const muts = (j.spec.components ?? []).flatMap(c => c.mutations ?? []).join(' ');
            const exp = String(j.spec.autopilot_experiment ?? '');
            return `${j.title} ${muts} ${j.id} ${exp}`.toLowerCase().includes(q);
        });
    }, [jobs, filter, query]);
    const activeCount = jobs.filter(j => j.status === 'running' || j.status === 'queued').length;
    const failedCount = jobs.filter(j => j.status === 'failed' || j.status === 'cancelled').length;
    const queuedCount = jobs.filter(j => j.status === 'queued').length;
    const stoppedCount = jobs.filter(j => j.status === 'cancelled').length;

    const act = async (fn: () => Promise<unknown>) => {
        try {
            await fn();
            await refreshJobs();
        } catch (e) {
            toast('error', errorMessage(e));
        }
    };

    return (
        <div className="panel jobs">
            <div className="panel-head jobs-head">
                <label className="search-box">
                    <Icon name="search" size={13} />
                    <input value={query} onChange={e => setQuery(e.target.value)} placeholder="名前・変異で絞り込み" aria-label="ジョブを検索" />
                </label>
                <Menu label="整理" items={[
                    { label: confirmCancelAll ? `本当に待機中をすべて取り消す (${queuedCount})` : `待機中をすべて取り消す (${queuedCount})`,
                      hint: '実行中のジョブはそのまま。取り消したジョブは「再実行」で戻せます',
                      disabled: queuedCount === 0,
                      onSelect: () => {
                          if (!confirmCancelAll) {
                              setConfirmCancelAll(true);
                              window.setTimeout(() => setConfirmCancelAll(false), 4000);
                              return;
                          }
                          setConfirmCancelAll(false);
                          void act(async () => {
                              const r = await api.cancelQueued();
                              toast('success', `${r.count} 件を取り消しました`);
                          });
                      } },
                    { label: `失敗・キャンセルしたジョブを削除 (${failedCount})`, disabled: failedCount === 0, onSelect: () => void act(async () => {
                        const r = await api.cleanupStorage({ intermediate: false, aligned_older_than_days: null, delete_failed_jobs: true });
                        toast('success', `${r.jobs_deleted} 件を削除しました`);
                    }) },
                ]} />
            </div>
            <div className="seg-wrap">
                <div className="seg" role="radiogroup" aria-label="種類で絞り込み">
                    {FILTERS.map(f => (
                        <button type="button" key={f.id} role="radio" aria-checked={filter === f.id} className={filter === f.id ? 'active' : ''} onClick={() => setFilter(f.id)}>
                            {f.label}{f.id === 'active' && activeCount > 0 ? ` ${activeCount}` : ''}
                        </button>
                    ))}
                </div>
            </div>
            <QueueEtaLine active={activeCount} />
            {(activeCount > 0 || stoppedCount > 0) && (
                <div className="bulk-bar small">
                    {activeCount > 0 && (
                        <Button size="sm" variant={armed === 'stop' ? 'danger' : 'ghost'}
                            title="実行中のジョブも含めて、いま動いているものと待っているものを全部止めます。結果は消えません"
                            onClick={() => {
                                if (armed !== 'stop') return arm('stop');
                                setArmed(null);
                                void act(async () => {
                                    const r = await api.cancelQueued(undefined, true);
                                    toast('success', `${r.count} 件を中断しました`);
                                });
                            }}>
                            {armed === 'stop' ? `本当にすべて中断 (${activeCount})` : `すべて中断 (${activeCount})`}
                        </Button>
                    )}
                    {stoppedCount > 0 && (
                        <Button size="sm" variant={armed === 'clear' ? 'danger' : 'ghost'}
                            title="中断・キャンセルしたジョブの行と作業フォルダを削除します。成功したジョブには触れません"
                            onClick={() => {
                                if (armed !== 'clear') return arm('clear');
                                setArmed(null);
                                void act(async () => {
                                    const r = await api.cleanupStorage({ intermediate: false, aligned_older_than_days: null, delete_failed_jobs: true });
                                    toast('success', `${r.jobs_deleted} 件を削除しました`);
                                });
                            }}>
                            {armed === 'clear' ? `本当に削除 (${failedCount})` : `中断・失敗を削除 (${failedCount})`}
                        </Button>
                    )}
                </div>
            )}
            {compareFrom && (
                <div className="compare-bar small">
                    比較元: <strong>{byId.get(compareFrom)?.title}</strong> — 重ねたい結果の「重ねる」を押してください
                    <button type="button" className="link" onClick={() => setCompareFrom(null)}>やめる</button>
                </div>
            )}
            <div className="panel-scroll">
                {!jobsLoaded && <Empty><Spinner /></Empty>}
                {jobsLoaded && shown.length === 0 && (
                    <Empty>{jobs.length === 0 ? <>ジョブはまだありません。<br />作業台で「構造を予測する」を押すとここに並びます。</> : '条件に合うジョブがありません'}</Empty>
                )}
                {shown.map(j => (
                    <JobRow key={j.id} job={j} parent={j.parent_id ? byId.get(j.parent_id) : undefined}
                        selected={selectedJobId === j.id}
                        renaming={renaming === j.id}
                        confirmDelete={confirmDelete === j.id}
                        compareFrom={compareFrom}
                        onOpen={openJob}
                        onStar={() => void act(() => api.patchJob(j.id, { starred: !j.starred }))}
                        onRename={title => { setRenaming(null); if (title && title !== j.title) void act(() => api.patchJob(j.id, { title })); }}
                        onStartRename={() => setRenaming(j.id)}
                        onCancel={() => void act(() => api.cancelJob(j.id))}
                        onReorder={action => void act(() => api.reorderJob(j.id, action))}
                        onRetry={() => void retryJob(j.id)}
                        onExport={how => void exportJob(j.id, how)}
                        onDelete={() => {
                            if (confirmDelete !== j.id) {
                                setConfirmDelete(j.id);
                                window.setTimeout(() => setConfirmDelete(c => (c === j.id ? null : c)), 4000);
                                return;
                            }
                            setConfirmDelete(null);
                            void act(() => api.deleteJob(j.id));
                        }}
                        onCompare={() => {
                            if (!compareFrom) {
                                setCompareFrom(j.id);
                            } else if (compareFrom !== j.id) {
                                void compare(compareFrom, j.id);
                                setCompareFrom(null);
                            }
                        }} />
                ))}
            </div>
        </div>
    );
}

/** The only honest progress signal a long Boltz run has.
 *
 *  Boltz prints no progress inside the diffusion phase, so a job that is merely big and one
 *  that is wedged look identical from the outside. What separates them is the footprint: it
 *  keeps moving, and whether it sits under physical memory or above it decides whether the
 *  run is going at full speed or at roughly 1/300th of it — measured on this machine, 97 GB/s
 *  of page traffic resident against 296 MB/s once swapping. Nothing here stops anything; a
 *  multi-day run is a legitimate thing to want, it just should not be a surprise.
 */
function RunMemory({ mem, overrun }: { mem?: LiveMemory; overrun: number }) {
    if (!mem || !mem.footprint_gb) return null;
    const total = mem.total_gb;
    const over = total ? mem.footprint_gb - total : 0;
    const swapping = over > 0;
    return (
        <div className="job-meta small mono">
            <span className={swapping ? 'warn' : 'muted'}>
                {swapping
                    ? `物理メモリを ${over.toFixed(1)} GB 超過 — スワップ動作中`
                    : `メモリ ${mem.footprint_gb.toFixed(1)}${total ? ` / ${total.toFixed(0)}` : ''} GB`}
            </span>
            {swapping && <span className="muted" title="スワップ中はページアウトが続きます。実測で約 190 MB/s、1 日あたり約 16 TB">SSD 書き込み 約 190 MB/s</span>}
            {swapping && mem.free_disk_gb !== null && (
                <span className={mem.free_disk_gb < 20 ? 'bad-text' : 'muted'}>空き {mem.free_disk_gb} GB</span>
            )}
            {typeof mem.user_share === 'number' && (mem.cpu_sec ?? 0) >= 5 && (
                // Boltz reports no progress from inside the diffusion phase, and CPU time is
                // not a substitute — the GPU does the arithmetic, so a healthy run's CPU
                // efficiency falls with size (0.55 at 76 tokens, 0.10 at 608). What the CPU
                // does say is what it is busy with: computing, or moving pages. Healthy runs
                // measured 64-86 % user time; one that paged from end to end, 1.6 %.
                <span className={mem.user_share < 0.25 ? 'warn' : 'muted'}
                    title={`CPU 時間のうち計算に使われた割合。ページング待ちになると落ちます${
                        typeof mem.efficiency === 'number' ? `\nCPU 時間 ÷ 実時間 は ${(mem.efficiency * 100).toFixed(0)}%（大きいジョブほど GPU 待ちで下がるので、これ自体は不調の指標になりません）` : ''}`}>
                    {mem.user_share < 0.25 ? 'ページング待ち' : '計算中'} {(mem.user_share * 100).toFixed(0)}%
                </span>
            )}
            {overrun >= 2 && (
                <span className="muted" title="Boltz の内部進捗は最後まで 1/1 のままです">
                    進捗表示なし
                </span>
            )}
        </div>
    );
}

function JobRow(props: {
    job: JobSummary; parent?: JobSummary; selected: boolean; renaming: boolean; confirmDelete: boolean; compareFrom: string | null;
    onOpen: (id: string) => void; onStar: () => void; onRename: (t: string) => void; onStartRename: () => void;
    onCancel: () => void; onRetry: () => void; onExport: (how: 'download' | 'folder') => void; onDelete: () => void; onCompare: () => void;
    onReorder: (action: 'top' | 'up' | 'down' | 'bottom') => void;
}) {
    const { job: j, parent } = props;
    const conf = j.result?.confidence;
    const active = j.status === 'running' || j.status === 'queued';
    const progress = typeof j.live.progress === 'number' ? j.live.progress : null;
    const elapsed = j.started_at ? ((j.finished_at ?? Date.now() / 1000) - j.started_at) : 0;
    const estimateSec = typeof j.live.estimate_sec === 'number' ? j.live.estimate_sec : null;
    const overrun = estimateSec && estimateSec > 0 ? elapsed / estimateSec : 0;
    const mutations = (j.spec.components ?? []).flatMap(c => c.mutations ?? []);
    return (
        <div className={`job-row status-row-${j.status} ${props.selected ? 'selected' : ''}`} role="button" tabIndex={0}
            aria-current={props.selected} onClick={() => props.onOpen(j.id)}
            onKeyDown={e => { if ((e.key === 'Enter' || e.key === ' ') && e.target === e.currentTarget) { e.preventDefault(); props.onOpen(j.id); } }}>
            <div className="job-line">
                <StatusDot status={j.status} />
                {props.renaming ? (
                    <input autoFocus defaultValue={j.title} className="grow" aria-label="ジョブ名" onClick={e => e.stopPropagation()}
                        onBlur={e => props.onRename(e.target.value.trim())}
                        onKeyDown={e => { e.stopPropagation(); if (e.key === 'Enter') e.currentTarget.blur(); if (e.key === 'Escape') props.onRename(''); }} />
                ) : (
                    <span className="job-title grow" onDoubleClick={props.onStartRename} title={`${j.title}\nダブルクリックで名前を変更`}>{j.title}</span>
                )}
                <button type="button" className={`icon-btn star ${j.starred ? 'on' : ''}`} onClick={e => { e.stopPropagation(); props.onStar(); }}
                    aria-label={j.starred ? 'お気に入りを外す' : 'お気に入り'} aria-pressed={j.starred}><Icon name="star" size={13} /></button>
            </div>
            <div className="job-meta small">
                <span className={`kind kind-${j.kind}`}>{KIND_LABEL[j.kind]}</span>
                {j.origin === 'qwen' && <span className="qwen-badge">LLM</span>}
                {isAuto(j) && <span className="auto-badge" title="自律ループが投入したジョブ">自律{
                    j.spec.autopilot_experiment ? ` · ${String(j.spec.autopilot_experiment)}` : ''}</span>}
                <span className="muted">{timeLabel(j.created_at)}</span>
                {parent && <span className="muted ellipsis" title={`元: ${parent.title}`}>↳ {parent.title}</span>}
            </div>
            {active && (
                <>
                    <div className="job-meta small">
                        {j.status === 'queued' ? <>待機中{j.queue_position ? ` (${j.queue_position} 番目)` : ''}</> : <><Spinner size={10} /> {j.live.label ?? j.phase}</>}
                        {j.status === 'running' && elapsed > 0 && <span className="muted">· {formatDuration(elapsed)}</span>}
                        {j.status === 'running' && estimateSec !== null && estimateSec > 0 && (
                            <span className={overrun >= 2 ? 'warn' : 'muted'}
                                title="開始時点の見積もりです。過去の同じくらいの大きさのジョブから当てはめています">
                                / 目安 {formatDuration(estimateSec)}{overrun >= 1.5 ? `（${overrun.toFixed(1)} 倍）` : ''}
                            </span>
                        )}
                    </div>
                    {j.status === 'running' && (
                        <div className={`progress ${progress === null ? 'indeterminate' : ''}`}
                            title={progress === null ? 'Boltz は拡散中の進捗を出しません（内部の表示は最後まで 1/1 のまま）。残り時間は測れないので、経過時間とメモリの動きで判断してください' : undefined}>
                            <i style={progress !== null ? { width: `${Math.round(progress * 100)}%` } : undefined} />
                        </div>
                    )}
                    {j.status === 'running' && <RunMemory mem={j.live.memory} overrun={overrun} />}
                </>
            )}
            {j.kind === 'predict' && mutations.length > 0 && (
                <div className="job-meta small res-chips">
                    {mutations.slice(0, 6).map(m => <span key={m} className="chip chip-mut">{m}</span>)}
                    {mutations.length > 6 && <span className="muted">他 {mutations.length - 6}</span>}
                </div>
            )}
            {j.status === 'failed' && <div className="job-meta small warn ellipsis" title={j.error ?? ''}>{(j.error ?? '').split('\n')[0]}</div>}
            {j.status === 'succeeded' && j.kind === 'predict' && conf && (
                <div className="job-meta small mono">
                    信頼度 {fmt(conf.confidence_score)} · pTM {fmt(conf.ptm)}
                    {typeof conf.iptm === 'number' && conf.iptm > 0 && ` · ipTM ${fmt(conf.iptm)}`}
                    {j.result?.affinity && ` · 結合 ${(j.result.affinity.affinity_probability_binary * 100).toFixed(0)}%`}
                </div>
            )}
            {j.status === 'succeeded' && j.kind !== 'predict' && j.result?.pseudo_perplexity !== undefined && (
                <div className="job-meta small mono">PPPL {j.result.start_pseudo_perplexity !== undefined ? `${j.result.start_pseudo_perplexity} → ` : ''}{j.result.pseudo_perplexity}</div>
            )}
            <div className="job-actions" onClick={e => e.stopPropagation()}>
                {j.kind === 'predict' && j.status === 'succeeded' && (
                    <Button size="sm" variant="ghost" onClick={props.onCompare}
                        title={props.compareFrom ? 'この結果を比較元に重ねて表示' : '重ね合わせ比較の基準にする'}>
                        {props.compareFrom === j.id ? '比較元' : props.compareFrom ? '重ねる' : '比較'}
                    </Button>
                )}
                {j.status === 'succeeded' && (
                    <Menu label={<Icon name="download" size={13} />} title="書き出し" items={[
                        { label: 'zip を保存…', onSelect: () => props.onExport('download'), hint: '構造・スコア・入力・ログをまとめて保存' },
                        { label: 'ダウンロードフォルダに書き出して Finder で表示', onSelect: () => props.onExport('folder') },
                    ]} />
                )}
                {(j.status === 'failed' || j.status === 'cancelled') && <Button size="sm" variant="ghost" onClick={props.onRetry} title="同じ入力でもう一度実行"><Icon name="retry" size={13} /> 再実行</Button>}
                {j.status === 'queued' && (
                    <span className="queue-move" role="group" aria-label="順番を入れ替える">
                        <button type="button" className="icon-btn" title="先頭へ" aria-label="先頭へ"
                            onClick={() => props.onReorder('top')}><Icon name="to-top" size={13} /></button>
                        <button type="button" className="icon-btn" title="1 つ前へ" aria-label="1 つ前へ"
                            onClick={() => props.onReorder('up')}><Icon name="chevron-up" size={13} /></button>
                        <button type="button" className="icon-btn" title="1 つ後ろへ" aria-label="1 つ後ろへ"
                            onClick={() => props.onReorder('down')}><Icon name="chevron-down" size={13} /></button>
                        <button type="button" className="icon-btn" title="最後へ" aria-label="最後へ"
                            onClick={() => props.onReorder('bottom')}><Icon name="to-bottom" size={13} /></button>
                    </span>
                )}
                {active && <Button size="sm" variant="ghost" onClick={props.onCancel}>キャンセル</Button>}
                {!active && <Button size="sm" variant={props.confirmDelete ? 'danger' : 'ghost'} onClick={props.onDelete} aria-label="削除">{props.confirmDelete ? '本当に削除' : <Icon name="trash" size={13} />}</Button>}
            </div>
        </div>
    );
}

/** The queue's own finish line, above the rows it is made of. */
function QueueEtaLine({ active }: { active: number }) {
    const [eta, setEta] = useState<QueueEtaData | null>(null);
    useEffect(() => {
        let alive = true;
        const load = () => api.queueEta().then(v => { if (alive) setEta(v); }).catch(() => { if (alive) setEta(null); });
        load();
        const t = window.setInterval(load, 20000);
        return () => { alive = false; window.clearInterval(t); };
    }, [active]);
    if (!eta || !eta.jobs) return null;
    const finish = eta.finish_at ? new Date(eta.finish_at * 1000) : null;
    const hm = finish
        ? `${String(finish.getHours()).padStart(2, '0')}:${String(finish.getMinutes()).padStart(2, '0')}`
        : null;
    const sameDay = finish ? finish.toDateString() === new Date().toDateString() : true;
    const mins = Math.round(eta.seconds / 60);
    const left = eta.seconds < 60 ? `${Math.round(eta.seconds)} 秒`
        : mins < 60 ? `${mins} 分`
            : `${Math.floor(mins / 60)} 時間${mins % 60 ? ` ${mins % 60} 分` : ''}`;
    return (
        <div className="queue-eta small" role="status">
            <Icon name="rotate" size={12} />
            <span>残り <strong>{left}</strong>{eta.unknown ? '以上' : ''}</span>
            {hm && <span className="muted">終了見込み {sameDay ? '' : `${finish!.getMonth() + 1}/${finish!.getDate()} `}{hm}</span>}
            <span className="muted">{eta.jobs} 件</span>
            {eta.unknown > 0 && <span className="warn">{eta.unknown} 件は見積もれません</span>}
        </div>
    );
}
