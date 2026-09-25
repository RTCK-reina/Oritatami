import { useEffect, useMemo, useRef, useState } from 'react';
import { api, errorMessage } from '../api';
import type { AutopilotResult, FunctionRisk } from '../api';
import { useStore } from '../store';
import type { Confidence, Estimate, FullJob, Geometry, JobSummary, MsaAlignment, PredictJob, PredictResult, RefineJob, ScanJob } from '../types';
import { viewerBus } from '../viewer/bus';
import { assignChains, fmt, formatDuration } from '../workbench';
import { MsaCoverage, PaeHeatmap, PlddtPlot, ScanHeatmap } from './charts';
import { Button, Empty, Icon, InfoTip, Menu, Metric, Spinner, StatusDot, Tabs } from './ui';
import { t } from '../i18n';

export function ResultsPanel() {
    const { selectedJobId, jobs, getJob, jobCache } = useStore();
    const summary = jobs.find(j => j.id === selectedJobId);
    const [error, setError] = useState<string | null>(null);

    const status = summary?.status;
    useEffect(() => {
        if (!selectedJobId || !status) return;
        let alive = true;
        const terminal = ['succeeded', 'failed', 'cancelled'].includes(status);
        setError(null);
        getJob(selectedJobId, !terminal).catch(e => alive && setError(errorMessage(e)));
        return () => { alive = false; };
    }, [selectedJobId, status, getJob]);

    const current = selectedJobId ? jobCache[selectedJobId] : undefined;

    if (!selectedJobId) {
        return (
            <div className="panel results">
                <Empty>
                    <p>{t('ジョブを選ぶと、ここに結果 (信頼度・グラフ・界面など) が出ます。')}</p>
                    <p className="small muted">{t('計算が終わると自動でここに表示されます。')}</p>
                </Empty>
            </div>
        );
    }
    if (!summary) return <div className="panel results"><Empty>{t('このジョブは削除されました')}</Empty></div>;
    if (error) return <div className="panel results"><div className="warn pad">{error}</div></div>;
    if (summary.status !== 'succeeded') return <JobProgress job={summary} full={current} />;
    if (!current || current.status !== 'succeeded') return <div className="panel results"><Empty><Spinner /></Empty></div>;
    if (current.kind === 'predict') return <PredictResultView job={current} />;
    if (current.kind === 'scan') return <ScanResultView job={current} />;
    return <RefineResultView job={current} />;
}

const PHASE_STEPS: { key: string; label: string }[] = [
    { key: 'starting', label: t('起動') },
    { key: 'preprocess', label: t('前処理') },
    { key: 'msa', label: t('MSA 検索') },
    { key: 'structure', label: t('構造予測') },
    { key: 'affinity', label: t('親和性') },
    { key: 'collect', label: t('集計') },
];

function JobProgress({ job, full }: { job: JobSummary; full?: FullJob }) {
    const { toast, retryJob, loadJobIntoWorkbench } = useStore();
    const [log, setLog] = useState('');
    const [showLog, setShowLog] = useState(job.status === 'failed');
    const [now, setNow] = useState(Date.now());
    const [estimate, setEstimate] = useState<Estimate | null>(null);
    const running = job.status === 'running';

    useEffect(() => {
        setShowLog(job.status === 'failed');
    }, [job.id, job.status]);

    useEffect(() => {
        let alive = true;
        const load = () => api.jobLog(job.id).then(t => alive && setLog(t)).catch(e => alive && setLog(errorMessage(e)));
        void load();
        const t = window.setInterval(() => {
            setNow(Date.now());
            if (running && showLog) void load();
        }, 1000);
        return () => {
            alive = false;
            window.clearInterval(t);
        };
    }, [job.id, running, showLog]);

    // one estimate per job, from its full spec (the full job is refetched while running; estimate only once)
    const fullSpec = full?.kind === 'predict' && full.id === job.id ? full.spec : null;
    const estimating = useRef<string | null>(null);
    useEffect(() => {
        if (!fullSpec || !['queued', 'running'].includes(job.status) || estimating.current === job.id) return;
        estimating.current = job.id;
        setEstimate(null);
        api.estimate(fullSpec).then(e => estimating.current === job.id && setEstimate(e)).catch(() => undefined);
    }, [job.id, job.status, fullSpec]);

    const elapsed = job.started_at ? (job.finished_at ?? now / 1000) - job.started_at : 0;
    const phase = job.live.phase ?? job.phase ?? 'queued';
    const hasAffinity = job.kind === 'predict' && !!job.spec.affinity_binder;
    const steps = PHASE_STEPS.filter(s => s.key !== 'affinity' || hasAffinity);
    const phaseIndex = steps.findIndex(s => s.key === phase);
    const remaining = estimate && running ? estimate.seconds - elapsed : null;
    const kind = job.error_kind ?? 'error';
    const explained = ['oom', 'msa', 'download', 'input'].includes(kind);
    const hint = job.status === 'failed' && job.error ? (explained ? job.error.split('\n\n')[0] : job.error.split('\n')[0]) : null;

    return (
        <div className="panel results">
            <div className="results-head">
                <StatusDot status={job.status} />
                <strong className="ellipsis">{job.title}</strong>
                <span className="muted small">{job.status === 'queued' ? `${t('待機中')}${job.queue_position ? ` (${job.queue_position} ${t('番目')})` : ''}` : job.live.label ?? job.phase}</span>
                {job.started_at && <span className="small muted">{formatDuration(elapsed)}</span>}
                <span className="spacer" />
                {(job.status === 'running' || job.status === 'queued') && (
                    <Button size="sm" variant="danger" onClick={() => void api.cancelJob(job.id).catch(e => toast('error', errorMessage(e)))}>{t('キャンセル')}</Button>
                )}
            </div>
            <div className="results-body">
                {job.kind === 'predict' && job.status !== 'failed' && job.status !== 'cancelled' && (
                    <ol className="stepper" aria-label={t('進行状況')}>
                        {steps.map((s, i) => (
                            <li key={s.key} className={i < phaseIndex ? 'done' : i === phaseIndex ? 'current' : ''}>
                                <span className="step-dot">{i < phaseIndex ? <Icon name="check" size={11} /> : i + 1}</span>{s.label}
                            </li>
                        ))}
                    </ol>
                )}
                {running && (
                    <div className={`progress big ${typeof job.live.progress === 'number' ? '' : 'indeterminate'}`}>
                        <i style={typeof job.live.progress === 'number' ? { width: `${Math.round(job.live.progress * 100)}%` } : undefined} />
                    </div>
                )}
                {remaining !== null && (
                    <p className="small muted">
                        {remaining > 5 ? <>{t('残り約')} <strong>{formatDuration(remaining)}</strong> {t('(目安)')}</> : t('予想より時間がかかっています。ログで進み具合を確認できます')}
                        {phase === 'msa' && t(' · MSA サーバーが混んでいると数分かかることがあります')}
                    </p>
                )}
                {job.status === 'queued' && <p className="small muted">{t('前のジョブが終わると自動で始まります。予測は 1 件ずつ、変異スキャンは予測と並行して動きます。')}</p>}
                {job.status === 'failed' && (
                    <div className="failure">
                        <div className="failure-title"><Icon name="warning" /> {t('計算が失敗しました')}</div>
                        {hint && <p>{hint}</p>}
                        <div className="row wrap">
                            {kind === 'oom' && job.kind === 'predict' && (
                                <>
                                    <Button size="sm" variant="primary" onClick={() => void retryJob(job.id, { diffusion_samples: 1 })}>{t('サンプル 1 で再実行')}</Button>
                                    <Button size="sm" onClick={() => void retryJob(job.id, { accelerator: 'cpu' })}>{t('CPU で再実行 (遅い)')}</Button>
                                </>
                            )}
                            {kind === 'msa' && job.kind === 'predict' && (
                                <Button size="sm" variant="primary" onClick={() => void retryJob(job.id, { msa: 'single' })}>{t('MSA なしで再実行')}</Button>
                            )}
                            <Button size="sm" variant={['oom', 'msa'].includes(kind) ? 'default' : 'primary'} onClick={() => void retryJob(job.id)}>
                                <Icon name="retry" size={13} /> {t('そのまま再実行')}
                            </Button>
                            {job.kind === 'predict' && (
                                <Button size="sm" variant="ghost" onClick={() => void loadJobIntoWorkbench(job.id)}>{t('作業台に読み込んで直す')}</Button>
                            )}
                        </div>
                    </div>
                )}
                {job.status === 'cancelled' && (
                    <div className="row"><span className="muted">{t('キャンセルしました。')}</span><Button size="sm" onClick={() => void retryJob(job.id)}><Icon name="retry" size={13} /> {t('再実行')}</Button></div>
                )}
                <button type="button" className="link small" aria-expanded={showLog} onClick={() => setShowLog(s => !s)}>{showLog ? t('▾ ログを隠す') : t('▸ ログを表示')}</button>
                {showLog && <pre className="log">{log || t('ログはまだありません')}</pre>}
                {showLog && job.status === 'failed' && job.error && <pre className="error-box">{job.error}</pre>}
            </div>
        </div>
    );
}

function tone(v: number | undefined, good: number, ok: number): 'good' | 'ok' | 'bad' | undefined {
    if (typeof v !== 'number') return undefined;
    return v >= good ? 'good' : v >= ok ? 'ok' : 'bad';
}

/** A plain-language reading of the confidence numbers. */
/**
 * Whether the molecule still works, next to the number that says it got better.
 *
 * pLDDT knows nothing about function, so a variant can climb by deleting the parts that carry
 * it. The backend checks the mutations against what is known about this molecule (database
 * annotations, measured contacts, ESM-2 conservation) and against where the gain came from,
 * and anything it finds is said here rather than left in a log.
 */
function FunctionWarning({ risk }: { risk: FunctionRisk | null }) {
    if (!risk || risk.level === 'ok' || !risk.findings.length) return null;
    // Two different kinds of bad news share this box. One is about the molecule's job — a
    // mutation landed on a residue that carries function. The other is about the coordinates
    // themselves — atoms inside each other, a backbone that came apart, NaN. Saying
    // "機能を壊している" over a clash warning would send you looking in the wrong place.
    const GEOMETRY = ['nonfinite', 'clash', 'chain_break', 'empty'];
    const geometry = risk.findings.filter(f => GEOMETRY.includes(f.kind));
    const functional = risk.findings.filter(f => !GEOMETRY.includes(f.kind));
    const head = risk.level === 'danger' ? t('この結果は使えません')
        : functional.length && geometry.length ? t('構造と機能の両方に問題があります')
            : geometry.length ? t('構造そのものに無理があります')
                : t('機能を壊している可能性があります');
    return (
        <div className={`function-risk risk-${risk.level}`} role="alert">
            <div className="function-risk-head">
                <Icon name="warning" size={14} />
                {head}
            </div>
            <ul>
                {risk.findings.map((f, i) => <li key={i}>{f.text}</li>)}
            </ul>
            {functional.length > 0 && (
                <p className="small muted">
                    {t('スコアが上がっていても、機能を担う残基が置き換わっていれば別の分子になっています。')}
                    {t('残したい残基は「自律ループ」の禁止リストに入れてください。')}
                </p>
            )}
            {geometry.length > 0 && (
                <p className="small muted">
                    {t('信頼度スコアはこの種の破綻を検出しません。pLDDT が高いまま原子が重なっていることも、')}
                    {t('座標が NaN のまま自信満々に返ってくることもあります。')}
                </p>
            )}
        </div>
    );
}

/** The geometry block, or null for a result predicted before the check existed / that failed it. */
function geometryOf(r: PredictResult): Geometry | null {
    const g = r.geometry;
    return g && !('error' in g) ? g : null;
}

function verdict(r: PredictResult): { level: 'good' | 'ok' | 'bad'; lines: string[] } {
    const c: Confidence = r.models[0]?.confidence ?? {};
    const plddt = (c.complex_plddt ?? 0) * 100;
    const ptm = c.ptm ?? 0;
    const lines: string[] = [];
    let level: 'good' | 'ok' | 'bad';
    if (plddt >= 85 && ptm >= 0.8) {
        level = 'good';
        lines.push(t('全体の形はとても信頼できる予測です。'));
    } else if (plddt >= 70 && ptm >= 0.55) {
        level = 'ok';
        lines.push(t('全体の形はおおむね信頼できます。色の薄い (黄・橙) 部分は形が定まっていない可能性があります。'));
    } else {
        level = 'bad';
        lines.push(plddt >= 50 ? t('形は部分的にしか信頼できません。') : t('形はほとんど定まっていません。天然には存在しない配列や、単独ではほどけているタンパク質でよく起こります。'));
    }
    const polymers = r.chains.filter(ch => ch.type !== 'ligand');
    if (r.chains.length > 1 && typeof c.iptm === 'number') {
        if (c.iptm >= 0.8) lines.push(t('チェーン同士の組み合わさり方にも自信があります。'));
        else if (c.iptm >= 0.6) lines.push(t('チェーン同士の組み合わさり方はある程度もっともらしい、という程度です。'));
        else {
            lines.push(t('チェーン同士の組み合わさり方は当てになりません (実際には結合しない組み合わせかもしれません)。'));
            if (level === 'good') level = 'ok';
        }
    }
    const low = polymers.reduce((n, ch) => n + (r.models[0]?.plddt[ch.chain] ?? []).filter(v => v < 50).length, 0);
    if (low > 0) lines.push(`${t('pLDDT が 50 未満の残基が')} ${low} ${t('個あります (ほどけた領域の可能性)。')}`);
    if (r.affinity) {
        const p = r.affinity.affinity_probability_binary;
        lines.push(p >= 0.7 ? `${t('リガンドは結合すると予測されました (確率')} ${(p * 100).toFixed(0)}${t('%)。')}`
            : p >= 0.4 ? `${t('リガンドが結合するかは五分五分です (確率')} ${(p * 100).toFixed(0)}${t('%)。')}`
                : `${t('リガンドは結合しにくいと予測されました (確率')} ${(p * 100).toFixed(0)}${t('%)。')}`);
    }
    return { level, lines };
}

type PredictTab = 'summary' | 'plddt' | 'pae' | 'interfaces' | 'variants' | 'msa' | 'autopilot' | 'log';

function PredictResultView({ job }: { job: PredictJob }) {
    const store = useStore();
    const { jobs, setView, loadJobIntoWorkbench, compare, requestAssistant, toast, setSelectedResidue, workbench, exportJob } = store;
    const [tab, setTab] = useState<PredictTab>('summary');
    const [log, setLog] = useState<string | null>(null);
    const [msa, setMsa] = useState<MsaAlignment[] | null>(null);
    const r = job.result;
    const mutated = useMemo(() => {
        const out: Record<string, number[]> = {};
        (r?.chains ?? []).forEach(c => { out[c.chain] = c.mutations.map(m => Number(m.slice(1, -1))); });
        return out;
    }, [r]);
    const root = job.parent_id ? jobs.find(j => j.id === job.parent_id) : undefined;
    const familyRoot = root ?? jobs.find(j => j.id === job.id);
    const family = useMemo(() => familyRoot
        ? [familyRoot, ...jobs.filter(j => j.parent_id === familyRoot.id && j.kind === 'predict' && j.status !== 'cancelled')]
            .sort((a, b) => a.created_at - b.created_at)
        : [], [jobs, familyRoot]);

    useEffect(() => {
        setLog(null);
        setMsa(null);
        setTab(t => (t === 'variants' && family.length < 2 ? 'summary' : t));
    }, [job.id, family.length]);
    useEffect(() => {
        if (tab === 'log' && log === null) api.jobLog(job.id).then(setLog).catch(e => setLog(errorMessage(e)));
    }, [tab, log, job.id]);
    useEffect(() => {
        if (tab === 'msa' && msa === null) api.jobMsa(job.id).then(x => setMsa(x.alignments)).catch(() => setMsa([]));
    }, [tab, msa, job.id]);
    const polymerSeries = useMemo(() => (r ? r.chains.filter(x => x.type !== 'ligand').map(p => ({ chain: p.chain, values: r.models[0]?.plddt[p.chain] ?? [] })) : []), [r]);
    if (!r) return null;
    const m0 = r.models[0];
    const c = m0.confidence;
    const parent = job.parent_id ? jobs.find(j => j.id === job.parent_id && j.status === 'succeeded') : undefined;
    const chainIds = r.chains.map(x => x.chain);
    const interfaces = r.interfaces && 'interfaces' in r.interfaces ? r.interfaces.interfaces : null;
    // The same request answers two questions: is anything wrong with this result, and what
    // are its geometry numbers. Results stored before the check existed have the block
    // measured server-side on this first view, so it arrives here rather than in the job.
    const [risk, setRisk] = useState<FunctionRisk | null>(null);
    useEffect(() => {
        let alive = true;
        setRisk(null);
        api.functionRisk(job.id).then(x => { if (alive) setRisk(x); }).catch(() => { if (alive) setRisk(null); });
        return () => { alive = false; };
    }, [job.id]);
    const geom = geometryOf(r) ?? (risk?.geometry && !('error' in risk.geometry) ? risk.geometry : null);
    const matchesWorkbench = workbench.parentJobId === job.id;
    const v = verdict(r);

    return (
        <div className="panel results">
            <div className="results-head">
                <StatusDot status="succeeded" />
                <strong className="ellipsis" title={job.title}>{job.title}</strong>
                {job.origin === 'qwen' && <span className="qwen-badge">LLM</span>}
                <span className="muted small">{formatDuration(r.elapsed_sec)} · {r.accelerator.toUpperCase()}
                    {r.msa.reused_from ? t(' · MSA 再利用') : r.msa.server ? t(' · MSA サーバー') : ''}
                    {r.msa.single_sequence_chains.length ? ` ${t('· 単一配列')} ${r.msa.single_sequence_chains.join(',')}` : ''}</span>
                <span className="spacer" />
                <Tabs<PredictTab> small value={tab} onChange={setTab} tabs={[
                    { id: 'summary', label: t('概要') },
                    { id: 'plddt', label: 'pLDDT' },
                    { id: 'pae', label: 'PAE' },
                    { id: 'interfaces', label: t('界面'), badge: interfaces?.length || undefined },
                    ...(family.length > 1 ? [{ id: 'variants' as const, label: t('比較'), badge: family.length, title: t('同じ元から作った変異体を並べて比較') }] : []),
                    { id: 'msa', label: 'MSA', title: t('アラインメントの被覆率とサンプル配列') },
                    { id: 'autopilot', label: t('AI解析') },
                    { id: 'log', label: t('ログ') },
                ]} />
            </div>
            <div className="results-actions">
                <Button size="sm" variant="primary" onClick={() => setView({ kind: 'job', jobId: job.id, model: 0 })}>{t('3D で見る')}</Button>
                <Button size="sm" onClick={() => void loadJobIntoWorkbench(job.id)} disabled={matchesWorkbench}
                    title={t('この結果の入力を作業台に戻し、変異などを加えて再予測します')}>{matchesWorkbench ? t('作業台で編集中') : t('作業台で改変')}</Button>
                {parent && <Button size="sm" onClick={() => void compare(parent.id, job.id)}>{t('元の結果と重ねる')}</Button>}
                <Button size="sm" variant="qwen" onClick={() => requestAssistant('explain', '', job.id)}>{t('LLM に解説させる')}</Button>
                <span className="spacer" />
                <Menu label={<><Icon name="download" size={13} /> {t('書き出し')}</>} items={[
                    { label: t('まとめて zip で保存…'), hint: t('構造 (mmCIF/PDB)・スコア・入力・ログ'), onSelect: () => void exportJob(job.id, 'download') },
                    { label: t('構造を PDB 形式で保存…'), onSelect: () => void exportJob(job.id, 'pdb', 0) },
                    { label: t('メソッド記述を保存… (日本語)'), hint: t('論文の方法節向けにエンジン・パラメータ・版をまとめたテキスト'), onSelect: () => void exportJob(job.id, 'methods-ja') },
                    { label: t('メソッド記述を保存… (English)'), hint: t('エンジン・パラメータ・版をまとめた英語のメソッド記述'), onSelect: () => void exportJob(job.id, 'methods-en') },
                    'separator',
                    { label: t('ダウンロードフォルダに書き出して Finder で表示'), onSelect: () => void exportJob(job.id, 'folder') },
                    { label: t('ジョブのフォルダを Finder で開く'), onSelect: () => void api.revealJob(job.id).catch(e => toast('error', errorMessage(e))) },
                ]} />
            </div>
            <div className="results-body">
                {tab === 'summary' && (
                    <div className="summary">
                        <FunctionWarning risk={risk} />
                        <div className={`verdict verdict-${v.level}`}>
                            <Icon name={v.level === 'good' ? 'check' : v.level === 'ok' ? 'info' : 'warning'} />
                            <div>{v.lines.map((line, i) => <p key={i}>{line}</p>)}</div>
                        </div>
                        <div className="metrics">
                            <Metric label={t('総合信頼度')} term="confidence" value={fmt(c.confidence_score)} tone={tone(c.confidence_score, 0.8, 0.6)} />
                            <Metric label={t('平均 pLDDT')} term="plddt" value={fmt((c.complex_plddt ?? 0) * 100, 1)} tone={tone((c.complex_plddt ?? 0) * 100, 80, 60)} />
                            <Metric label="pTM" term="ptm" value={fmt(c.ptm)} tone={tone(c.ptm, 0.8, 0.5)} />
                            {chainIds.length > 1 && <Metric label="ipTM" term="iptm" value={fmt(c.iptm)} tone={tone(c.iptm, 0.8, 0.6)} />}
                            {geom && (
                                <Metric
                                    label={t('原子の衝突')} term="clashscore"
                                    value={geom.nonfinite_atoms ? 'NaN' : `${geom.clashes}`}
                                    hint={geom.nonfinite_atoms ? t('座標が数値になっていません')
                                        : `${t('1000 原子あたり')} ${geom.clashscore}${geom.severe_clashes ? ` ${t('/ 深い重なり')} ${geom.severe_clashes} ${t('箇所')}` : ''}`}
                                    tone={geom.nonfinite_atoms || geom.severe_clashes ? 'bad'
                                        : (geom.clashscore ?? 0) >= 10 ? 'ok' : 'good'} />
                            )}
                            {typeof c.complex_pae === 'number' && <Metric label={t('PAE 平均')} term="pae" value={`${fmt(c.complex_pae, 1)} Å`} />}
                        </div>
                        {r.affinity && (
                            <div className="affinity-card">
                                <div className="affinity-title">{t('結合親和性 (チェーン')} {r.affinity.binder})<InfoTip term="affinity" /></div>
                                <div className="metrics">
                                    <Metric label={t('結合する確率')} value={`${(r.affinity.affinity_probability_binary * 100).toFixed(0)}%`}
                                        tone={tone(r.affinity.affinity_probability_binary, 0.7, 0.4)} hint={t('結合分子 (binder) と非結合分子 (decoy) を区別する出力')} />
                                    <Metric label={t('IC50 予測')} value={r.affinity.ic50_um !== null && r.affinity.ic50_um !== undefined ? formatIc50(r.affinity.ic50_um) : '—'}
                                        hint={t('log10(IC50/µM) の予測値から換算。結合することが分かっている分子同士の比較に使う')} />
                                    <Metric label={t('ΔG 換算')} value={`${fmt(r.affinity.delta_g_kcal, 1)} kcal/mol`} hint="ΔG ≈ −1.363 × (6 − log10 IC50[µM])" />
                                    <Metric label="log10 IC50 (µM)" value={fmt(r.affinity.affinity_pred_value)} />
                                </div>
                            </div>
                        )}
                        {chainIds.length > 1 && c.pair_chains_iptm && (
                            <div className="table-wrap">
                                <table className="matrix">
                                    <thead><tr><th>ipTM</th>{chainIds.map(id => <th key={id}>{id}</th>)}</tr></thead>
                                    <tbody>
                                        {chainIds.map(a => (
                                            <tr key={a}><th>{a}</th>{chainIds.map(b => {
                                                const val = c.pair_chains_iptm?.[a]?.[b];
                                                return <td key={b} style={{ background: typeof val === 'number' ? `rgba(88,213,201,${val * 0.6})` : undefined }}>{fmt(val)}</td>;
                                            })}</tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                        <div className="table-wrap">
                            <table className="chains">
                                <thead><tr><th>{t('チェーン')}</th><th>{t('種類')}</th><th>{t('名前')}</th><th>{t('長さ')}</th><th>{t('平均 pLDDT')}</th><th>{t('変異')}</th></tr></thead>
                                <tbody>
                                    {r.chains.map(row => {
                                        const vals = m0.plddt[row.chain];
                                        const mean = vals?.length ? vals.reduce((a, b) => a + b, 0) / vals.length : m0.ligand_plddt[row.chain];
                                        return (
                                            <tr key={row.chain}>
                                                <td>{row.chain}</td><td>{row.type}</td><td>{row.label}</td>
                                                <td>{row.length ?? (row.ccd ? row.ccd : 'SMILES')}</td>
                                                <td>{fmt(mean, 1)}</td>
                                                <td className="mono">{row.mutations.join(' ')}</td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
                {tab === 'plddt' && (
                    <PlddtPlot series={polymerSeries} mutated={mutated}
                        onPick={(chain, pos) => {
                            setSelectedResidue({ chain, position: pos });
                            viewerBus.focus(chain, [pos]);
                        }} />
                )}
                {tab === 'pae' && (r.pae ? <PaeHeatmap matrix={r.pae.matrix} factor={r.pae.factor} segments={r.pae.segments} size={r.pae.size} /> : <Empty>{t('PAE がありません')}</Empty>)}
                {tab === 'interfaces' && (
                    r.interfaces && 'error' in r.interfaces ? <div className="warn">{r.interfaces.error}</div>
                        : interfaces && interfaces.length ? (
                            <div className="interfaces">
                                <p className="small muted">{t('4.5 Å 以内で接している残基。ポイントすると 3D で強調、クリックでその残基に寄ります。')}</p>
                                {interfaces.map(itf => (
                                    <div key={itf.chains.join('-')} className="interface">
                                        <div><strong>{itf.chains[0]} ⇄ {itf.chains[1]}</strong> <span className="muted small">{itf.contact_pairs} {t('残基対')}</span></div>
                                        {itf.chains.map(ch => (
                                            <div key={ch} className="res-chips">
                                                <span className="small muted">{ch}:</span>
                                                {itf.residues[ch].map(res => {
                                                    const num = Number(res.replace(/^\D+/, ''));
                                                    const isPolymer = r.chains.find(x => x.chain === ch)?.type !== 'ligand';
                                                    return (
                                                        <button type="button" key={res} className="chip" disabled={!isPolymer}
                                                            onMouseEnter={() => isPolymer && viewerBus.highlight(ch, [num])}
                                                            onMouseLeave={() => viewerBus.highlight(ch, [])}
                                                            onClick={() => { setSelectedResidue({ chain: ch, position: num }); viewerBus.focus(ch, [num]); }}>{res}</button>
                                                    );
                                                })}
                                            </div>
                                        ))}
                                    </div>
                                ))}
                            </div>
                        ) : <Empty>{r.chains.length > 1 ? t('チェーン間の接触はありません') : t('単一チェーンのため界面はありません')}</Empty>
                )}
                {tab === 'variants' && <VariantsTable family={family} current={job.id} />}
                {tab === 'msa' && (
                    msa === null ? <Empty>{t('読み込み中…')}</Empty>
                        : msa.length === 0 ? <Empty>{t('このジョブの MSA はありません (単一配列入力、または MSA ファイルが未保存)')}</Empty>
                            : <div>
                                {msa.map(a => <MsaSection key={a.file} a={a} />)}
                            </div>
                )}
                {tab === 'autopilot' && <AutopilotSection jobId={job.id} />}
                {tab === 'log' && <pre className="log">{log ?? t('読み込み中…')}</pre>}
            </div>
        </div>
    );
}

function AutopilotSection({ jobId }: { jobId: string }) {
    const { setThreadId } = useStore();
    const [result, setResult] = useState<AutopilotResult | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let alive = true;
        let timer: ReturnType<typeof setTimeout>;

        const poll = () => {
            api.autopilot(jobId)
                .then(r => { if (alive) { setResult(r); setLoading(false); } })
                .catch(e => {
                    if (alive) {
                        const msg = errorMessage(e);
                        if (msg.includes('not yet available')) {
                            // still running — poll again
                            timer = setTimeout(poll, 5000);
                        } else {
                            setError(msg);
                            setLoading(false);
                        }
                    }
                });
        };
        poll();
        return () => { alive = false; clearTimeout(timer); };
    }, [jobId]);

    if (loading) return (
        <div className="autopilot-section">
            <div className="autopilot-running">
                <Spinner /> {t('AI が自律解析を実行中です…')}
            </div>
        </div>
    );
    if (error) return <div className="warn pad">{error}</div>;
    if (!result) return null;

    const openThread = (tid: string | undefined) => {
        if (!tid) return;
        setThreadId(tid);
    };

    return (
        <div className="autopilot-section">
            {result.analyses.map((a, i) => (
                <div key={i} className="autopilot-card">
                    <div className="autopilot-card-head">
                        <span className="autopilot-badge">{a.mode === 'explain' ? t('解説') : `${t('変異提案 (チェーン')} ${a.chain ?? '?'})`}</span>
                        {a.elapsed_sec && <span className="small muted">{a.elapsed_sec.toFixed(1)}s</span>}
                        {a.thread_id && (
                            <button type="button" className="link small" onClick={() => openThread(a.thread_id!)}>
                                {t('LLM で続ける →')}
                            </button>
                        )}
                    </div>
                    {a.error ? (
                        <div className="warn small">{a.error}</div>
                    ) : (
                        <>
                            <p className="autopilot-reply">{a.reply}</p>
                            {a.reply_issues && a.reply_issues.length > 0 ? (
                                <div className="claim-warnings">
                                    <strong>{t('本文に配列と矛盾する記述があります')}</strong>
                                    <ul>{a.reply_issues.map((x, i) => <li key={i}>{x}</li>)}</ul>
                                </div>
                            ) : a.reply && <p className="unverified-note">{t('この文章は LLM が書いたもので検証されていません (照合・ESM-2 スコアリングを受けるのは下の案だけです)')}</p>}
                            {a.proposals && a.proposals.length > 0 && (
                                <div className="autopilot-proposals">
                                    {a.proposals.map((p, j) => (
                                        <div key={j} className={`proposal-chip ${p.status}`}>
                                            <span className="proposal-title">{p.title}</span>
                                            {p.status === 'ok' && p.apply && <span className="proposal-ok">{t('✓ 適用可能')}</span>}
                                            {p.status === 'warning' && <span className="proposal-warn">△</span>}
                                            {p.status === 'invalid' && <span className="proposal-bad">✗</span>}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </>
                    )}
                </div>
            ))}
            <p className="small muted" style={{ marginTop: 8 }}>
                {t('自動解析')} {result.elapsed_sec ? `· ${result.elapsed_sec.toFixed(0)} ${t('秒')}` : ''}
                {/* Autopilot analyses are not kept as chat threads — only a hand-started one links out. */}
                {result.analyses.some(a => a.thread_id) && <> · <button type="button" className="link small"
                    onClick={() => {
                        const tid = result.analyses.find(a => a.thread_id)?.thread_id;
                        if (tid) openThread(tid);
                    }}>{t('LLM パネルで詳しく見る')}</button></>}
            </p>
        </div>
    );
}


function VariantsTable({ family, current }: { family: JobSummary[]; current: string }) {
    const { openJob, compare } = useStore();
    const root = family[0];
    const rootPlddt = root.result?.mean_plddt ?? null;
    const rootAff = root.result?.affinity?.affinity_probability_binary ?? null;
    const muts = (j: JobSummary) => (j.spec.components ?? []).flatMap(c => (c.mutations ?? []).map(m => `${c.chains[0]}:${m}`));
    const delta = (value: number | null | undefined, base: number | null, digits = 1, scale = 1) => {
        if (typeof value !== 'number' || base === null) return null;
        const d = (value - base) * scale;
        if (Math.abs(d) < 10 ** -digits / 2) return <span className="muted">±0</span>;
        return <span className={d > 0 ? 'pos-text' : 'bad-text'}>{d > 0 ? '+' : ''}{d.toFixed(digits)}</span>;
    };
    return (
        <div className="stack">
            <p className="small muted">{t('「')}{root.title}{t('」から作った結果を並べています。Δ は元の結果との差 (プラス = 信頼度が上がった)。構造の変化は「重ねる」で確認できます。')}</p>
            <div className="table-wrap">
                <table className="variants">
                    <thead>
                        <tr><th>{t('名前')}</th><th>{t('変異')}</th><th>{t('平均 pLDDT')}</th><th>Δ</th><th>pTM</th><th>ipTM</th><th>{t('結合確率')}</th><th>Δ</th><th /></tr>
                    </thead>
                    <tbody>
                        {family.map(j => {
                            const conf = j.result?.confidence;
                            const aff = j.result?.affinity?.affinity_probability_binary;
                            return (
                                <tr key={j.id} className={j.id === current ? 'current' : ''}>
                                    <td className="ellipsis" title={j.title}><StatusDot status={j.status} /> {j.title}</td>
                                    <td className="mono small">{j.id === root.id ? t('元') : muts(j).join(' ') || '—'}</td>
                                    <td>{fmt(j.result?.mean_plddt, 1)}</td>
                                    <td>{j.id === root.id ? '' : delta(j.result?.mean_plddt, rootPlddt)}</td>
                                    <td>{fmt(conf?.ptm)}</td>
                                    <td>{typeof conf?.iptm === 'number' && conf.iptm > 0 ? fmt(conf.iptm) : '—'}</td>
                                    <td>{typeof aff === 'number' ? `${(aff * 100).toFixed(0)}%` : '—'}</td>
                                    <td>{j.id === root.id ? '' : delta(aff, rootAff, 0, 100)}</td>
                                    <td className="row">
                                        {j.status === 'succeeded' && <Button size="sm" variant="ghost" onClick={() => openJob(j.id)}>{t('表示')}</Button>}
                                        {j.status === 'succeeded' && j.id !== root.id && root.status === 'succeeded' && (
                                            <Button size="sm" variant="ghost" onClick={() => void compare(root.id, j.id)}>{t('重ねる')}</Button>
                                        )}
                                        {j.status !== 'succeeded' && <span className="small muted">{j.status === 'failed' ? t('失敗') : j.status === 'cancelled' ? t('中止') : t('計算中')}</span>}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function formatIc50(um: number): string {
    if (um < 1e-3) return `${(um * 1e6).toPrecision(2)} pM`;
    if (um < 1) return `${(um * 1e3).toPrecision(2)} nM`;
    if (um < 1000) return `${um.toPrecision(2)} µM`;
    return `${(um / 1000).toPrecision(2)} mM`;
}

function ScanResultView({ job }: { job: ScanJob }) {
    const { workbench, updateComponent, toast, requestAssistant, runVariants, exportJob, health } = useStore();
    const [batchSize, setBatchSize] = useState(5);
    const r = job.result;
    const chains = useMemo(() => assignChains(workbench.components), [workbench.components]);
    const target = r ? workbench.components.find(c => c.sequence === r.sequence) ?? workbench.components.find(c => c.baseSequence === r.sequence) : undefined;
    const mutated = useMemo(() => {
        const out = new Set<number>();
        if (target?.sequence && target.baseSequence) {
            [...target.sequence].forEach((ch, i) => { if (target.baseSequence && target.baseSequence[i] !== ch) out.add(i + 1); });
        }
        return out;
    }, [target?.sequence, target?.baseSequence]);
    if (!r) return null;
    const targetChain = target ? chains.get(target.uid)?.[0] : undefined;
    // variants are applied to the workbench sequence, which must still be the scanned one
    const canBatch = !!target && target.sequence === r.sequence && !!targetChain && !!health?.boltz.bin;
    const top = r.top_substitutions.filter(t => t.llr > 0);

    return (
        <div className="panel results">
            <div className="results-head">
                <StatusDot status="succeeded" />
                <strong className="ellipsis">{job.title}</strong>
                <span className="muted small">{r.model.split('/').pop()} {t('· 疑似パープレキシティ')} {r.pseudo_perplexity}<InfoTip term="pppl" /> · {r.elapsed_sec} {t('秒')}</span>
                <span className="spacer" />
                <Menu label={<Icon name="download" size={13} />} title={t('書き出し')} items={[
                    { label: t('スコア行列 (CSV) を含む zip を保存…'), onSelect: () => void exportJob(job.id, 'download') },
                    { label: t('メソッド記述を保存… (日本語)'), onSelect: () => void exportJob(job.id, 'methods-ja') },
                    { label: t('メソッド記述を保存… (English)'), onSelect: () => void exportJob(job.id, 'methods-en') },
                ]} />
            </div>
            <div className="results-actions">
                <label className="inline-toggle small">
                    {t('ESM の上位')}
                    <select value={batchSize} onChange={e => setBatchSize(Number(e.target.value))}>
                        {[3, 5, 10].map(n => <option key={n} value={n}>{n}</option>)}
                    </select>
                    {t('個の変異体を')}
                </label>
                <Button size="sm" variant="primary" disabled={!canBatch || top.length === 0}
                    title={!target ? t('この配列が作業台にありません') : target.sequence !== r.sequence ? t('作業台の配列が変更されています (変異を戻すと使えます)') : t('それぞれ 1 変異ずつ構造を予測して、元と比べます')}
                    onClick={() => targetChain && void runVariants(top.slice(0, batchSize).map(t => ({ chain: targetChain, mutations: [t.mutation] })))}>
                    {t('まとめて予測')}
                </Button>
                {target && <Button size="sm" variant="qwen" onClick={() => requestAssistant('mutations', '', null)}>{t('このスキャンを見て LLM に提案させる')}</Button>}
            </div>
            <div className="results-body">
                {!target && <div className="warn small">{t('この配列は作業台にありません (クリックしても適用されません)')}</div>}
                <ScanHeatmap sequence={r.sequence} matrix={r.matrix} highlight={mutated} onPick={mutation => {
                    if (!target?.sequence) return;
                    const pos = Number(mutation.slice(1, -1));
                    const wt = mutation[0];
                    if (target.sequence[pos - 1] !== wt && target.baseSequence?.[pos - 1] !== wt) {
                        toast('error', `${t('作業台の')} ${pos} ${t('番目はもう')} ${target.sequence[pos - 1]} ${t('に変わっています')}`);
                        return;
                    }
                    const chars = [...target.sequence];
                    chars[pos - 1] = mutation.slice(-1);
                    updateComponent(target.uid, { sequence: chars.join('') });
                    toast('info', `${targetChain ?? ''} ${t('に')} ${mutation} ${t('を入れました。「構造を予測する」で確かめられます')}`);
                }} />
                <div className="top-subs">
                    <div className="small muted">LLR<InfoTip term="llr" /> {t('が高い置換 (ESM-2 が自然と見なすもの)。安定化や機能向上を保証するものではありません。')}</div>
                    <div className="res-chips">
                        {r.top_substitutions.slice(0, 30).map(t => (
                            <span key={t.mutation} className="chip" onMouseEnter={() => targetChain && viewerBus.highlight(targetChain, [Number(t.mutation.slice(1, -1))])}
                                onMouseLeave={() => targetChain && viewerBus.highlight(targetChain, [])}>
                                {t.mutation} <small>{t.llr > 0 ? '+' : ''}{t.llr.toFixed(1)}</small>
                            </span>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

function RefineResultView({ job }: { job: RefineJob }) {
    const { addComponent, setWorkbench, runPrediction, toast, exportJob } = useStore();
    const r = job.result;
    if (!r) return null;
    const start = r.history[0]?.sequence ?? '';
    const newWorkbench = () => ({
        name: `${r.label ?? t('設計')} ${t('(ESM 改良)')}`,
        components: [{ uid: `c${Date.now().toString(36)}`, type: 'protein' as const, label: r.label ?? t('設計配列'), copies: 1,
            sequence: r.sequence, baseSequence: r.sequence, msa: 'single' as const, origin: 'qwen' as const }],
        affinityBinderUid: null,
        params: { diffusion_samples: 1, recycling_steps: 3, sampling_steps: 200, use_potentials: false, seed: null },
        parentJobId: null,
    });
    const changed = [...r.sequence].filter((ch, i) => start[i] !== ch).length;
    return (
        <div className="panel results">
            <div className="results-head">
                <StatusDot status="succeeded" />
                <strong className="ellipsis">{job.title}</strong>
                <span className="muted small">{t('疑似パープレキシティ')}<InfoTip term="pppl" /> {r.start_pseudo_perplexity} → {r.pseudo_perplexity} · {changed} {t('残基を変更')}</span>
                <span className="spacer" />
                <Button size="sm" onClick={() => {
                    addComponent({ type: 'protein', label: `${r.label ?? t('設計配列')} (ESM)`, sequence: r.sequence, msa: 'single' });
                    toast('success', t('作業台に追加しました'));
                }}>{t('作業台に追加')}</Button>
                <Button size="sm" variant="primary" onClick={() => {
                    const wb = newWorkbench();
                    setWorkbench(() => wb);
                    void runPrediction({ workbench: wb, title: wb.name, origin: 'qwen' });
                }}>{t('新しい作業台で予測')}</Button>
                <Menu label={<Icon name="download" size={13} />} title={t('書き出し')} items={[
                    { label: t('FASTA を含む zip を保存…'), onSelect: () => void exportJob(job.id, 'download') },
                    { label: t('メソッド記述を保存… (日本語)'), onSelect: () => void exportJob(job.id, 'methods-ja') },
                    { label: t('メソッド記述を保存… (English)'), onSelect: () => void exportJob(job.id, 'methods-en') },
                ]} />
            </div>
            <div className="results-body">
                <div className="mono seq-compare">
                    {[...r.sequence].map((ch, i) => <span key={i} className={start[i] !== ch ? 'diff' : ''} title={`${start[i]}${i + 1}${ch}`}>{ch}</span>)}
                </div>
                <table className="chains">
                    <thead><tr><th>{t('ラウンド')}</th><th>PPPL</th><th>{t('採用')}</th><th>{t('変更')}</th></tr></thead>
                    <tbody>
                        {r.history.map(h => (
                            <tr key={h.round}>
                                <td>{h.round}</td><td>{h.pseudo_perplexity}</td><td>{h.round === 0 ? '—' : h.accepted ? '✓' : '×'}</td>
                                <td className="mono small">{h.changed.join(' ')}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

/** One entity's alignment: coverage strip plus a few sample rows for eyeballing. */
function MsaSection({ a }: { a: MsaAlignment }) {
    const keyW = Math.max(4, ...a.sample.map(s => s.key.length)) + 1;
    const rows = [
        `${a.query_key.padEnd(keyW)} ${a.query}`,
        ...a.sample.map(s => `${s.key.padEnd(keyW)} ${s.seq}`),
    ];
    return (
        <section className="msa-section">
            <div className="msa-meta small muted">
                <span>{a.file}</span>
                <span>{t('クエリ')} {a.query.length} {t('残基')}</span>
                <span>{t('ヒット')} {a.depth.toLocaleString()} {t('件')}</span>
                <span>{t('アラインメント長')} {a.columns.toLocaleString()} {t('列')}</span>
            </div>
            <MsaCoverage coverage={a.coverage} identity={a.identity} />
            <pre className="msa-rows">{rows.join('\n')}</pre>
            <p className="small muted">{t('先頭行はクエリ配列。ヒットは先頭')} {a.sample.length} {t('件のみ表示しています。')}</p>
        </section>
    );
}
