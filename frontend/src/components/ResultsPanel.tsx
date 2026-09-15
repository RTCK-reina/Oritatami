import { useEffect, useMemo, useRef, useState } from 'react';
import { api, errorMessage } from '../api';
import type { AutopilotResult, FunctionRisk } from '../api';
import { useStore } from '../store';
import type { Confidence, Estimate, FullJob, JobSummary, PredictJob, PredictResult, RefineJob, ScanJob } from '../types';
import { viewerBus } from '../viewer/bus';
import { assignChains, fmt, formatDuration } from '../workbench';
import { PaeHeatmap, PlddtPlot, ScanHeatmap } from './charts';
import { Button, Empty, Icon, InfoTip, Menu, Metric, Spinner, StatusDot, Tabs } from './ui';

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
                    <p>ジョブを選ぶと、ここに結果 (信頼度・グラフ・界面など) が出ます。</p>
                    <p className="small muted">計算が終わると自動でここに表示されます。</p>
                </Empty>
            </div>
        );
    }
    if (!summary) return <div className="panel results"><Empty>このジョブは削除されました</Empty></div>;
    if (error) return <div className="panel results"><div className="warn pad">{error}</div></div>;
    if (summary.status !== 'succeeded') return <JobProgress job={summary} full={current} />;
    if (!current || current.status !== 'succeeded') return <div className="panel results"><Empty><Spinner /></Empty></div>;
    if (current.kind === 'predict') return <PredictResultView job={current} />;
    if (current.kind === 'scan') return <ScanResultView job={current} />;
    return <RefineResultView job={current} />;
}

const PHASE_STEPS: { key: string; label: string }[] = [
    { key: 'starting', label: '起動' },
    { key: 'preprocess', label: '前処理' },
    { key: 'msa', label: 'MSA 検索' },
    { key: 'structure', label: '構造予測' },
    { key: 'affinity', label: '親和性' },
    { key: 'collect', label: '集計' },
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
                <span className="muted small">{job.status === 'queued' ? `待機中${job.queue_position ? ` (${job.queue_position} 番目)` : ''}` : job.live.label ?? job.phase}</span>
                {job.started_at && <span className="small muted">{formatDuration(elapsed)}</span>}
                <span className="spacer" />
                {(job.status === 'running' || job.status === 'queued') && (
                    <Button size="sm" variant="danger" onClick={() => void api.cancelJob(job.id).catch(e => toast('error', errorMessage(e)))}>キャンセル</Button>
                )}
            </div>
            <div className="results-body">
                {job.kind === 'predict' && job.status !== 'failed' && job.status !== 'cancelled' && (
                    <ol className="stepper" aria-label="進行状況">
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
                        {remaining > 5 ? <>残り約 <strong>{formatDuration(remaining)}</strong> (目安)</> : '予想より時間がかかっています。ログで進み具合を確認できます'}
                        {phase === 'msa' && ' · MSA サーバーが混んでいると数分かかることがあります'}
                    </p>
                )}
                {job.status === 'queued' && <p className="small muted">前のジョブが終わると自動で始まります。予測は 1 件ずつ、変異スキャンは予測と並行して動きます。</p>}
                {job.status === 'failed' && (
                    <div className="failure">
                        <div className="failure-title"><Icon name="warning" /> 計算が失敗しました</div>
                        {hint && <p>{hint}</p>}
                        <div className="row wrap">
                            {kind === 'oom' && job.kind === 'predict' && (
                                <>
                                    <Button size="sm" variant="primary" onClick={() => void retryJob(job.id, { diffusion_samples: 1 })}>サンプル 1 で再実行</Button>
                                    <Button size="sm" onClick={() => void retryJob(job.id, { accelerator: 'cpu' })}>CPU で再実行 (遅い)</Button>
                                </>
                            )}
                            {kind === 'msa' && job.kind === 'predict' && (
                                <Button size="sm" variant="primary" onClick={() => void retryJob(job.id, { msa: 'single' })}>MSA なしで再実行</Button>
                            )}
                            <Button size="sm" variant={['oom', 'msa'].includes(kind) ? 'default' : 'primary'} onClick={() => void retryJob(job.id)}>
                                <Icon name="retry" size={13} /> そのまま再実行
                            </Button>
                            {job.kind === 'predict' && (
                                <Button size="sm" variant="ghost" onClick={() => void loadJobIntoWorkbench(job.id)}>作業台に読み込んで直す</Button>
                            )}
                        </div>
                    </div>
                )}
                {job.status === 'cancelled' && (
                    <div className="row"><span className="muted">キャンセルしました。</span><Button size="sm" onClick={() => void retryJob(job.id)}><Icon name="retry" size={13} /> 再実行</Button></div>
                )}
                <button type="button" className="link small" aria-expanded={showLog} onClick={() => setShowLog(s => !s)}>{showLog ? '▾ ログを隠す' : '▸ ログを表示'}</button>
                {showLog && <pre className="log">{log || 'ログはまだありません'}</pre>}
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
function FunctionWarning({ jobId }: { jobId: string }) {
    const [risk, setRisk] = useState<FunctionRisk | null>(null);
    useEffect(() => {
        let alive = true;
        setRisk(null);
        api.functionRisk(jobId).then(r => { if (alive) setRisk(r); }).catch(() => { if (alive) setRisk(null); });
        return () => { alive = false; };
    }, [jobId]);
    if (!risk || risk.level === 'ok' || !risk.findings.length) return null;
    return (
        <div className={`function-risk risk-${risk.level}`} role="alert">
            <div className="function-risk-head">
                <Icon name="warning" size={14} />
                {risk.level === 'danger' ? 'この結果は使えない可能性があります' : '機能を壊している可能性があります'}
            </div>
            <ul>
                {risk.findings.map((f, i) => <li key={i}>{f.text}</li>)}
            </ul>
            <p className="small muted">
                スコアが上がっていても、機能を担う残基が置き換わっていれば別の分子になっています。
                残したい残基は「自律ループ」の禁止リストに入れてください。
            </p>
        </div>
    );
}

function verdict(r: PredictResult): { level: 'good' | 'ok' | 'bad'; lines: string[] } {
    const c: Confidence = r.models[0]?.confidence ?? {};
    const plddt = (c.complex_plddt ?? 0) * 100;
    const ptm = c.ptm ?? 0;
    const lines: string[] = [];
    let level: 'good' | 'ok' | 'bad';
    if (plddt >= 85 && ptm >= 0.8) {
        level = 'good';
        lines.push('全体の形はとても信頼できる予測です。');
    } else if (plddt >= 70 && ptm >= 0.55) {
        level = 'ok';
        lines.push('全体の形はおおむね信頼できます。色の薄い (黄・橙) 部分は形が定まっていない可能性があります。');
    } else {
        level = 'bad';
        lines.push(plddt >= 50 ? '形は部分的にしか信頼できません。' : '形はほとんど定まっていません。天然には存在しない配列や、単独ではほどけているタンパク質でよく起こります。');
    }
    const polymers = r.chains.filter(ch => ch.type !== 'ligand');
    if (r.chains.length > 1 && typeof c.iptm === 'number') {
        if (c.iptm >= 0.8) lines.push('チェーン同士の組み合わさり方にも自信があります。');
        else if (c.iptm >= 0.6) lines.push('チェーン同士の組み合わさり方はある程度もっともらしい、という程度です。');
        else {
            lines.push('チェーン同士の組み合わさり方は当てになりません (実際には結合しない組み合わせかもしれません)。');
            if (level === 'good') level = 'ok';
        }
    }
    const low = polymers.reduce((n, ch) => n + (r.models[0]?.plddt[ch.chain] ?? []).filter(v => v < 50).length, 0);
    if (low > 0) lines.push(`pLDDT が 50 未満の残基が ${low} 個あります (ほどけた領域の可能性)。`);
    if (r.affinity) {
        const p = r.affinity.affinity_probability_binary;
        lines.push(p >= 0.7 ? `リガンドは結合すると予測されました (確率 ${(p * 100).toFixed(0)}%)。`
            : p >= 0.4 ? `リガンドが結合するかは五分五分です (確率 ${(p * 100).toFixed(0)}%)。`
                : `リガンドは結合しにくいと予測されました (確率 ${(p * 100).toFixed(0)}%)。`);
    }
    return { level, lines };
}

type PredictTab = 'summary' | 'plddt' | 'pae' | 'interfaces' | 'variants' | 'autopilot' | 'log';

function PredictResultView({ job }: { job: PredictJob }) {
    const store = useStore();
    const { jobs, setView, loadJobIntoWorkbench, compare, requestAssistant, toast, setSelectedResidue, workbench, exportJob } = store;
    const [tab, setTab] = useState<PredictTab>('summary');
    const [log, setLog] = useState<string | null>(null);
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
        setTab(t => (t === 'variants' && family.length < 2 ? 'summary' : t));
    }, [job.id, family.length]);
    useEffect(() => {
        if (tab === 'log' && log === null) api.jobLog(job.id).then(setLog).catch(e => setLog(errorMessage(e)));
    }, [tab, log, job.id]);
    const polymerSeries = useMemo(() => (r ? r.chains.filter(x => x.type !== 'ligand').map(p => ({ chain: p.chain, values: r.models[0]?.plddt[p.chain] ?? [] })) : []), [r]);
    if (!r) return null;
    const m0 = r.models[0];
    const c = m0.confidence;
    const parent = job.parent_id ? jobs.find(j => j.id === job.parent_id && j.status === 'succeeded') : undefined;
    const chainIds = r.chains.map(x => x.chain);
    const interfaces = r.interfaces && 'interfaces' in r.interfaces ? r.interfaces.interfaces : null;
    const matchesWorkbench = workbench.parentJobId === job.id;
    const v = verdict(r);

    return (
        <div className="panel results">
            <div className="results-head">
                <StatusDot status="succeeded" />
                <strong className="ellipsis" title={job.title}>{job.title}</strong>
                {job.origin === 'qwen' && <span className="qwen-badge">LLM</span>}
                <span className="muted small">{formatDuration(r.elapsed_sec)} · {r.accelerator.toUpperCase()}
                    {r.msa.reused_from ? ' · MSA 再利用' : r.msa.server ? ' · MSA サーバー' : ''}
                    {r.msa.single_sequence_chains.length ? ` · 単一配列 ${r.msa.single_sequence_chains.join(',')}` : ''}</span>
                <span className="spacer" />
                <Tabs<PredictTab> small value={tab} onChange={setTab} tabs={[
                    { id: 'summary', label: '概要' },
                    { id: 'plddt', label: 'pLDDT' },
                    { id: 'pae', label: 'PAE' },
                    { id: 'interfaces', label: '界面', badge: interfaces?.length || undefined },
                    ...(family.length > 1 ? [{ id: 'variants' as const, label: '比較', badge: family.length, title: '同じ元から作った変異体を並べて比較' }] : []),
                    { id: 'autopilot', label: 'AI解析' },
                    { id: 'log', label: 'ログ' },
                ]} />
            </div>
            <div className="results-actions">
                <Button size="sm" variant="primary" onClick={() => setView({ kind: 'job', jobId: job.id, model: 0 })}>3D で見る</Button>
                <Button size="sm" onClick={() => void loadJobIntoWorkbench(job.id)} disabled={matchesWorkbench}
                    title="この結果の入力を作業台に戻し、変異などを加えて再予測します">{matchesWorkbench ? '作業台で編集中' : '作業台で改変'}</Button>
                {parent && <Button size="sm" onClick={() => void compare(parent.id, job.id)}>元の結果と重ねる</Button>}
                <Button size="sm" variant="qwen" onClick={() => requestAssistant('explain', '', job.id)}>LLM に解説させる</Button>
                <span className="spacer" />
                <Menu label={<><Icon name="download" size={13} /> 書き出し</>} items={[
                    { label: 'まとめて zip で保存…', hint: '構造 (mmCIF/PDB)・スコア・入力・ログ', onSelect: () => void exportJob(job.id, 'download') },
                    { label: '構造を PDB 形式で保存…', onSelect: () => void exportJob(job.id, 'pdb', 0) },
                    'separator',
                    { label: 'ダウンロードフォルダに書き出して Finder で表示', onSelect: () => void exportJob(job.id, 'folder') },
                    { label: 'ジョブのフォルダを Finder で開く', onSelect: () => void api.revealJob(job.id).catch(e => toast('error', errorMessage(e))) },
                ]} />
            </div>
            <div className="results-body">
                {tab === 'summary' && (
                    <div className="summary">
                        <FunctionWarning jobId={job.id} />
                        <div className={`verdict verdict-${v.level}`}>
                            <Icon name={v.level === 'good' ? 'check' : v.level === 'ok' ? 'info' : 'warning'} />
                            <div>{v.lines.map((line, i) => <p key={i}>{line}</p>)}</div>
                        </div>
                        <div className="metrics">
                            <Metric label="総合信頼度" term="confidence" value={fmt(c.confidence_score)} tone={tone(c.confidence_score, 0.8, 0.6)} />
                            <Metric label="平均 pLDDT" term="plddt" value={fmt((c.complex_plddt ?? 0) * 100, 1)} tone={tone((c.complex_plddt ?? 0) * 100, 80, 60)} />
                            <Metric label="pTM" term="ptm" value={fmt(c.ptm)} tone={tone(c.ptm, 0.8, 0.5)} />
                            {chainIds.length > 1 && <Metric label="ipTM" term="iptm" value={fmt(c.iptm)} tone={tone(c.iptm, 0.8, 0.6)} />}
                            {typeof c.complex_pae === 'number' && <Metric label="PAE 平均" term="pae" value={`${fmt(c.complex_pae, 1)} Å`} />}
                        </div>
                        {r.affinity && (
                            <div className="affinity-card">
                                <div className="affinity-title">結合親和性 (チェーン {r.affinity.binder})<InfoTip term="affinity" /></div>
                                <div className="metrics">
                                    <Metric label="結合する確率" value={`${(r.affinity.affinity_probability_binary * 100).toFixed(0)}%`}
                                        tone={tone(r.affinity.affinity_probability_binary, 0.7, 0.4)} hint="結合分子 (binder) と非結合分子 (decoy) を区別する出力" />
                                    <Metric label="IC50 予測" value={r.affinity.ic50_um !== null && r.affinity.ic50_um !== undefined ? formatIc50(r.affinity.ic50_um) : '—'}
                                        hint="log10(IC50/µM) の予測値から換算。結合することが分かっている分子同士の比較に使う" />
                                    <Metric label="ΔG 換算" value={`${fmt(r.affinity.delta_g_kcal, 1)} kcal/mol`} hint="ΔG ≈ −1.363 × (6 − log10 IC50[µM])" />
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
                                <thead><tr><th>チェーン</th><th>種類</th><th>名前</th><th>長さ</th><th>平均 pLDDT</th><th>変異</th></tr></thead>
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
                {tab === 'pae' && (r.pae ? <PaeHeatmap matrix={r.pae.matrix} factor={r.pae.factor} segments={r.pae.segments} size={r.pae.size} /> : <Empty>PAE がありません</Empty>)}
                {tab === 'interfaces' && (
                    r.interfaces && 'error' in r.interfaces ? <div className="warn">{r.interfaces.error}</div>
                        : interfaces && interfaces.length ? (
                            <div className="interfaces">
                                <p className="small muted">4.5 Å 以内で接している残基。ポイントすると 3D で強調、クリックでその残基に寄ります。</p>
                                {interfaces.map(itf => (
                                    <div key={itf.chains.join('-')} className="interface">
                                        <div><strong>{itf.chains[0]} ⇄ {itf.chains[1]}</strong> <span className="muted small">{itf.contact_pairs} 残基対</span></div>
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
                        ) : <Empty>{r.chains.length > 1 ? 'チェーン間の接触はありません' : '単一チェーンのため界面はありません'}</Empty>
                )}
                {tab === 'variants' && <VariantsTable family={family} current={job.id} />}
                {tab === 'autopilot' && <AutopilotSection jobId={job.id} />}
                {tab === 'log' && <pre className="log">{log ?? '読み込み中…'}</pre>}
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
                <Spinner /> AI が自律解析を実行中です…
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
                        <span className="autopilot-badge">{a.mode === 'explain' ? '解説' : `変異提案 (チェーン ${a.chain ?? '?'})`}</span>
                        {a.elapsed_sec && <span className="small muted">{a.elapsed_sec.toFixed(1)}s</span>}
                        {a.thread_id && (
                            <button type="button" className="link small" onClick={() => openThread(a.thread_id!)}>
                                LLM で続ける →
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
                                    <strong>本文に配列と矛盾する記述があります</strong>
                                    <ul>{a.reply_issues.map((x, i) => <li key={i}>{x}</li>)}</ul>
                                </div>
                            ) : a.reply && <p className="unverified-note">この文章は LLM が書いたもので検証されていません (照合・ESM-2 スコアリングを受けるのは下の案だけです)</p>}
                            {a.proposals && a.proposals.length > 0 && (
                                <div className="autopilot-proposals">
                                    {a.proposals.map((p, j) => (
                                        <div key={j} className={`proposal-chip ${p.status}`}>
                                            <span className="proposal-title">{p.title}</span>
                                            {p.status === 'ok' && p.apply && <span className="proposal-ok">✓ 適用可能</span>}
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
                自動解析 {result.elapsed_sec ? `· ${result.elapsed_sec.toFixed(0)} 秒` : ''}
                {/* Autopilot analyses are not kept as chat threads — only a hand-started one links out. */}
                {result.analyses.some(a => a.thread_id) && <> · <button type="button" className="link small"
                    onClick={() => {
                        const tid = result.analyses.find(a => a.thread_id)?.thread_id;
                        if (tid) openThread(tid);
                    }}>LLM パネルで詳しく見る</button></>}
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
            <p className="small muted">「{root.title}」から作った結果を並べています。Δ は元の結果との差 (プラス = 信頼度が上がった)。構造の変化は「重ねる」で確認できます。</p>
            <div className="table-wrap">
                <table className="variants">
                    <thead>
                        <tr><th>名前</th><th>変異</th><th>平均 pLDDT</th><th>Δ</th><th>pTM</th><th>ipTM</th><th>結合確率</th><th>Δ</th><th /></tr>
                    </thead>
                    <tbody>
                        {family.map(j => {
                            const conf = j.result?.confidence;
                            const aff = j.result?.affinity?.affinity_probability_binary;
                            return (
                                <tr key={j.id} className={j.id === current ? 'current' : ''}>
                                    <td className="ellipsis" title={j.title}><StatusDot status={j.status} /> {j.title}</td>
                                    <td className="mono small">{j.id === root.id ? '元' : muts(j).join(' ') || '—'}</td>
                                    <td>{fmt(j.result?.mean_plddt, 1)}</td>
                                    <td>{j.id === root.id ? '' : delta(j.result?.mean_plddt, rootPlddt)}</td>
                                    <td>{fmt(conf?.ptm)}</td>
                                    <td>{typeof conf?.iptm === 'number' && conf.iptm > 0 ? fmt(conf.iptm) : '—'}</td>
                                    <td>{typeof aff === 'number' ? `${(aff * 100).toFixed(0)}%` : '—'}</td>
                                    <td>{j.id === root.id ? '' : delta(aff, rootAff, 0, 100)}</td>
                                    <td className="row">
                                        {j.status === 'succeeded' && <Button size="sm" variant="ghost" onClick={() => openJob(j.id)}>表示</Button>}
                                        {j.status === 'succeeded' && j.id !== root.id && root.status === 'succeeded' && (
                                            <Button size="sm" variant="ghost" onClick={() => void compare(root.id, j.id)}>重ねる</Button>
                                        )}
                                        {j.status !== 'succeeded' && <span className="small muted">{j.status === 'failed' ? '失敗' : j.status === 'cancelled' ? '中止' : '計算中'}</span>}
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
                <span className="muted small">{r.model.split('/').pop()} · 疑似パープレキシティ {r.pseudo_perplexity}<InfoTip term="pppl" /> · {r.elapsed_sec} 秒</span>
                <span className="spacer" />
                <Menu label={<Icon name="download" size={13} />} title="書き出し" items={[
                    { label: 'スコア行列 (CSV) を含む zip を保存…', onSelect: () => void exportJob(job.id, 'download') },
                ]} />
            </div>
            <div className="results-actions">
                <label className="inline-toggle small">
                    ESM の上位
                    <select value={batchSize} onChange={e => setBatchSize(Number(e.target.value))}>
                        {[3, 5, 10].map(n => <option key={n} value={n}>{n}</option>)}
                    </select>
                    個の変異体を
                </label>
                <Button size="sm" variant="primary" disabled={!canBatch || top.length === 0}
                    title={!target ? 'この配列が作業台にありません' : target.sequence !== r.sequence ? '作業台の配列が変更されています (変異を戻すと使えます)' : 'それぞれ 1 変異ずつ構造を予測して、元と比べます'}
                    onClick={() => targetChain && void runVariants(top.slice(0, batchSize).map(t => ({ chain: targetChain, mutations: [t.mutation] })))}>
                    まとめて予測
                </Button>
                {target && <Button size="sm" variant="qwen" onClick={() => requestAssistant('mutations', '', null)}>このスキャンを見て LLM に提案させる</Button>}
            </div>
            <div className="results-body">
                {!target && <div className="warn small">この配列は作業台にありません (クリックしても適用されません)</div>}
                <ScanHeatmap sequence={r.sequence} matrix={r.matrix} highlight={mutated} onPick={mutation => {
                    if (!target?.sequence) return;
                    const pos = Number(mutation.slice(1, -1));
                    const wt = mutation[0];
                    if (target.sequence[pos - 1] !== wt && target.baseSequence?.[pos - 1] !== wt) {
                        toast('error', `作業台の ${pos} 番目はもう ${target.sequence[pos - 1]} に変わっています`);
                        return;
                    }
                    const chars = [...target.sequence];
                    chars[pos - 1] = mutation.slice(-1);
                    updateComponent(target.uid, { sequence: chars.join('') });
                    toast('info', `${targetChain ?? ''} に ${mutation} を入れました。「構造を予測する」で確かめられます`);
                }} />
                <div className="top-subs">
                    <div className="small muted">LLR<InfoTip term="llr" /> が高い置換 (ESM-2 が自然と見なすもの)。安定化や機能向上を保証するものではありません。</div>
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
        name: `${r.label ?? '設計'} (ESM 改良)`,
        components: [{ uid: `c${Date.now().toString(36)}`, type: 'protein' as const, label: r.label ?? '設計配列', copies: 1,
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
                <span className="muted small">疑似パープレキシティ<InfoTip term="pppl" /> {r.start_pseudo_perplexity} → {r.pseudo_perplexity} · {changed} 残基を変更</span>
                <span className="spacer" />
                <Button size="sm" onClick={() => {
                    addComponent({ type: 'protein', label: `${r.label ?? '設計配列'} (ESM)`, sequence: r.sequence, msa: 'single' });
                    toast('success', '作業台に追加しました');
                }}>作業台に追加</Button>
                <Button size="sm" variant="primary" onClick={() => {
                    const wb = newWorkbench();
                    setWorkbench(() => wb);
                    void runPrediction({ workbench: wb, title: wb.name, origin: 'qwen' });
                }}>新しい作業台で予測</Button>
                <Menu label={<Icon name="download" size={13} />} title="書き出し" items={[
                    { label: 'FASTA を含む zip を保存…', onSelect: () => void exportJob(job.id, 'download') },
                ]} />
            </div>
            <div className="results-body">
                <div className="mono seq-compare">
                    {[...r.sequence].map((ch, i) => <span key={i} className={start[i] !== ch ? 'diff' : ''} title={`${start[i]}${i + 1}${ch}`}>{ch}</span>)}
                </div>
                <table className="chains">
                    <thead><tr><th>ラウンド</th><th>PPPL</th><th>採用</th><th>変更</th></tr></thead>
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
