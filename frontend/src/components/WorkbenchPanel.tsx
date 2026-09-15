import { useEffect, useMemo, useState } from 'react';
import { api, errorMessage } from '../api';
import { useDebounced } from '../hooks';
import { useStore } from '../store';
import type { Estimate } from '../types';
import { uiEvents, type AddTab } from '../uiEvents';
import { viewerBus } from '../viewer/bus';
import { assignChains, buildSpec, formatDuration, mutationsOf } from '../workbench';
import { AddDialog } from './AddDialog';
import { ComponentEditor } from './ComponentEditor';
import { Welcome } from './Welcome';
import { Button, Field, Icon, InfoTip, Kbd, Spinner } from './ui';

export function WorkbenchPanel() {
    const store = useStore();
    const { workbench, setWorkbench, jobs, jobCache, selectedJobId, runPrediction, toast, health } = store;
    const [dialog, setDialog] = useState<AddTab | null>(null);
    const [showParams, setShowParams] = useState(false);
    const [confirmClear, setConfirmClear] = useState(false);
    const [submitting, setSubmitting] = useState(false);
    const chains = useMemo(() => assignChains(workbench.components), [workbench.components]);

    useEffect(() => uiEvents.on('openAdd', tab => setDialog(tab)), []);

    // pLDDT for a component if the selected result predicted the same sequence
    const selected = selectedJobId ? jobCache[selectedJobId] : undefined;
    const plddtFor = (seq: string | undefined, compChains: string[]): number[] | undefined => {
        if (!selected || selected.kind !== 'predict' || !selected.result || !seq) return undefined;
        const row = selected.result.chains.find(c => c.sequence === seq && compChains.includes(c.chain))
            ?? selected.result.chains.find(c => c.sequence === seq);
        return row ? selected.result.models[0]?.plddt[row.chain] : undefined;
    };

    const running = jobs.filter(j => j.kind === 'predict' && (j.status === 'running' || j.status === 'queued')).length;
    const parent = workbench.parentJobId ? jobs.find(j => j.id === workbench.parentJobId) : undefined;
    const totalMutations = workbench.components.reduce((n, c) => n + mutationsOf(c).codes.length, 0);
    const p = workbench.params;
    const setParam = <K extends keyof typeof p>(key: K, value: (typeof p)[K]) =>
        setWorkbench(wb => ({ ...wb, params: { ...wb.params, [key]: value } }));

    const submit = async () => {
        if (submitting) return;
        setSubmitting(true);
        await runPrediction();
        setSubmitting(false);
    };

    return (
        <div className="panel workbench">
            <div className="panel-head">
                <input className="wb-name" aria-label="作業台の名前" value={workbench.name} onChange={e => setWorkbench(wb => ({ ...wb, name: e.target.value }))} />
                <Button size="sm" variant="ghost" title="作業台をライブラリに保存 (あとで「＋ 追加 → ライブラリ」から開けます)" disabled={!workbench.components.length} onClick={() => {
                    void api.addLibrary('workbench', workbench.name, { name: workbench.name, components: workbench.components })
                        .then(() => toast('success', '作業台をライブラリに保存しました'))
                        .catch(e => toast('error', errorMessage(e)));
                }}>保存</Button>
                <Button size="sm" variant="ghost" title="保存した作業台・分子をライブラリから開く"
                    onClick={() => setDialog('library')}>開く</Button>
                <Button size="sm" variant={confirmClear ? 'danger' : 'ghost'} onBlur={() => setConfirmClear(false)} disabled={!workbench.components.length} onClick={() => {
                    if (!confirmClear) {
                        setConfirmClear(true);
                        return;
                    }
                    setWorkbench(() => ({ name: '新しい作業台', components: [], affinityBinderUid: null, params: workbench.params, parentJobId: null }));
                    setConfirmClear(false);
                }}>{confirmClear ? '本当に空にする' : '新規'}</Button>
            </div>
            {parent && (
                <div className="parent-line small">
                    元の結果: <button type="button" className="link" onClick={() => store.openJob(parent.id)}>{parent.title}</button>
                    {totalMutations > 0 && <span className="chip chip-mut">変異 {totalMutations}</span>}
                </div>
            )}
            <div className="panel-scroll">
                {workbench.components.length === 0 ? <Welcome onAdd={setDialog} /> : (
                    <>
                        {workbench.components.map(c => (
                            <ComponentEditor key={c.uid} comp={c} chains={chains.get(c.uid) ?? []} plddt={plddtFor(c.sequence, chains.get(c.uid) ?? [])}
                                onHoverResidue={(chain, pos) => viewerBus.highlight(chain, pos ? [pos] : [])} />
                        ))}
                        <div className="add-row">
                            <Button size="sm" onClick={() => setDialog('uniprot')}>＋ タンパク質</Button>
                            <Button size="sm" onClick={() => setDialog('ligand')}>＋ リガンド・薬</Button>
                            <Button size="sm" onClick={() => setDialog('paste')}>＋ 配列を貼る</Button>
                            <Button size="sm" onClick={() => setDialog('pdb')}>＋ PDB から</Button>
                            <Button size="sm" variant="ghost" onClick={() => setDialog('library')} title="保存した作業台・分子を開く">ライブラリ</Button>
                        </div>
                    </>
                )}
            </div>
            {workbench.components.length > 0 && (
                <div className="run-box">
                    <button type="button" className="link small" aria-expanded={showParams} onClick={() => setShowParams(s => !s)}>{showParams ? '▾' : '▸'} 予測の設定</button>
                    {showParams && (
                        <div className="params">
                            <Field label="サンプル数" hint="多いほど別の形の候補が出る (時間は増える)">
                                <input type="number" min={1} max={10} value={p.diffusion_samples} onChange={e => setParam('diffusion_samples', clampInt(e.target.value, 1, 10))} />
                            </Field>
                            <Field label="リサイクル" hint="構造を練り直す回数">
                                <input type="number" min={1} max={10} value={p.recycling_steps} onChange={e => setParam('recycling_steps', clampInt(e.target.value, 1, 10))} />
                            </Field>
                            <Field label="拡散ステップ" hint="少ないほど速いが粗い">
                                <input type="number" min={10} max={500} step={10} value={p.sampling_steps} onChange={e => setParam('sampling_steps', clampInt(e.target.value, 10, 500))} />
                            </Field>
                            <Field label="シード" hint="空欄でランダム。同じ値なら同じ結果">
                                <input type="number" value={p.seed ?? ''} onChange={e => setParam('seed', e.target.value === '' ? null : Math.trunc(Number(e.target.value)))} />
                            </Field>
                            <Field label="計算デバイス" hint="通常は自動 (GPU)。メモリ不足時は CPU">
                                <select value={p.accelerator ?? 'auto'} onChange={e => setParam('accelerator', e.target.value as 'auto' | 'mps' | 'cpu')}>
                                    <option value="auto">自動</option>
                                    <option value="mps">GPU (MPS)</option>
                                    <option value="cpu">CPU (遅い)</option>
                                </select>
                            </Field>
                            <label className="inline-toggle" title="推論時ポテンシャルで原子の衝突や結合長の破綻を減らす (少し遅くなる)">
                                <input type="checkbox" checked={p.use_potentials} onChange={e => setParam('use_potentials', e.target.checked)} /> 物理的な補正
                            </label>
                        </div>
                    )}
                    <EstimateLine />
                    {!health?.boltz.bin && (
                        <p className="blocked-note" role="status">
                            <Icon name="warning" size={13} />
                            <span>
                                Boltz-2 がインストールされていないので予測を実行できません。
                                {' '}<button type="button" className="link" onClick={() => uiEvents.emit('openSettings')}>設定 → 準備状況</button>
                                {' '}から導入してください。
                            </span>
                        </p>
                    )}
                    <Button variant="primary" className="run-btn" disabled={submitting || !health?.boltz.bin} onClick={() => void submit()}
                        title={health?.boltz.bin ? '構造予測をキューに追加' : 'Boltz-2 がインストールされていません (設定 → 準備状況)'}>
                        {submitting ? <Spinner /> : <>構造を予測する <Kbd combo="mod+enter" /></>}
                        {running > 0 && <span className="small"> (実行中・待機 {running})</span>}
                    </Button>
                </div>
            )}
            {dialog && <AddDialog initialTab={dialog} onClose={() => setDialog(null)} />}
        </div>
    );
}

function clampInt(raw: string, lo: number, hi: number): number {
    const v = Math.trunc(Number(raw));
    return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : lo;
}

/** Runtime / memory estimate for the current workbench, refreshed as it changes. */
function EstimateLine() {
    const { workbench, jobs } = useStore();
    // re-estimate when the queue changes too (jobs ahead, and finished jobs refine the calibration)
    const queueKey = jobs.filter(j => j.kind === 'predict').map(j => `${j.id}:${j.status}`).slice(0, 30).join(',');
    const spec = useMemo(() => JSON.stringify(buildSpec(workbench)), [workbench]);
    const debounced = useDebounced(`${spec}\u0000${queueKey}`, 450);
    const [est, setEst] = useState<Estimate | null>(null);
    const [problem, setProblem] = useState<string | null>(null);

    useEffect(() => {
        let alive = true;
        api.estimate(JSON.parse(debounced.split('\u0000')[0]))
            .then(r => { if (alive) { setEst(r); setProblem(null); } })
            .catch(e => { if (alive) { setEst(null); setProblem(errorMessage(e)); } });
        return () => { alive = false; };
    }, [debounced]);

    if (problem) return <div className="estimate estimate-problem small" role="status"><span className="warn">入力を確認してください: {problem}</span></div>;
    if (!est) return null;
    const range = `${formatDuration(est.low)}〜${formatDuration(est.high)}`;
    return (
        <div className={`estimate small mem-${est.memory.level}`} role="status">
            <span title={`起動 ${est.breakdown.startup}s · MSA ${est.breakdown.msa}s · 構造 ${est.breakdown.structure}s · 親和性 ${est.breakdown.affinity}s`}>
                目安 <strong>{formatDuration(est.seconds)}</strong> <span className="muted">({range}{est.basis === 'history' ? `、過去 ${est.samples} 件の実績から` : '、実績が増えると精度が上がります'})</span>
            </span>
            <span className="muted">{est.tokens} トークン<InfoTip term="tokens" />{est.msa_reuse ? ' · MSA 再利用' : est.needs_msa_search ? ' · MSA 検索あり' : ''}{est.queued_ahead ? ` · 先に ${est.queued_ahead} 件` : ''}</span>
            {est.memory.level !== 'ok' && (
                <span
                    className={est.memory.level === 'danger' ? 'bad-text' : 'warn'}
                    title={est.memory.basis === 'history'
                        ? `実測 ${est.memory.samples} 件から当てはめた見込みです`
                        : '実測がまだ足りないため既定のモデルによる見込みです。予測を走らせるほど正確になります'}
                >
                    メモリ目安 {est.memory.peak_gb} GB / {est.memory.total_gb} GB
                    {est.memory.basis === 'history' ? `（実測 ${est.memory.samples} 件）` : '（実測前の概算）'}
                    {' — '}
                    {est.memory.level === 'danger'
                        ? <>
                            {est.memory.beyond_physical && !est.memory.applecare && (
                                <strong className="bad-text">保証なしで回すな — </strong>
                            )}
                            {est.memory.beyond_physical
                                ? '搭載メモリを超えるので走行中ずっとスワップし、遅くなるうえ SSD を消耗させます（止まりはしません）。'
                                : '物理メモリを超えるとスワップに落ちて実測で約 300 倍遅くなります（止まりはしません）。'}
                            構成要素・コピー数を減らすか、長い配列をドメインに切り出してください
                        </>
                        : '余裕が少なめです。他のアプリを閉じると安定します'}
                </span>
            )}
        </div>
    );
}
