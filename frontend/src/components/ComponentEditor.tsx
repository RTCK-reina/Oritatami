import { useEffect, useMemo, useState } from 'react';
import { api, errorMessage } from '../api';
import { useStore } from '../store';
import type { ChemInfo, ScanResult, WBComponent } from '../types';
import { AA_INFO, AMINO_ACIDS, applyMutationCodes, cleanSequence, llrColor, mutationsOf } from '../workbench';
import { SequenceStrip, type StripColor } from './SequenceStrip';
import { Button, Icon, InfoTip, Spinner, TYPE_LABEL } from './ui';

const chemCache = new Map<string, Promise<ChemInfo>>();
function describe(smiles: string): Promise<ChemInfo> {
    let p = chemCache.get(smiles);
    if (!p) {
        p = api.chemDescribe(smiles);
        chemCache.set(smiles, p);
        p.catch(() => chemCache.delete(smiles));
    }
    return p;
}

export function ComponentEditor({ comp, chains, plddt, onHoverResidue }: {
    comp: WBComponent;
    chains: string[];
    plddt?: number[];
    onHoverResidue: (chain: string, pos: number | null) => void;
}) {
    const store = useStore();
    const { updateComponent, removeComponent, workbench, setWorkbench, scanJobFor, scanComponent, jobCache, getJob, toast } = store;
    const [collapsed, setCollapsed] = useState(false);
    const [editing, setEditing] = useState(false);
    const [draft, setDraft] = useState(comp.sequence ?? '');
    const [mutInput, setMutInput] = useState('');
    const [color, setColor] = useState<StripColor>('group');
    const [chem, setChem] = useState<ChemInfo | null>(null);
    const [chemError, setChemError] = useState<string | null>(null);
    const isPolymer = comp.type !== 'ligand';
    const muts = mutationsOf(comp);
    const mutatedSet = useMemo(() => new Set(muts.codes.map(c => Number(c.slice(1, -1)))), [muts.codes]);
    const selected = store.selectedResidue && chains.includes(store.selectedResidue.chain) ? store.selectedResidue.position : null;

    const scanJob = comp.type === 'protein' ? scanJobFor(comp.sequence) ?? scanJobFor(comp.baseSequence) : undefined;
    const scanFull = scanJob ? jobCache[scanJob.id] : undefined;
    const scan: ScanResult | null = scanFull?.kind === 'scan' && scanFull.result ? scanFull.result : null;
    useEffect(() => {
        if (scanJob?.status === 'succeeded' && !scanFull) void getJob(scanJob.id);
    }, [scanJob?.id, scanJob?.status, scanFull, getJob]);

    useEffect(() => {
        if (comp.type !== 'ligand') return;
        let alive = true;
        setChemError(null);
        if (comp.smiles) {
            describe(comp.smiles).then(info => alive && setChem(info)).catch(e => alive && setChemError(errorMessage(e)));
        } else if (comp.ccd) {
            api.ccd(comp.ccd).then(info => {
                if (!alive) return;
                if (info.smiles) {
                    describe(info.smiles).then(d => alive && setChem({ ...d, svg: d.svg })).catch(e => alive && setChemError(errorMessage(e)));
                }
            }).catch(e => alive && setChemError(errorMessage(e)));
        }
        return () => { alive = false; };
    }, [comp.type, comp.smiles, comp.ccd]);

    const applyMutations = (codes: string[]) => {
        if (!comp.sequence) return;
        try {
            const sequence = applyMutationCodes(comp.sequence, codes);
            updateComponent(comp.uid, { sequence });
        } catch (e) {
            toast('error', errorMessage(e));
        }
    };

    const revertPosition = (pos: number) => {
        if (!comp.sequence || !comp.baseSequence) return;
        const chars = [...comp.sequence];
        chars[pos - 1] = comp.baseSequence[pos - 1];
        updateComponent(comp.uid, { sequence: chars.join('') });
    };

    const saveDraft = () => {
        if (comp.type === 'ligand') return;
        const { sequence, error } = cleanSequence(draft, comp.type);
        if (error) {
            toast('error', error);
            return;
        }
        updateComponent(comp.uid, { sequence, baseSequence: comp.baseSequence && comp.baseSequence.length === sequence.length ? comp.baseSequence : sequence });
        setEditing(false);
    };

    const isBinder = workbench.affinityBinderUid === comp.uid;

    return (
        <div className={`comp comp-${comp.type}`}>
            <div className="comp-head">
                <button type="button" className="icon-btn" onClick={() => setCollapsed(c => !c)} aria-label={collapsed ? '展開' : '折りたたむ'} aria-expanded={!collapsed}>{collapsed ? '▸' : '▾'}</button>
                <span className={`type-badge type-${comp.type}`}>{TYPE_LABEL[comp.type]}</span>
                <input className="comp-label" value={comp.label} onChange={e => updateComponent(comp.uid, { label: e.target.value })} />
                {comp.origin === 'qwen' && <span className="qwen-badge" title="LLM の提案から追加・変更">LLM</span>}
                <span className="chain-ids" title="チェーン ID">{chains.join(' ')}</span>
                <span className="copies">
                    <button className="icon-btn" disabled={comp.copies <= 1} onClick={() => updateComponent(comp.uid, { copies: comp.copies - 1 })}>−</button>
                    <span title="コピー数 (ホモ多量体)">×{comp.copies}</span>
                    <button className="icon-btn" disabled={comp.copies >= 12} onClick={() => updateComponent(comp.uid, { copies: comp.copies + 1 })}>+</button>
                </span>
                <button type="button" className="icon-btn danger" onClick={() => removeComponent(comp.uid)} aria-label="削除" title="作業台から外す"><Icon name="trash" size={14} /></button>
            </div>
            {!collapsed && isPolymer && comp.sequence && (
                <div className="comp-body">
                    <div className="comp-meta">
                        <span>{comp.sequence.length} 残基</span>
                        {comp.source && <span className="src">{comp.source.db} {comp.source.id}</span>}
                        {comp.type === 'protein' && (
                            <label className="inline-toggle" title="MSA: 進化情報を ColabFold サーバーから取得 (精度↑)。単一配列: 新規設計配列向け・速い">
                                <select value={comp.msa ?? 'server'} onChange={e => updateComponent(comp.uid, { msa: e.target.value as 'server' | 'single' })}>
                                    <option value="server">MSA あり</option>
                                    <option value="single">単一配列</option>
                                </select>
                                <InfoTip term="msa" />
                            </label>
                        )}
                        <label className="inline-toggle"
                            title="N 末端と C 末端をつないだ環状ポリマーとして予測します。直鎖の分子にこれを入れると、存在しない結合を作った構造を返します">
                            <input type="checkbox" checked={!!comp.cyclic} onChange={e => updateComponent(comp.uid, { cyclic: e.target.checked })} /> 環状
                        </label>
                        <span className="spacer" />
                        <select className="mini-select" value={color} onChange={e => setColor(e.target.value as StripColor)} title="配列の色">
                            <option value="group">性質で色分け</option>
                            <option value="plddt" disabled={!plddt}>pLDDT</option>
                            <option value="tolerance" disabled={!scan}>ESM 許容度</option>
                            <option value="none">色なし</option>
                        </select>
                    </div>
                    {comp.cyclic && (
                        <p className="opt-warning" role="status">
                            環状としてこの鎖の両末端をつなぎます。環状ペプチドや環状化した設計配列でないかぎり外してください
                            — 直鎖の分子に付けると、実在しない末端間の結合を前提にした構造が返り、pLDDT もその前提のまま高く出ます。
                        </p>
                    )}
                    {editing ? (
                        <div className="seq-edit">
                            <textarea value={draft} onChange={e => setDraft(e.target.value)} rows={5} spellCheck={false} />
                            <div className="row">
                                <Button size="sm" variant="primary" onClick={saveDraft}>保存</Button>
                                <Button size="sm" variant="ghost" onClick={() => setEditing(false)}>やめる</Button>
                                <span className="hint">長さを変えると、変異の基準配列もこの配列に置き換わります</span>
                            </div>
                        </div>
                    ) : (
                        <SequenceStrip sequence={comp.sequence} mutated={mutatedSet} selected={selected} color={color}
                            plddt={plddt} tolerance={scan?.sequence === comp.sequence ? scan.position_tolerance : undefined}
                            onClick={pos => store.setSelectedResidue(selected === pos ? null : { chain: chains[0], position: pos })}
                            onHover={pos => onHoverResidue(chains[0], pos)} />
                    )}
                    {comp.type === 'protein' && selected && comp.sequence[selected - 1] && (
                        <MutationPicker
                            wt={comp.sequence[selected - 1]} base={comp.baseSequence?.[selected - 1]} position={selected}
                            scan={scan}
                            onPick={aa => {
                                const current = comp.sequence?.[selected - 1];
                                if (!current || aa === current) return;
                                applyMutations([`${current}${selected}${aa}`]);
                            }}
                            onClose={() => store.setSelectedResidue(null)} />
                    )}
                    {comp.type === 'protein' && (
                        <div className="mut-bar">
                            {muts.lengthChanged && <span className="warn">長さが基準配列と違います</span>}
                            {muts.codes.map(code => (
                                <span key={code} className="chip chip-mut">
                                    {code}
                                    <button onClick={() => revertPosition(Number(code.slice(1, -1)))} aria-label={`${code} を戻す`}>×</button>
                                </span>
                            ))}
                            <form className="mut-form" onSubmit={e => {
                                e.preventDefault();
                                const codes = mutInput.split(/[\s,;/+]+/).filter(Boolean);
                                if (codes.length) applyMutations(codes);
                                setMutInput('');
                            }}>
                                <input value={mutInput} onChange={e => setMutInput(e.target.value)} placeholder="変異を入力 (例 K48R L73P)" />
                            </form>
                        </div>
                    )}
                    <div className="row wrap">
                        <Button size="sm" variant="ghost" onClick={() => { setDraft(comp.sequence ?? ''); setEditing(true); }}>配列を編集</Button>
                        {comp.type === 'protein' && muts.codes.length > 0 && (
                            <>
                                <Button size="sm" variant="ghost" onClick={() => updateComponent(comp.uid, { sequence: comp.baseSequence })}>変異をすべて戻す</Button>
                                <Button size="sm" variant="ghost" onClick={() => updateComponent(comp.uid, { baseSequence: comp.sequence })}>この配列を基準にする</Button>
                            </>
                        )}
                        {comp.type === 'protein' && (
                            <Button size="sm" onClick={() => void scanComponent(comp.uid)} disabled={scanJob?.status === 'running' || scanJob?.status === 'queued'}
                                title="ESM-2 で全ての 1 残基置換を採点し、変異の手がかりにします">
                                {scanJob?.status === 'running' || scanJob?.status === 'queued' ? <><Spinner size={10} /> スキャン中</> : scan ? '変異スキャン結果' : '変異スキャン (ESM-2)'}
                            </Button>
                        )}
                        {comp.type === 'protein' && (
                            <Button size="sm" variant="ghost" title="ESM-2 で不自然な残基を置き換え、より天然らしい配列に近づけるジョブを追加します"
                                onClick={() => void api.submitRefine({ sequence: comp.sequence ?? '', label: comp.label })
                                    .then(() => { toast('info', 'ESM で磨くジョブを追加しました', { label: 'ジョブを見る', run: () => store.showLeftTab('jobs') }); void store.refreshJobs(); })
                                    .catch(e => toast('error', errorMessage(e)))}>ESM で磨く</Button>
                        )}
                        <Button size="sm" variant="ghost" onClick={() => {
                            const data: Record<string, unknown> = { ...comp };
                            delete data.uid;
                            void api.addLibrary(comp.type, comp.label || comp.type, data)
                                .then(() => toast('success', 'ライブラリに保存しました'))
                                .catch(e => toast('error', errorMessage(e)));
                        }}>ライブラリへ保存</Button>
                    </div>
                </div>
            )}
            {!collapsed && comp.type === 'ligand' && (
                <div className="comp-body ligand-body">
                    <div className="ligand-svg">
                        {chem?.svg ? <img alt={comp.label} src={`data:image/svg+xml;utf8,${encodeURIComponent(chem.svg)}`} /> : chemError ? <span className="warn">{chemError}</span> : <Spinner />}
                    </div>
                    <div className="ligand-info">
                        <div className="mono small">{comp.smiles ? comp.smiles : `CCD: ${comp.ccd}`}</div>
                        {chem && (
                            <div className="kv">
                                <span>{chem.formula}</span><span>MW {chem.molecular_weight}</span><span>重原子 {chem.heavy_atoms}</span>
                                <span>logP {chem.logp}</span>
                            </div>
                        )}
                        <label className="inline-toggle" title="Boltz-2 の親和性ヘッドでこのリガンドの結合強さを予測します (1 分子のみ)">
                            <input type="radio" name="affinity-binder" checked={isBinder}
                                disabled={comp.copies > 1}
                                onChange={() => setWorkbench(wb => ({ ...wb, affinityBinderUid: comp.uid }))} />
                            結合親和性を予測する
                            {isBinder && <button className="link" onClick={() => setWorkbench(wb => ({ ...wb, affinityBinderUid: null }))}>解除</button>}
                        </label>
                        {chem && !chem.affinity_ok && isBinder && <div className="warn small">重原子 56 超または複数分子のため、親和性の信頼性が下がります</div>}
                    </div>
                </div>
            )}
        </div>
    );
}

function MutationPicker({ wt, base, position, scan, onPick, onClose }: {
    wt: string; base?: string; position: number; scan: ScanResult | null; onPick: (aa: string) => void; onClose: () => void;
}) {
    const row = scan?.matrix[position - 1];
    return (
        <div className="mut-picker">
            <div className="mut-picker-head">
                <strong>{wt}{position}</strong> {AA_INFO[wt]?.name}
                {base && base !== wt && <span className="small muted">(基準 {base})</span>}
                {row ? <span className="small muted">数字は ESM-2 LLR (スキャンした配列の {scan?.sequence[position - 1]}{position} に対する値。+ ほど自然)</span> : <span className="small muted">変異スキャンを実行するとスコアが出ます</span>}
                <span className="spacer" />
                <button className="icon-btn" onClick={onClose}>×</button>
            </div>
            <div className="mut-picker-grid">
                {[...AMINO_ACIDS].map((aa, j) => (
                    <button key={aa} className={`aa-btn ${aa === wt ? 'current' : ''} ${aa === base ? 'base' : ''}`}
                        style={row ? { background: llrColor(row[j]) } : undefined}
                        title={`${AA_INFO[aa].name} (${AA_INFO[aa].group})${row ? ` LLR ${row[j].toFixed(2)}` : ''}`}
                        onClick={() => onPick(aa)}>
                        <span>{aa}</span>
                        {row && <small>{row[j] > 0 ? '+' : ''}{row[j].toFixed(1)}</small>}
                    </button>
                ))}
            </div>
        </div>
    );
}
