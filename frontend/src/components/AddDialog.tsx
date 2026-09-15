import { useEffect, useState } from 'react';
import { api, errorMessage } from '../api';
import { useStore } from '../store';
import type { ChemInfo, ComponentType, ImportedStructure, LibraryItem, PolymerType, SearchRecord, UniProtHit, WBComponent, Workbench } from '../types';
import { cleanSequence, defaultParams, newUid } from '../workbench';
import { Button, Empty, Field, Modal, Spinner, Tabs, TYPE_LABEL } from './ui';

type Tab = 'uniprot' | 'pdb' | 'paste' | 'ligand' | 'file' | 'library' | 'history';

export function AddDialog({ onClose, initialTab = 'uniprot' }: { onClose: () => void; initialTab?: Tab }) {
    const [tab, setTab] = useState<Tab>(initialTab);
    return (
        <Modal title="構成要素を追加" onClose={onClose} wide>
            <Tabs<Tab> value={tab} onChange={setTab} tabs={[
                { id: 'uniprot', label: 'UniProt 検索' },
                { id: 'pdb', label: 'PDB / AlphaFold DB' },
                { id: 'paste', label: '配列を貼る' },
                { id: 'ligand', label: 'リガンド・薬' },
                { id: 'file', label: '構造ファイル' },
                { id: 'library', label: 'ライブラリ' },
                { id: 'history', label: '検索履歴' },
            ]} />
            <div className="tab-body">
                {tab === 'uniprot' && <UniProtTab onDone={onClose} />}
                {tab === 'pdb' && <PdbTab onDone={onClose} />}
                {tab === 'paste' && <PasteTab onDone={onClose} />}
                {tab === 'ligand' && <LigandTab onDone={onClose} />}
                {tab === 'file' && <FileTab onDone={onClose} />}
                {tab === 'library' && <LibraryTab onDone={onClose} />}
                {tab === 'history' && <HistoryTab onDone={onClose} />}
            </div>
        </Modal>
    );
}

function UniProtTab({ onDone }: { onDone: () => void }) {
    const { addComponent, addImport, setView, toast } = useStore();
    const [text, setText] = useState('');
    const [hits, setHits] = useState<UniProtHit[] | null>(null);
    const [busy, setBusy] = useState<string | null>(null);

    const search = async () => {
        setBusy('search');
        try {
            setHits(await api.uniprotSearch(text));
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setBusy(null);
        }
    };

    const add = async (hit: UniProtHit) => {
        setBusy(hit.accession);
        try {
            const entry = await api.uniprotEntry(hit.accession);
            addComponent({ type: 'protein', label: entry.name.slice(0, 60), sequence: entry.sequence, msa: 'server',
                source: { db: 'UniProt', id: entry.accession } });
            toast('success', `${entry.name} を追加しました`);
            onDone();
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setBusy(null);
        }
    };

    const viewAfdb = async (hit: UniProtHit) => {
        setBusy(`afdb-${hit.accession}`);
        try {
            const item = await api.importAfdb(hit.accession);
            addImport(item);
            setView({ kind: 'import', item });
            onDone();
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setBusy(null);
        }
    };

    return (
        <div className="stack">
            <form className="row" onSubmit={e => { e.preventDefault(); void search(); }}>
                <input autoFocus className="grow" value={text} onChange={e => setText(e.target.value)}
                    placeholder="タンパク質名・遺伝子名・アクセッション (例: ubiquitin human, P00918, GFP)" />
                <Button type="submit" variant="primary" disabled={!text.trim() || busy === 'search'}>{busy === 'search' ? <Spinner /> : '検索'}</Button>
            </form>
            {hits === null ? <Empty>UniProt から配列を取り込みます。★ はレビュー済み (Swiss-Prot) のエントリです。</Empty>
                : hits.length === 0 ? <Empty>見つかりませんでした</Empty> : (
                    <div className="result-list">
                        {hits.map(h => (
                            <div className="result-item" key={h.accession}>
                                <div className="grow">
                                    <div className="result-title">{h.reviewed && <span className="reviewed-star">★</span>} {h.name}</div>
                                    <div className="small muted">{h.accession} · {h.organism} · {h.length} 残基 {h.genes.length ? `· ${h.genes.join(', ')}` : ''}</div>
                                </div>
                                <Button size="sm" variant="ghost" onClick={() => void viewAfdb(h)} disabled={!!busy}>
                                    {busy === `afdb-${h.accession}` ? <Spinner /> : 'AFDB 構造を見る'}
                                </Button>
                                <Button size="sm" variant="primary" onClick={() => void add(h)} disabled={!!busy}>
                                    {busy === h.accession ? <Spinner /> : '作業台に追加'}
                                </Button>
                            </div>
                        ))}
                    </div>
                )}
        </div>
    );
}

function StructurePicker({ item, onDone }: { item: ImportedStructure; onDone: () => void }) {
    const { addComponent, setView, addImport, toast } = useStore();
    const { chains, ligands, warnings } = item.summary;
    // A virus assembly arrives as 180 chains. Ticking them all would put a 60-copy capsid
    // on the workbench, which is a prediction nobody wants and no machine can run — so for
    // a big assembly only the first copy of each distinct sequence starts selected.
    const many = chains.length > 12;
    const [picked, setPicked] = useState<Set<string>>(() => {
        const seen = new Set<string>();
        const keep = chains.filter(c => {
            if (!many) return true;
            const key = `${c.kind}:${c.sequence}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
        return new Set([
            ...keep.map(c => `p:${c.chain}`),
            ...(many ? [] : ligands.filter(l => !l.additive).map(l => `l:${l.chain}:${l.seqid}`)),
        ]);
    });
    const toggle = (key: string) => setPicked(s => {
        const n = new Set(s);
        if (n.has(key)) n.delete(key);
        else n.add(key);
        return n;
    });

    const addPicked = () => {
        // group identical polymer sequences into one component with copies
        const groups = new Map<string, { kind: PolymerType; sequence: string; chains: string[] }>();
        for (const c of chains) {
            if (!picked.has(`p:${c.chain}`)) continue;
            const key = `${c.kind}:${c.sequence}`;
            const g = groups.get(key) ?? { kind: c.kind, sequence: c.sequence, chains: [] };
            g.chains.push(c.chain);
            groups.set(key, g);
        }
        let n = 0;
        for (const g of groups.values()) {
            addComponent({ type: g.kind, label: `${item.source.id} ${g.chains.join('')}`, sequence: g.sequence,
                copies: g.chains.length, msa: g.kind === 'protein' ? 'server' : undefined,
                source: item.source.db === 'AFDB' ? { db: 'UniProt', id: item.source.id } : item.source });
            n++;
        }
        const ligGroups = new Map<string, number>();
        for (const l of ligands) {
            if (!picked.has(`l:${l.chain}:${l.seqid}`)) continue;
            ligGroups.set(l.ccd, (ligGroups.get(l.ccd) ?? 0) + 1);
        }
        for (const [ccd, count] of ligGroups) {
            addComponent({ type: 'ligand', label: ccd, ccd, copies: count });
            n++;
        }
        toast('success', `${n} 個の構成要素を追加しました`);
        onDone();
    };

    return (
        <div className="stack">
            <div className="row">
                <strong className="grow">{item.title}</strong>
                <Button size="sm" variant="ghost" onClick={() => { addImport(item); setView({ kind: 'import', item }); onDone(); }}>ビューアで見る</Button>
            </div>
            {many && (
                <div className="small muted">
                    {chains.length} 本のチェーンがあります。作業台に入れるぶんは、同じ配列ごとに 1 本だけ選んであります
                    (全体を見るだけなら「ビューアで見る」)。
                </div>
            )}
            {warnings.map(w => <div key={w} className="warn small">{w}</div>)}
            <div className="pick-list">
                {chains.map(c => (
                    <label key={c.chain} className="pick">
                        <input type="checkbox" checked={picked.has(`p:${c.chain}`)} onChange={() => toggle(`p:${c.chain}`)} />
                        <span className={`type-badge type-${c.kind}`}>{TYPE_LABEL[c.kind]}</span>
                        チェーン {c.chain} · {c.sequence.length} 残基 <span className="muted small">(観測 {c.observed_residues})</span>
                    </label>
                ))}
                {ligands.map(l => (
                    <label key={`${l.chain}:${l.seqid}`} className="pick">
                        <input type="checkbox" checked={picked.has(`l:${l.chain}:${l.seqid}`)} onChange={() => toggle(`l:${l.chain}:${l.seqid}`)} />
                        <span className="type-badge type-ligand">リガンド</span>
                        {l.ccd} <span className="muted small">チェーン {l.chain} · {l.atoms} 原子 {l.additive ? '· 結晶化添加物の可能性' : ''}</span>
                    </label>
                ))}
            </div>
            <div className="row">
                <Button variant="primary" onClick={addPicked} disabled={picked.size === 0}>選択したものを作業台に追加</Button>
                <span className="hint">同じ配列のチェーンはコピー数としてまとめます</span>
            </div>
        </div>
    );
}

/** Whole particles, verified against the RCSB assembly files (sizes measured 2026-09-14). */
const VIRUS_PICKS: [string, string, string][] = [
    ['1A34', 'サテライトタバコモザイクウイルス', '17 nm・最小級のウイルス・25 MB'],
    ['2MS2', 'MS2 バクテリオファージ', '大腸菌に感染する RNA ファージ・22 MB'],
    ['1CWP', 'カウピー退緑斑紋ウイルス', '植物ウイルス・27 MB'],
    ['4RHV', 'ヒトライノウイルス 14', '風邪のウイルス・46 MB'],
    ['1HXS', 'ポリオウイルス (Mahoney 株)', '53 MB'],
    ['6VXX', 'SARS-CoV-2 スパイク三量体', 'ウイルス全体ではなく突起 1 本・3 MB'],
];

function PdbTab({ onDone }: { onDone: () => void }) {
    const { toast } = useStore();
    const [id, setId] = useState('');
    const [afdb, setAfdb] = useState('');
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<{ id: string; title: string }[] | null>(null);
    const [item, setItem] = useState<ImportedStructure | null>(null);
    const [busy, setBusy] = useState(false);
    // Off by default: for an ordinary protein the assembly is the same file, and for a
    // capsid it is 60x larger.
    const [assembly, setAssembly] = useState(false);

    const run = async (fn: () => Promise<void>) => {
        setBusy(true);
        try {
            await fn();
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setBusy(false);
        }
    };

    const open = (pdbId: string) => run(async () => {
        const got = await api.importPdb(pdbId, assembly);
        if (got.note) toast('info', got.note);
        setItem(got);
    });

    if (item) return <StructurePicker item={item} onDone={onDone} />;
    return (
        <div className="stack">
            <label className="inline-toggle" title="結晶やクライオ EM の登録データは、対称性のある粒子だと殻の一部だけが入っています。生物学的単位は対称操作を展開した「粒子まるごと」で、ウイルスのカプシドならこちらを選ばないと破片しか見えません。ふつうのタンパク質では中身は変わりません">
                <input type="checkbox" checked={assembly} onChange={e => setAssembly(e.target.checked)} />
                生物学的単位で取り込む (ウイルス粒子など、対称操作を展開した全体)
            </label>
            {assembly && (
                <div className="quick-picks">
                    <span className="small muted">まるごと見られるウイルス</span>
                    {VIRUS_PICKS.map(([pid, name, note]) => (
                        <button type="button" key={pid} className="chip" disabled={busy}
                            title={`${pid} — ${note}`} onClick={() => void open(pid)}>{name}</button>
                    ))}
                </div>
            )}
            <form className="row" onSubmit={e => { e.preventDefault(); void open(id); }}>
                <Field label="PDB ID"><input value={id} onChange={e => setId(e.target.value)} placeholder="例: 1CRN, 3HS4" maxLength={4} /></Field>
                <Button type="submit" variant="primary" disabled={id.trim().length !== 4 || busy}>取得</Button>
            </form>
            <form className="row" onSubmit={e => { e.preventDefault(); void run(async () => setItem(await api.importAfdb(afdb))); }}>
                <Field label="AlphaFold DB (UniProt アクセッション)"><input value={afdb} onChange={e => setAfdb(e.target.value.toUpperCase())} placeholder="例: P69905" /></Field>
                <Button type="submit" disabled={!afdb.trim() || busy}>取得</Button>
            </form>
            <form className="row" onSubmit={e => { e.preventDefault(); void run(async () => setResults(await api.pdbSearch(query))); }}>
                <input className="grow" value={query} onChange={e => setQuery(e.target.value)} placeholder="PDB を全文検索 (例: carbonic anhydrase acetazolamide)" />
                <Button type="submit" disabled={!query.trim() || busy}>{busy ? <Spinner /> : '検索'}</Button>
            </form>
            {results && (results.length === 0 ? <Empty>見つかりませんでした</Empty> : (
                <div className="result-list">
                    {results.map(r => (
                        <div key={r.id} className="result-item">
                            <div className="grow"><strong>{r.id}</strong> <span className="small">{r.title}</span></div>
                            <Button size="sm" onClick={() => void open(r.id)} disabled={busy}>開く</Button>
                        </div>
                    ))}
                </div>
            ))}
        </div>
    );
}

const SOURCE_LABEL: Record<string, string> = {
    uniprot: 'UniProt', pdb: 'PDB', afdb: 'AlphaFold DB', pubchem: 'PubChem',
};

/** What was searched for before, so a component found once can be added again without
 *  remembering the accession. */
function HistoryTab({ onDone }: { onDone: () => void }) {
    const { addComponent, addImport, setView, toast } = useStore();
    const [rows, setRows] = useState<SearchRecord[] | null>(null);
    const [busy, setBusy] = useState<string | null>(null);
    const [filter, setFilter] = useState('');

    const load = () => { void api.searches().then(setRows).catch(e => toast('error', errorMessage(e))); };
    useEffect(load, []);   // eslint-disable-line react-hooks/exhaustive-deps

    const addAgain = async (source: string, id: string, title: string) => {
        setBusy(`${source}:${id}`);
        try {
            if (source === 'uniprot') {
                const entry = await api.uniprotEntry(id);
                addComponent({ type: 'protein', label: entry.name.slice(0, 60), sequence: entry.sequence,
                    msa: 'server', source: { db: 'UniProt', id: entry.accession } });
                toast('success', `${entry.name} を追加しました`);
            } else if (source === 'pubchem') {
                const r = await api.pubchem(title || id);
                addComponent({ type: 'ligand', label: r.name, smiles: r.smiles });
                toast('success', `${r.name} を追加しました`);
            } else {
                const item = await (source === 'afdb' ? api.importAfdb(id) : api.importPdb(id));
                addImport(item);
                setView({ kind: 'import', item });
                toast('success', `${id} を開きました`);
            }
            onDone();
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setBusy(null);
        }
    };

    if (rows === null) return <Empty><Spinner /></Empty>;
    const needle = filter.trim().toLowerCase();
    const shown = needle
        ? rows.filter(r => r.query.toLowerCase().includes(needle)
            || r.top.some(h => `${h.id} ${h.title ?? ''}`.toLowerCase().includes(needle)))
        : rows;
    if (rows.length === 0) return <Empty>まだ検索していません。UniProt や PDB を検索すると、ここに残ります</Empty>;
    return (
        <div className="stack">
            <div className="row">
                <input className="grow" value={filter} onChange={e => setFilter(e.target.value)}
                    placeholder="検索履歴を絞り込む" aria-label="検索履歴を絞り込む" />
                <Button onClick={() => void api.clearSearches().then(load).catch(e => toast('error', errorMessage(e)))}>
                    履歴を消す
                </Button>
            </div>
            {shown.length === 0 ? <Empty>一致する履歴がありません</Empty> : (
                <div className="result-list">
                    {shown.map(r => (
                        <div key={r.id} className="result-item history-item">
                            <div className="row">
                                <span className="kind">{SOURCE_LABEL[r.source] ?? r.source}</span>
                                <strong className="grow">{r.query}</strong>
                                <span className="small muted">{r.hits} 件 · {new Date(r.created_at * 1000).toLocaleString('ja-JP')}</span>
                            </div>
                            {r.picked && <div className="small muted">前回追加したのは {r.picked.id} ({r.picked.title})</div>}
                            <div className="chips">
                                {r.top.map(h => (
                                    <button type="button" key={h.id} className="chip" disabled={busy !== null}
                                        title={h.title ?? h.id}
                                        onClick={() => void addAgain(r.source, h.id, h.title ?? '')}>
                                        {busy === `${r.source}:${h.id}` ? <Spinner size={10} /> : h.id}
                                    </button>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

function PasteTab({ onDone }: { onDone: () => void }) {
    const { addComponent, toast } = useStore();
    const [type, setType] = useState<PolymerType>('protein');
    const [label, setLabel] = useState('');
    const [text, setText] = useState('');

    const add = () => {
        const records: { header: string; body: string }[] = [];
        let current: { header: string; body: string } | null = null;
        for (const line of text.split(/\r?\n/)) {
            if (line.startsWith('>')) {
                current = { header: line.slice(1).trim(), body: '' };
                records.push(current);
            } else {
                if (!current) {
                    current = { header: '', body: '' };
                    records.push(current);
                }
                current.body += line;
            }
        }
        const cleaned = records.filter(r => r.body.trim()).map(r => ({ ...r, ...cleanSequence(r.body, type) }));
        if (!cleaned.length) {
            toast('error', '配列が空です');
            return;
        }
        const bad = cleaned.find(r => r.error);
        if (bad) {
            toast('error', `${bad.header || '配列'}: ${bad.error}`);
            return;
        }
        cleaned.forEach((r, i) => addComponent({
            type, label: r.header.slice(0, 60) || label || `${TYPE_LABEL[type]} ${i + 1}`,
            sequence: r.sequence, msa: type === 'protein' ? 'server' : undefined,
        }));
        onDone();
    };

    return (
        <div className="stack">
            <div className="row">
                <Field label="種類">
                    <select value={type} onChange={e => setType(e.target.value as PolymerType)}>
                        <option value="protein">タンパク質</option>
                        <option value="dna">DNA</option>
                        <option value="rna">RNA</option>
                    </select>
                </Field>
                <Field label="名前 (FASTA のヘッダーがあればそちらを使います)">
                    <input value={label} onChange={e => setLabel(e.target.value)} placeholder="例: 設計したヘリックス" />
                </Field>
            </div>
            <textarea className="mono" rows={10} value={text} onChange={e => setText(e.target.value)} spellCheck={false}
                placeholder={'>my_protein\nMKTAYIAKQRQISFVKSHFSRQ...\n(複数の > を貼ると、それぞれが構成要素になります)'} />
            <div className="row"><Button variant="primary" onClick={add} disabled={!text.trim()}>追加</Button></div>
        </div>
    );
}

function LigandTab({ onDone }: { onDone: () => void }) {
    const { addComponent, toast } = useStore();
    const [name, setName] = useState('');
    const [smiles, setSmiles] = useState('');
    const [ccd, setCcd] = useState('');
    const [preview, setPreview] = useState<{ label: string; smiles?: string; ccd?: string; info?: ChemInfo; svg?: string; note?: string } | null>(null);
    const [busy, setBusy] = useState(false);

    const run = async (fn: () => Promise<void>) => {
        setBusy(true);
        try {
            await fn();
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setBusy(false);
        }
    };

    const pickCcd = (code: string) => void run(async () => {
        setCcd(code);
        const info = await api.ccd(code);
        setPreview({ label: info.name ?? info.ccd, ccd: info.ccd, svg: info.svg, note: info.formula });
    });
    const pickName = (value: string) => void run(async () => {
        setName(value);
        const r = await api.pubchem(value);
        setPreview({ label: r.name, smiles: r.smiles, info: r.describe, svg: r.svg, note: `PubChem CID ${r.cid}` });
    });

    return (
        <div className="stack">
            <div className="quick-picks">
                <span className="small muted">よく使う補因子・イオン</span>
                {COMMON_CCD.map(([code, label]) => (
                    <button type="button" key={code} className="chip" disabled={busy} title={label} onClick={() => pickCcd(code)}>{code}</button>
                ))}
                <span className="small muted">薬の例</span>
                {COMMON_DRUGS.map(([value, label]) => (
                    <button type="button" key={value} className="chip" disabled={busy} onClick={() => pickName(value)}>{label}</button>
                ))}
            </div>
            <form className="row" onSubmit={e => { e.preventDefault(); void run(async () => {
                const r = await api.pubchem(name);
                setPreview({ label: r.name, smiles: r.smiles, info: r.describe, svg: r.svg, note: `PubChem CID ${r.cid}` });
            }); }}>
                <Field label="名前で探す (PubChem)"><input autoFocus value={name} onChange={e => setName(e.target.value)} placeholder="例: acetazolamide, caffeine, imatinib" /></Field>
                <Button type="submit" disabled={!name.trim() || busy}>{busy ? <Spinner /> : '検索'}</Button>
            </form>
            <form className="row" onSubmit={e => { e.preventDefault(); void run(async () => {
                const info = await api.chemDescribe(smiles);
                setPreview({ label: info.formula, smiles, info, svg: info.svg });
            }); }}>
                <Field label="SMILES"><input className="mono" value={smiles} onChange={e => setSmiles(e.target.value)} placeholder="例: CC(=O)Oc1ccccc1C(=O)O" /></Field>
                <Button type="submit" disabled={!smiles.trim() || busy}>確認</Button>
            </form>
            <form className="row" onSubmit={e => { e.preventDefault(); void run(async () => {
                const info = await api.ccd(ccd);
                setPreview({ label: info.name ?? info.ccd, ccd: info.ccd, svg: info.svg, note: info.formula });
            }); }}>
                <Field label="CCD コード (PDB の化学成分辞書)" hint="補因子・金属イオンに便利: ATP, HEM, NAD, FAD, ZN, MG, CA"><input value={ccd} onChange={e => setCcd(e.target.value.toUpperCase())} placeholder="例: ATP" maxLength={5} /></Field>
                <Button type="submit" disabled={!ccd.trim() || busy}>確認</Button>
            </form>
            {preview && (
                <div className="ligand-preview">
                    {preview.svg && <img alt={preview.label} src={`data:image/svg+xml;utf8,${encodeURIComponent(preview.svg)}`} />}
                    <div className="stack grow">
                        <strong>{preview.label}</strong>
                        {preview.note && <span className="small muted">{preview.note}</span>}
                        {preview.smiles && <span className="mono small">{preview.smiles}</span>}
                        {preview.info && (
                            <span className="small">{preview.info.formula} · MW {preview.info.molecular_weight} · 重原子 {preview.info.heavy_atoms}
                                {!preview.info.affinity_ok && <span className="warn"> · 親和性予測の推奨範囲外</span>}</span>
                        )}
                        <div className="row">
                            <Button variant="primary" onClick={() => {
                                addComponent({ type: 'ligand', label: preview.label.slice(0, 40), ...(preview.ccd ? { ccd: preview.ccd } : { smiles: preview.smiles }) });
                                onDone();
                            }}>作業台に追加</Button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

function FileTab({ onDone }: { onDone: () => void }) {
    const { toast } = useStore();
    const [item, setItem] = useState<ImportedStructure | null>(null);
    const [busy, setBusy] = useState(false);
    if (item) return <StructurePicker item={item} onDone={onDone} />;
    return (
        <div className="stack">
            <label className="drop">
                <input type="file" accept=".cif,.mmcif,.pdb,.ent" onChange={async e => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    setBusy(true);
                    try {
                        setItem(await api.importUpload(file.name, await file.text()));
                    } catch (err) {
                        toast('error', errorMessage(err));
                    } finally {
                        setBusy(false);
                    }
                }} />
                {busy ? <Spinner /> : 'mmCIF / PDB ファイルを選ぶ'}
            </label>
        </div>
    );
}

function LibraryTab({ onDone }: { onDone: () => void }) {
    const { addComponent, setWorkbench, toast } = useStore();
    const [items, setItems] = useState<LibraryItem[] | null>(null);
    // Saving a workbench is how work is kept; removing one was a single unconfirmed click next
    // to the button that opens it. Ask once, the way the job rows do.
    const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
    useEffect(() => {
        api.library().then(setItems).catch(e => toast('error', errorMessage(e)));
    }, [toast]);
    if (items === null) return <Spinner />;
    if (!items.length) return <Empty>保存した構成要素や作業台はまだありません</Empty>;
    return (
        <div className="result-list">
            {items.map(it => (
                <div key={it.id} className="result-item">
                    <span className={`type-badge type-${it.type}`}>{it.type === 'workbench' ? '作業台' : TYPE_LABEL[it.type]}</span>
                    <div className="grow">{it.name}</div>
                    <Button size="sm" variant="primary" onClick={() => {
                        if (it.type === 'workbench') {
                            const wb = libraryWorkbench(it.data);
                            if (!wb) {
                                toast('error', 'ライブラリの作業台データが壊れています');
                                return;
                            }
                            setWorkbench(() => wb);
                        } else {
                            const comp = libraryComponent(it.data);
                            if (!comp) {
                                toast('error', 'ライブラリの構成要素データが壊れています');
                                return;
                            }
                            addComponent(comp);
                        }
                        onDone();
                    }}>{it.type === 'workbench' ? '開く' : '追加'}</Button>
                    <Button size="sm" variant={confirmDelete === it.id ? 'danger' : 'ghost'}
                        title="ライブラリから消します (元に戻せません)"
                        onClick={() => {
                            if (confirmDelete !== it.id) {
                                setConfirmDelete(it.id);
                                window.setTimeout(() => setConfirmDelete(c => (c === it.id ? null : c)), 4000);
                                return;
                            }
                            setConfirmDelete(null);
                            void api.deleteLibrary(it.id)
                                .then(() => setItems(list => (list ?? []).filter(x => x.id !== it.id)))
                                .catch(e => toast('error', errorMessage(e)));
                        }}>{confirmDelete === it.id ? '本当に削除' : '削除'}</Button>
                </div>
            ))}
        </div>
    );
}

const COMMON_CCD: [string, string][] = [
    ['ATP', 'ATP (アデノシン三リン酸)'], ['ADP', 'ADP'], ['NAD', 'NAD+'], ['FAD', 'FAD'], ['HEM', 'ヘム'],
    ['SAM', 'S-アデノシルメチオニン'], ['ZN', '亜鉛イオン'], ['MG', 'マグネシウムイオン'], ['CA', 'カルシウムイオン'],
];
const COMMON_DRUGS: [string, string][] = [
    ['acetazolamide', 'アセタゾラミド'], ['caffeine', 'カフェイン'], ['ibuprofen', 'イブプロフェン'], ['imatinib', 'イマチニブ'],
];

const TYPES: ComponentType[] = ['protein', 'dna', 'rna', 'ligand'];
const str = (v: unknown): string | undefined => (typeof v === 'string' ? v : undefined);

/** Library rows are JSON written by earlier sessions; rebuild a component only from fields that check out. */
function libraryComponent(data: Record<string, unknown>): Omit<WBComponent, 'uid'> | null {
    const type = TYPES.find(t => t === data.type);
    if (!type) return null;
    const sequence = str(data.sequence);
    const smiles = str(data.smiles);
    const ccd = str(data.ccd);
    if (type === 'ligand' ? !(smiles || ccd) : !sequence) return null;
    const src = data.source;
    const source = src && typeof src === 'object' && 'db' in src && 'id' in src
        ? { db: String((src as { db: unknown }).db), id: String((src as { id: unknown }).id) } : undefined;
    return {
        type,
        label: str(data.label) ?? type,
        copies: typeof data.copies === 'number' && data.copies >= 1 ? data.copies : 1,
        sequence,
        baseSequence: str(data.baseSequence) ?? sequence,
        smiles,
        ccd,
        msa: data.msa === 'single' ? 'single' : type === 'protein' ? 'server' : undefined,
        cyclic: data.cyclic === true,
        source,
    };
}

function libraryWorkbench(data: Record<string, unknown>): Workbench | null {
    if (!Array.isArray(data.components)) return null;
    const components: WBComponent[] = [];
    for (const raw of data.components) {
        if (!raw || typeof raw !== 'object') return null;
        const c = libraryComponent(raw as Record<string, unknown>);
        if (!c) return null;
        components.push({ ...c, uid: newUid() });
    }
    return { name: str(data.name) ?? '作業台', components, affinityBinderUid: null, params: defaultParams(), parentJobId: null };
}
