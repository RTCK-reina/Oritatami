import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { api, ApiError, errorMessage } from '../api';
import { useStore, type AssistantRequest } from '../store';
import type { ChatMessage, Proposal, Thread } from '../types';
import { uiEvents } from '../uiEvents';
import { assignChains, buildSpec } from '../workbench';
import { Button, Icon, Kbd, Spinner } from './ui';

type Mode = AssistantRequest['mode'];
const MODES: { id: Mode; label: string; placeholder: string }[] = [
    { id: 'mutations', label: '変異を提案', placeholder: '目的 (例: 熱安定性を上げたい / 結合を弱めたら何が起きる? / 面白い形の変化)' },
    { id: 'complex', label: '複合体を提案', placeholder: '目的 (例: 天然の結合相手と組ませたい / 阻害剤を試したい)' },
    { id: 'design', label: '新しい配列を設計', placeholder: '作りたいもの (例: 4 本のヘリックスが束になった小さなタンパク質)' },
    { id: 'explain', label: '結果を解説', placeholder: '特に知りたいこと (空欄でも可)' },
    { id: 'chat', label: '会話', placeholder: '質問や相談 (例: ipTM ってなに? この変異の意味は?)' },
];

/** Conversations are listed newest-first; the stamp is what separates two runs of the same task. */
function threadStamp(at: number): string {
    const d = new Date(at * 1000);
    const now = new Date();
    const time = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
    return d.toDateString() === now.toDateString() ? time : `${d.getMonth() + 1}/${d.getDate()} ${time}`;
}

const STARTERS: { mode: Mode; text: string }[] = [
    { mode: 'chat', text: 'pLDDT と PAE の違いをやさしく教えて' },
    { mode: 'mutations', text: '熱に強くなりそうな変異' },
    { mode: 'complex', text: '天然の結合相手と組ませたい' },
    { mode: 'design', text: '小さな β シートのタンパク質' },
];

export function AssistantPanel({ onCollapse }: { onCollapse?: () => void }) {
    const store = useStore();
    const { workbench, health, refreshHealth, selectedJobId, jobs, scanJobFor, threadId, setThreadId, toast, assistantRequest, clearAssistantRequest } = store;
    // "変異を提案" needs a protein on the workbench, so on an empty workbench the default mode
    // left the send button dead on first launch. Start in chat and switch once there is something
    // to mutate — unless the person has already picked a mode themselves.
    const [mode, setMode] = useState<Mode>(() => (store.workbench.components.some(c => c.type === 'protein') ? 'mutations' : 'chat'));
    const modePicked = useRef(false);
    const [text, setText] = useState('');
    const [count, setCount] = useState(3);
    const [focusChain, setFocusChain] = useState<string>('');
    const [thread, setThread] = useState<Thread | null>(null);
    const [threads, setThreads] = useState<Omit<Thread, 'messages'>[]>([]);
    const [busy, setBusy] = useState<{ since: number } | null>(null);
    const [, tick] = useState(0);
    const listRef = useRef<HTMLDivElement>(null);
    const composerRef = useRef<HTMLTextAreaElement>(null);
    // "/" from anywhere puts the cursor here, the way it does in most tools.
    useEffect(() => uiEvents.on('focusAssistant', () => composerRef.current?.focus()), []);
    const handled = useRef<number>(0);

    const hasProtein = workbench.components.some(c => c.type === 'protein');
    useEffect(() => {
        if (modePicked.current) return;
        setMode(hasProtein ? 'mutations' : 'chat');
    }, [hasProtein]);

    const chains = useMemo(() => assignChains(workbench.components), [workbench.components]);
    const proteinChains = workbench.components.filter(c => c.type === 'protein').flatMap(c => chains.get(c.uid) ?? []);
    const chain = proteinChains.includes(focusChain) ? focusChain : proteinChains[0] ?? '';
    const selectedJob = jobs.find(j => j.id === selectedJobId && j.kind === 'predict' && j.status === 'succeeded');
    const focusComp = workbench.components.find(c => (chains.get(c.uid) ?? []).includes(chain));
    const scan = scanJobFor(focusComp?.sequence) ?? scanJobFor(focusComp?.baseSequence);
    const llm = health?.llm;

    useEffect(() => {
        api.threads().then(setThreads).catch(e => toast('error', errorMessage(e)));
    }, [toast, thread?.id]);

    useEffect(() => {
        if (!threadId) {
            setThread(null);
            return;
        }
        api.thread(threadId).then(setThread).catch(e => {
            // a remembered conversation that was deleted: start fresh instead of reporting an error
            if (e instanceof ApiError && e.status === 404) setThreadId(null);
            else toast('error', errorMessage(e));
        });
    }, [threadId, toast, setThreadId]);

    useEffect(() => {
        listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
    }, [thread?.messages.length, busy]);

    useEffect(() => {
        if (!busy) return;
        const t = window.setInterval(() => tick(x => x + 1), 1000);
        return () => window.clearInterval(t);
    }, [busy]);

    // fetching Ollama takes a couple of minutes; the banner follows it
    const installing = !!llm?.install?.active;
    useEffect(() => {
        if (!installing) return;
        const t = window.setInterval(() => void refreshHealth(), 2000);
        return () => window.clearInterval(t);
    }, [installing, refreshHealth]);

    const send = async (m: Mode, message: string, jobId: string | null, heavy = false) => {
        if (busy) return;
        setBusy({ since: Date.now() });
        const optimistic: ChatMessage = { role: 'user', content: m === 'chat' ? message : `[${MODES.find(x => x.id === m)?.label}] ${message}`, mode: m, created_at: Date.now() / 1000 };
        setThread(t => (t ? { ...t, messages: [...t.messages, optimistic] } : { id: '', title: '', created_at: 0, updated_at: 0, messages: [optimistic] }));
        try {
            const res = await api.ask({
                thread_id: threadId, mode: m, message, workbench: buildSpec(workbench),
                job_id: jobId, scan_job_id: scan?.status === 'succeeded' ? scan.id : null, focus_chain: chain || null, count,
                heavy,
            });
            setThreadId(res.thread_id);
            setThread(await api.thread(res.thread_id));
            setText('');
        } catch (e) {
            const isGuard = e instanceof ApiError && e.code === 'safeguard';
            toast(
                'error',
                isGuard
                    ? '⚠️ モデルの安全フィルタが応答を拒否しました。質問の言い回しを変えてみてください'
                    : `LLM: ${errorMessage(e)}`,
            );
            setThread(t => (t ? { ...t, messages: t.messages.filter(x => x !== optimistic) } : t));
            if (!isGuard) void refreshHealth();
        } finally {
            setBusy(null);
        }
    };

    // requests from other panels ("LLM に解説させる") fire once per nonce with the latest workbench
    const sendRef = useRef(send);
    sendRef.current = send;
    const selectedJobIdRef = useRef<string | null>(null);
    selectedJobIdRef.current = selectedJob?.id ?? null;
    useEffect(() => {
        if (!assistantRequest || assistantRequest.nonce === handled.current) return;
        handled.current = assistantRequest.nonce;
        clearAssistantRequest(assistantRequest.nonce);
        modePicked.current = true;
        setMode(assistantRequest.mode);
        if (assistantRequest.mode === 'design' || assistantRequest.mode === 'complex' || assistantRequest.mode === 'mutations') {
            // requests without a message from the command palette just pick the mode and focus the composer
            if (!assistantRequest.message && assistantRequest.mode !== 'mutations') {
                composerRef.current?.focus();
                return;
            }
        }
        void sendRef.current(assistantRequest.mode, assistantRequest.message, assistantRequest.jobId ?? selectedJobIdRef.current);
    }, [assistantRequest, clearAssistantRequest]);

    const current = MODES.find(x => x.id === mode) ?? MODES[0];
    const needsWorkbench = mode === 'mutations' && proteinChains.length === 0;
    const needsResult = mode === 'explain' && !selectedJob;

    return (
        <div className="panel assistant">
            <div className="panel-head">
                <span className="qwen-logo">LLM</span>
                <span className="small muted ellipsis model-name" title={llm?.model}>{llm?.model ?? ''}</span>
                <select className="mini-select grow" value={threadId ?? ''} onChange={e => setThreadId(e.target.value || null)} title="会話の履歴">
                    <option value="">新しい会話</option>
                    {threads.map(t => (
                        <option key={t.id} value={t.id}>{t.title} — {threadStamp(t.updated_at)}</option>
                    ))}
                </select>
                <Button size="sm" variant="ghost" onClick={() => setThreadId(null)} title="新しい会話を始める">新規</Button>
                {threadId && (
                    <button type="button" className="icon-btn danger" aria-label="この会話を削除"
                        title="開いている会話を履歴から削除する"
                        onClick={() => {
                            if (!window.confirm('この会話を履歴から削除します。よろしいですか?')) return;
                            void api.deleteThread(threadId)
                                .then(() => {
                                    setThreadId(null);
                                    setThread(null);
                                    return api.threads().then(setThreads);
                                })
                                .catch(e => toast('error', errorMessage(e)));
                        }}><Icon name="trash" size={13} /></button>
                )}
                {onCollapse && <button type="button" className="icon-btn" onClick={onCollapse} aria-label="LLM パネルを隠す" title="隠す (⌘⇧B)"><Icon name="chevron-right" /></button>}
            </div>
            {llm && (!llm.server || !llm.model_available) && (
                <div className="banner">
                    {llm.install?.active
                        ? `Ollama を取得しています… ${llm.install.total ? Math.round(((llm.install.completed ?? 0) / llm.install.total) * 100) : 0}%`
                        : !llm.server
                            ? (llm.binary ? 'Ollama が起動していません。' : `この Mac に Ollama がありません (アプリ用に約 ${llm.download_mb ?? 150} MB 取得します)。`)
                            : `モデル ${llm.model} がまだありません。`}
                    {llm.install?.active ? null : !llm.server
                        ? <Button size="sm" onClick={() => void api.llmStart().then(r => {
                            if (r.installing) toast('info', 'Ollama の取得を始めました (設定 → 準備状況 で進捗が見られます)');
                            return refreshHealth();
                        }).catch(e => toast('error', errorMessage(e)))}>{llm.binary ? 'Ollama を起動' : '用意する'}</Button>
                        : <Button size="sm" onClick={() => void api.llmPull(llm.model).then(() => toast('info', 'ダウンロードを開始しました (設定画面で進捗を確認できます)')).catch(e => toast('error', errorMessage(e)))}>ダウンロード</Button>}
                </div>
            )}
            <div className="messages" ref={listRef}>
                {!thread?.messages.length && (
                    <div className="assistant-intro">
                        <p>作業台の分子を見て、LLM が試す案を出します。案はそのまま使われず、配列との照合・データベース検索・ESM-2 のスコアで検証されてから表示されます。</p>
                        <ul className="small muted">
                            <li>変異: 作業台のタンパク質に対する変異セット (ESM-2 スキャンがあれば根拠に使います)</li>
                            <li>複合体: 結合相手のタンパク質・リガンド・核酸 (UniProt / PubChem で解決)</li>
                            <li>設計: 新しい配列 (ESM-2 の自然さを表示。「ESM で磨く」で改良)</li>
                        </ul>
                        <div className="starters">
                            {STARTERS.map(st => (
                                <button type="button" key={st.text} className="starter" onClick={() => { setMode(st.mode); setText(st.text); composerRef.current?.focus(); }}>
                                    <span className="kind">{MODES.find(m => m.id === st.mode)?.label}</span> {st.text}
                                </button>
                            ))}
                        </div>
                    </div>
                )}
                {thread?.messages.map((m, i) => <MessageView key={`${m.created_at}-${i}`} m={m} />)}
                {busy && <div className="msg msg-assistant"><Spinner /> 考えています… {Math.round((Date.now() - busy.since) / 1000)} 秒</div>}
            </div>
            <div className="composer">
                <div className="mode-chips">
                    {MODES.map(x => (
                        <button key={x.id} className={`mode-chip ${mode === x.id ? 'active' : ''}`}
                            onClick={() => { modePicked.current = true; setMode(x.id); }}>{x.label}</button>
                    ))}
                </div>
                <div className="composer-opts small">
                    {(mode === 'mutations') && proteinChains.length > 0 && (
                        <label>対象 <select value={chain} onChange={e => setFocusChain(e.target.value)}>
                            {proteinChains.map(c => <option key={c} value={c}>チェーン {c}</option>)}
                        </select></label>
                    )}
                    {mode !== 'chat' && mode !== 'explain' && (
                        <label>案の数 <input type="number" min={1} max={8} value={count} onChange={e => setCount(Number(e.target.value))} /></label>
                    )}
                    {selectedJob && <span className="muted" title="直近に選んだ予測結果を文脈に含めます">結果: {selectedJob.title}</span>}
                    {mode === 'mutations' && scan?.status === 'succeeded' && <span className="muted">ESM スキャン使用</span>}
                </div>
                {(needsWorkbench || needsResult) && (
                    <div className="warn small">{needsWorkbench ? '作業台にタンパク質を追加してください' : 'ジョブ一覧から予測結果を選んでください'}</div>
                )}
                <div className="composer-input">
                    <textarea ref={composerRef} rows={2} value={text} placeholder={current.placeholder} onChange={e => setText(e.target.value)}
                        aria-label="LLM への入力"
                        onKeyDown={e => {
                            if (e.nativeEvent.isComposing) return;
                            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                                e.preventDefault();
                                void send(mode, text, selectedJob?.id ?? null, e.shiftKey && !!llm?.heavy_available);
                            }
                        }} />
                    <Button variant="qwen" disabled={!!busy || needsWorkbench || needsResult || (mode === 'chat' && !text.trim())}
                        onClick={() => void send(mode, text, selectedJob?.id ?? null)} title="⌘+Enter">
                        {busy ? <Spinner /> : <>送る <Kbd combo="mod+enter" /></>}
                    </Button>
                    {llm?.heavy_model && (
                        <Button disabled={!!busy || !llm.heavy_available || needsWorkbench || needsResult || (mode === 'chat' && !text.trim())}
                            onClick={() => void send(mode, text, selectedJob?.id ?? null, true)}
                            title={llm.heavy_available
                                ? `${llm.heavy_model} で答えさせます (遅い代わりに本文が詳しくなることがあります。⇧⌘+Enter)`
                                : `${llm.heavy_model} がまだダウンロードされていません`}>
                            じっくり <Kbd combo="mod+shift+enter" />
                        </Button>
                    )}
                </div>
            </div>
        </div>
    );
}

/** The model writes light Markdown; show **bold** and `code` instead of raw markers (no HTML is ever injected). */
function renderInline(text: string): ReactNode[] {
    return text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`)/g).map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**') && part.length > 4) return <strong key={i}>{part.slice(2, -2)}</strong>;
        if (part.startsWith('`') && part.endsWith('`') && part.length > 2) return <code key={i}>{part.slice(1, -1)}</code>;
        return part.replace(/^#{1,4}\s+/gm, '');
    });
}

function MessageView({ m }: { m: ChatMessage }) {
    const { runVariants, health } = useStore();
    const [queued, setQueued] = useState(false);
    if (m.role === 'user') return <div className="msg msg-user">{m.content}</div>;
    const mutationSets = (m.proposals ?? []).filter(p => p.type === 'mutation_set' && p.status !== 'invalid' && p.apply?.action === 'mutate' && p.chain && p.mutations?.length);
    return (
        <div className="msg msg-assistant">
            <div className="reply">{renderInline(m.content)}</div>
            {m.reply_issues && m.reply_issues.length > 0 ? (
                <div className="claim-warnings">
                    <strong>本文に配列と矛盾する記述があります</strong>
                    <ul>{m.reply_issues.map((x, i) => <li key={i}>{x}</li>)}</ul>
                </div>
            ) : m.content.trim() && (
                <p className="unverified-note">この文章は LLM が書いたもので検証されていません (照合・ESM-2 スコアリングを受けるのは下の案だけです)</p>
            )}
            {m.proposals?.map((p, i) => <ProposalCard key={i} p={p} />)}
            {mutationSets.length > 1 && (
                <div className="row wrap batch-row">
                    <Button size="sm" variant="primary" disabled={queued || !health?.boltz.bin}
                        title="作業台の配列に各案を 1 つずつ適用した変異体を、まとめて予測キューに入れます"
                        onClick={async () => {
                            await runVariants(mutationSets.map(p => ({ chain: p.chain as string, mutations: p.mutations as string[] })), { origin: 'qwen' });
                            setQueued(true);
                        }}>{queued ? '追加しました' : `${mutationSets.length} 案をまとめて予測`}</Button>
                    <span className="small muted">結果は「比較」タブで並べて見られます</span>
                </div>
            )}
            {m.corrected && <div className="small muted">本文が配列と食い違っていたので、指摘して書き直させました</div>}
            {m.elapsed_sec !== undefined && <div className="small muted">{m.model} · {m.elapsed_sec} 秒</div>}
        </div>
    );
}

const TYPE_TEXT: Record<string, string> = {
    mutation_set: '変異', add_ligand: 'リガンド', add_protein: '結合相手', add_nucleic: '核酸', new_protein: '新しい配列',
};

function ProposalCard({ p }: { p: Proposal }) {
    const { applyProposal, toast } = useStore();
    const [showSeq, setShowSeq] = useState(false);
    const [done, setDone] = useState<string | null>(null);
    const ok = p.status !== 'invalid' && !!p.apply;
    return (
        <div className={`proposal proposal-${p.status}`}>
            <div className="proposal-head">
                <span className="kind">{TYPE_TEXT[p.type] ?? p.type}</span>
                <strong className="grow">{p.title}</strong>
                <span className={`badge badge-${p.status}`}>{p.status === 'ok' ? '検証OK' : p.status === 'warning' ? '注意あり' : '却下'}</span>
            </div>
            {p.rationale && <div className="rationale">{renderInline(p.rationale)}</div>}
            {p.mutations && (
                <div className="res-chips">
                    {p.mutations.map(code => {
                        const s = p.esm?.per_mutation.find(x => x.mutation === code || x.mutation.endsWith(code));
                        return <span key={code} className="chip chip-mut">{p.chain}:{code}{s && <small className={s.llr >= 0 ? 'pos' : 'neg'}> {s.llr > 0 ? '+' : ''}{s.llr.toFixed(1)}</small>}</span>;
                    })}
                    {p.esm && <span className="small muted">ESM-2 合計 LLR {p.esm.total_llr > 0 ? '+' : ''}{p.esm.total_llr.toFixed(2)}</span>}
                </div>
            )}
            {p.chem?.svg && <img className="proposal-svg" alt={p.title} src={`data:image/svg+xml;utf8,${encodeURIComponent(p.chem.svg)}`} />}
            {p.chem && <div className="small mono">{p.chem.formula} · MW {p.chem.molecular_weight} · 重原子 {p.chem.heavy_atoms}</div>}
            {p.resolved && <div className="small muted">{Object.entries(p.resolved).filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => `${k}: ${v}`).join(' · ')}</div>}
            {p.stats && (
                <div className="small mono">
                    {p.stats.length} 残基 · 疎水性 {Math.round(p.stats.hydrophobic_fraction * 100)}% · 最多 {p.stats.most_common}
                    {p.stats.pseudo_perplexity !== undefined && ` · ESM PPPL ${p.stats.pseudo_perplexity}`}
                </div>
            )}
            {p.sequence && (
                <div>
                    <button className="link small" onClick={() => setShowSeq(s => !s)}>{showSeq ? '配列を隠す' : '配列を見る'}</button>
                    {showSeq && <div className="mono small seq-wrap">{p.sequence}</div>}
                </div>
            )}
            {p.repaired && p.repaired.length > 0 && (
                <ul className="issues repaired" title="残基名は合っていて位置だけ違ったので、その残基がある位置に直して通しました">
                    {p.repaired.map((x, i) => <li key={i}>位置を直しました: {x}</li>)}
                </ul>
            )}
            {p.issues.length > 0 && <ul className="issues">{p.issues.map((x, i) => <li key={i}>{x}</li>)}</ul>}
            {ok && (
                <div className="row wrap">
                    {p.apply?.action !== 'new_protein' && (
                        <Button size="sm" onClick={async () => { await applyProposal(p, false); setDone('適用済み'); }}>作業台に適用</Button>
                    )}
                    <Button size="sm" variant="primary" onClick={async () => { await applyProposal(p, true); setDone('予測を開始'); }}>
                        {p.apply?.action === 'new_protein' ? '新しい作業台で予測' : '適用して予測'}
                    </Button>
                    {p.type === 'new_protein' && p.sequence && (
                        <Button size="sm" variant="ghost" title="ESM-2 で不自然な残基を置き換えて、より自然な配列に近づけます"
                            onClick={() => void api.submitRefine({ sequence: p.sequence ?? '', label: p.title, origin: 'qwen' })
                                .then(() => { toast('info', 'ESM で磨くジョブを追加しました'); setDone('ESM ジョブ追加'); })
                                .catch(e => toast('error', errorMessage(e)))}>ESM で磨く</Button>
                    )}
                    {done && <span className="small muted">{done}</span>}
                </div>
            )}
        </div>
    );
}
