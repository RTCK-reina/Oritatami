import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { api, errorMessage, formatBytes } from '../api';
import type { MemoryTrend, PdbWatcherConfig, SearchHistory } from '../api';
import { useStore } from '../store';
import type { GpuState, LlmStatus, Settings, SsdInfo, StorageInfo } from '../types';
import { uiEvents } from '../uiEvents';
import { SetupStatus } from './SetupStatus';
import { Button, Field, Icon, Modal, Spinner } from './ui';

const SUGGESTED = ['qwen3.5:9b', 'qwen3.5:4b', 'qwen3.5:2b', 'qwen3.5:27b', 'gemma3:4b', 'gemma3:12b', 'gemma3:27b'];

const SECTIONS = [
    { id: 'status', label: '準備状況', icon: 'check' },
    { id: 'llm', label: 'LLM', icon: 'sparkles' },
    { id: 'predict', label: '構造予測', icon: 'flask' },
    { id: 'esm', label: 'ESM-2', icon: 'layers' },
    { id: 'autopilot', label: 'オートパイロット', icon: 'play' },
    { id: 'watcher', label: 'PDB ウォッチャー', icon: 'target' },
    { id: 'jobs', label: 'ジョブと通知', icon: 'retry' },
    { id: 'records', label: 'やり取りの記録', icon: 'folder' },
    { id: 'system', label: 'GPU・SSD', icon: 'settings' },
    { id: 'storage', label: 'ストレージ', icon: 'download' },
] as const;
type SectionId = (typeof SECTIONS)[number]['id'];

/** Long measurement notes belong in the settings, but not in front of the control they explain. */
function More({ children }: { children: ReactNode }) {
    return (
        <details className="more">
            <summary>詳しく</summary>
            <div className="more-body">{children}</div>
        </details>
    );
}

export function SettingsDialog({ onClose }: { onClose: () => void }) {
    const { refreshHealth, toast } = useStore();
    const [section, setSection] = useState<SectionId>('status');
    const [s, setS] = useState<Settings | null>(null);
    const [saved, setSaved] = useState<Settings | null>(null);
    const [cfg, setCfg] = useState<PdbWatcherConfig | null>(null);
    const [savedCfg, setSavedCfg] = useState<PdbWatcherConfig | null>(null);
    const [llm, setLlm] = useState<LlmStatus | null>(null);
    const [saving, setSaving] = useState(false);
    const [pullModel, setPullModel] = useState('qwen3.5:9b');

    useEffect(() => {
        api.settings().then(v => { setS(v); setSaved(v); }).catch(e => toast('error', errorMessage(e)));
        api.llmStatus().then(setLlm).catch(e => toast('error', errorMessage(e)));
        api.pdbWatcher.config().then(v => { setCfg(v); setSavedCfg(v); }).catch(e => toast('error', errorMessage(e)));
    }, [toast]);

    useEffect(() => {
        if (!llm?.pull?.active) return;
        const t = window.setInterval(() => {
            api.llmStatus().then(st => {
                setLlm(st);
                if (!st.pull?.active) void refreshHealth();
            }).catch(e => toast('error', errorMessage(e)));
        }, 1500);
        return () => window.clearInterval(t);
    }, [llm?.pull?.active, refreshHealth, toast]);

    const dirty = useMemo(() => {
        const a = s && saved && JSON.stringify(s) !== JSON.stringify(saved);
        const b = cfg && savedCfg && JSON.stringify(cfg) !== JSON.stringify(savedCfg);
        return Boolean(a || b);
    }, [s, saved, cfg, savedCfg]);

    const close = () => {
        if (dirty && !window.confirm('保存していない変更があります。破棄して閉じますか?')) return;
        onClose();
    };

    if (!s) return <Modal title="設定" onClose={onClose}><Spinner /></Modal>;
    const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setS({ ...s, [k]: v });
    const setCfgKey = <K extends keyof PdbWatcherConfig>(k: K, v: PdbWatcherConfig[K]) =>
        setCfg(c => (c ? { ...c, [k]: v } : c));
    const pull = llm?.pull;

    /** One button saves everything on this dialog that is a *setting*. Actions that talk to the
     *  OS (the GPU limit) or delete files say so where they sit. */
    const save = async () => {
        setSaving(true);
        try {
            const next = await api.updateSettings(s);
            setS(next);
            setSaved(next);
            if (cfg && savedCfg && JSON.stringify(cfg) !== JSON.stringify(savedCfg)) {
                const w = await api.pdbWatcher.setConfig({
                    enabled: cfg.enabled, max_per_poll: cfg.max_per_poll,
                    min_seq_len: cfg.min_seq_len, max_seq_len: cfg.max_seq_len,
                });
                setCfg(w);
                setSavedCfg(w);
            }
            uiEvents.emit('autopilotChanged');
            await refreshHealth();
            toast('success', '設定を保存しました');
            onClose();
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setSaving(false);
        }
    };

    return (
        <Modal title="設定" onClose={close} wide className="settings-modal">
            <div className="settings-layout">
                <nav className="settings-nav" aria-label="設定の項目">
                    {SECTIONS.map(sec => (
                        <button type="button" key={sec.id} className={`settings-nav-item ${section === sec.id ? 'active' : ''}`}
                            aria-current={section === sec.id} onClick={() => setSection(sec.id)}>
                            <Icon name={sec.icon} size={13} /> <span>{sec.label}</span>
                        </button>
                    ))}
                </nav>

                <div className="settings-pane settings">
                    {section === 'status' && (
                        <section>
                            <h3>準備状況</h3>
                            <SetupStatus />
                        </section>
                    )}

                    {section === 'llm' && (
                        <section>
                            <h3>LLM (llama.cpp)</h3>
                            <Field label="使うモデル" hint={<>
                                予測の直前に LLM はメモリから解放されるので、Boltz とメモリを奪い合うことはありません。
                                <More>
                                    効くのは解析中の LLM と ESM-2 の同居分で、qwen3.5:9b は実測 5.9 GB です。
                                    24 GB なら 14B 級まで載りますが、生成速度がそのまま解析時間になります (実測 80〜105 文字/秒)。
                                </More>
                            </>}>
                                <select value={s.llm_model} onChange={e => set('llm_model', e.target.value)}>
                                    {[...new Set([s.llm_model, ...(llm?.models.map(m => m.name) ?? [])])].map(name => (
                                        <option key={name} value={name}>{name}{llm?.models.some(m => m.name === name) ? '' : ' (未ダウンロード)'}</option>
                                    ))}
                                </select>
                            </Field>
                            <Field label="じっくり答えるときのモデル" hint={<>
                                LLM パネルの「じっくり」ボタンだけが使います。空欄でボタンが消えます。
                                <More>
                                    自律ループは常に上のモデルを使います。実測では 27B (19 GB) はこの機体の GPU 上限 (約 18 GB) を
                                    超えて CPU に溢れ、1 問 15 分でも返りませんでした。載るのは 14B 級 (9〜14 GB) までです。
                                </More>
                            </>}>
                                <select value={s.llm_model_heavy} onChange={e => set('llm_model_heavy', e.target.value)}>
                                    <option value="">使わない</option>
                                    {[...new Set([...(s.llm_model_heavy ? [s.llm_model_heavy] : []), ...(llm?.models.map(m => m.name) ?? [])])].map(name => (
                                        <option key={name} value={name}>{name}{llm?.models.some(m => m.name === name) ? '' : ' (未ダウンロード)'}</option>
                                    ))}
                                </select>
                            </Field>
                            <div className="row wrap">
                                <Field label="温度"><input type="number" step={0.1} min={0} max={2} value={s.llm_temperature} onChange={e => set('llm_temperature', Number(e.target.value))} /></Field>
                                <Field label="llama-server URL"><input value={s.ollama_url} onChange={e => set('ollama_url', e.target.value)} /></Field>
                            </div>
                            <label className="inline-toggle">
                                <input type="checkbox" checked={s.llm_think} onChange={e => set('llm_think', e.target.checked)} /> 思考モード (回答前に長く考える)
                            </label>
                            <p className="field-hint">
                                自律運転させるならオフを推奨します。
                                <More>
                                    qwen3.5:9b での実測 — 配列の位置を当てる質問 10 件で、正答は 1/10 から 8/10 に上がる一方、
                                    所要は 2 秒から 1,556 秒 (26 分) になりました。うち 2 件は思考がコンテキスト窓 (16,384) を使い切って
                                    打ち切られ、答えが返りませんでした。1 件あたり最長 409 秒。オートパイロットは 1 周につき
                                    LLM を 2〜3 回呼びます。
                                </More>
                            </p>
                            {s.llm_think && s.autopilot_enabled && (
                                <div className="claim-warnings">
                                    <strong>思考モードとオートパイロットが同時に有効です</strong>
                                    解析が数十分に伸びたり、空の結果になることがあります。自律運転させるならオフを推奨します。
                                </div>
                            )}
                            {s.llm_think && llm?.thinking_supported === false && llm.model === s.llm_model && (
                                <div className="claim-warnings">
                                    <strong>{s.llm_model} は思考モードに対応していません</strong>
                                    サーバー側で通常モードに切り替わります。思考させたいときは対応モデルを選んでください。
                                </div>
                            )}
                            <hr className="settings-rule" />
                            <h4>モデルのダウンロード</h4>
                            <div className="row wrap end">
                                <Field label="モデル名">
                                    <input list="llm-models" value={pullModel} onChange={e => setPullModel(e.target.value)} />
                                </Field>
                                <datalist id="llm-models">{SUGGESTED.map(m => <option key={m} value={m} />)}</datalist>
                                <Button disabled={!!pull?.active || !pullModel.trim()} onClick={() => void api.llmPull(pullModel.trim())
                                    .then(p => setLlm(l => (l ? { ...l, pull: p } : l))).catch(e => toast('error', errorMessage(e)))}>取得</Button>
                                <span className="small muted">保存とは別に、押した時点で始まります</span>
                            </div>
                            {pull && pull.model && (
                                <div className="small">
                                    {pull.model}: {pull.error ? <span className="warn">{pull.error}</span> : pull.status}
                                    {pull.total ? ` ${Math.round(((pull.completed ?? 0) / pull.total) * 100)}%` : ''}
                                    {pull.active && pull.total ? <div className="progress"><i style={{ width: `${((pull.completed ?? 0) / pull.total) * 100}%` }} /></div> : null}
                                </div>
                            )}
                        </section>
                    )}

                    {section === 'predict' && (
                        <section>
                            <h3>構造予測 (Boltz-2)</h3>
                            <div className="row wrap end">
                                <Field label="計算デバイス">
                                    <select value={s.accelerator} onChange={e => set('accelerator', e.target.value)}>
                                        <option value="auto">自動 (MPS があれば GPU)</option>
                                        <option value="mps">GPU (MPS)</option>
                                        <option value="cpu">CPU</option>
                                    </select>
                                </Field>
                                <Field label="既定のサンプル数"><input type="number" min={1} max={10} value={s.diffusion_samples} onChange={e => set('diffusion_samples', Number(e.target.value))} /></Field>
                                <Field label="既定のリサイクル"><input type="number" min={1} max={10} value={s.recycling_steps} onChange={e => set('recycling_steps', Number(e.target.value))} /></Field>
                                <Field label="既定の拡散ステップ"><input type="number" min={10} max={500} value={s.sampling_steps} onChange={e => set('sampling_steps', Number(e.target.value))} /></Field>
                            </div>
                            <p className="field-hint">初期値は サンプル 1 / リサイクル 4 / 拡散 200 です。新しい作業台がこの値から始まります。</p>
                            <label className="inline-toggle">
                                <input type="checkbox" checked={s.mps_strict} onChange={e => set('mps_strict', e.target.checked)} /> CPU に落ちたらジョブを失敗させる (GPU 厳格モード)
                            </label>
                            <p className="field-hint">
                                Metal のカーネルが無い演算は既定では黙って CPU で処理され、その分だけ遅くなります。オンにすると気づけます。
                                <More>2026-09-14 の実測では単鎖 (35 秒) も複合体+親和性 (243 秒) も CPU 落ちゼロで完走しました。</More>
                            </p>
                            <Field label="GPU メモリの上限倍率" hint={<>
                                PyTorch が 1 プロセスに許す上限を、Metal の推奨値の何倍にするか。0 で上限なし。
                                <More>
                                    既定 1.7 では推奨 17.76 GB に対して 30.2 GB で頭打ちになり、1,696 残基 (3 本鎖) の予測が
                                    16 分ぶんを捨てて落ちました。上限を外すと落ちない代わりに、物理メモリを超えたぶんはスワップになり
                                    SSD に書き続けます (「GPU・SSD」の項を参照)。
                                </More>
                            </>}>
                                <input type="number" min={0} max={8} step={0.1} value={s.mps_memory_ratio}
                                    onChange={e => set('mps_memory_ratio', Number(e.target.value))} />
                            </Field>
                            <hr className="settings-rule" />
                            <h4>変異案の検証</h4>
                            <label className="inline-toggle">
                                <input type="checkbox" checked={s.mpnn_enabled} onChange={e => set('mpnn_enabled', e.target.checked)} /> 変異案を逆折り畳み (ProteinMPNN) でも確認する
                            </label>
                            <p className="field-hint">
                                Boltz と ESM-2 は似た盲点を持ちます。逆折り畳みは「この骨格にこの配列が載るか」を見るので向きが違います。
                                <More>
                                    ユビキチンの C 末端グリシンや K63 を壊す変異に Boltz と ESM-2 の両方が賛成しました。
                                    逆折り畳みは実測で I44A・L67D・G75C・G76C を最下位、無害な表面変異を上位に並べました。
                                </More>
                            </p>
                            {s.mpnn_enabled && (
                                <Field label="逆折り畳みの却下ライン" hint={<>
                                    親よりこれ以上悪化する案は予測にかけずに落とします。0 で却下せず並べ替えだけ。
                                    <More>480 件の履歴では上位半分に絞ると親超えの割合が 4.6% → 8.3% になりました。</More>
                                </>}>
                                    <input type="number" step={0.01} min={0} max={5} value={s.mpnn_veto}
                                        onChange={e => set('mpnn_veto', Number(e.target.value))} />
                                </Field>
                            )}
                            <hr className="settings-rule" />
                            <h4>MSA とファイル</h4>
                            <Field label="MSA サーバー" hint="タンパク質配列がこのサーバーに送られます (既定: ColabFold 公開サーバー)">
                                <input value={s.msa_server_url} onChange={e => set('msa_server_url', e.target.value)} />
                            </Field>
                            <label className="inline-toggle" title="点変異体では、元の配列の MSA を流用して検索時間を省きます (置換 5% 以下)">
                                <input type="checkbox" checked={s.reuse_msa_for_variants} onChange={e => set('reuse_msa_for_variants', e.target.checked)} /> 変異体で MSA を再利用する
                            </label>
                            <label className="inline-toggle" title="予測の完了後、Boltz だけが使う前処理ファイルと MSA 検索の生データを削除します (1 件あたり数 MB〜数十 MB)。構造・スコア・MSA は残ります">
                                <input type="checkbox" checked={s.cleanup_intermediate} onChange={e => set('cleanup_intermediate', e.target.checked)} /> 完了後に中間ファイルを自動で削除する
                            </label>
                            <Field label="キャッシュ (重み・化学辞書)"><input value={s.boltz_cache} onChange={e => set('boltz_cache', e.target.value)} /></Field>
                        </section>
                    )}

                    {section === 'esm' && (
                        <section>
                            <h3>ESM-2</h3>
                            <div className="row wrap end">
                                <Field label="モデル" hint="大きいほど精度は上がるが遅い">
                                    <select value={s.esm_model} onChange={e => set('esm_model', e.target.value)}>
                                        <option value="facebook/esm2_t12_35M_UR50D">35M (軽量)</option>
                                        <option value="facebook/esm2_t30_150M_UR50D">150M</option>
                                        <option value="facebook/esm2_t33_650M_UR50D">650M (推奨)</option>
                                    </select>
                                </Field>
                                <Field label="デバイス">
                                    <select value={s.esm_device} onChange={e => set('esm_device', e.target.value)}>
                                        <option value="auto">自動</option>
                                        <option value="mps">GPU (MPS)</option>
                                        <option value="cpu">CPU</option>
                                    </select>
                                </Field>
                            </div>
                        </section>
                    )}

                    {section === 'autopilot' && (
                        <section>
                            <h3>オートパイロット</h3>
                            <p className="small muted">
                                予測が終わるたびに LLM が自動で解析し、有望な変異を新しい予測として投入します。
                                無人で回すため、次の 3 つ (1 回の枝分かれ数・24 時間の上限・待機の上限) で歯止めをかけています。
                            </p>
                            <label className="inline-toggle" title="オフにすると自動解析後の変異体投入と PDB ウォッチャーの投入が止まります。実行中のジョブは最後まで走り、新規投入だけが止まります">
                                <input type="checkbox" checked={s.autopilot_enabled}
                                    onChange={e => set('autopilot_enabled', e.target.checked)} /> オートパイロットを有効にする
                            </label>
                            {s.autopilot_enabled && s.autopilot_max_depth === 0 && (
                                <div className="claim-warnings">
                                    <strong>連続稼働モード (世代の上限が 0)</strong>
                                    このチェックを外すまで系統が伸び続けます。外した時点で新規投入だけが止まり、実行中のジョブは最後まで走ります。
                                    {s.autopilot_max_variants_per_job > 1 &&
                                        ' 変異体数が 2 以上のままだと世代ごとに倍々に増えるので、1 に下げてください。'}
                                </div>
                            )}

                            <h4>探索のしかた</h4>
                            <div className="row wrap end">
                                <Field label="1ジョブあたりの変異体数" hint="解析 1 回から自動投入する上限">
                                    <input type="number" min={0} max={10} value={s.autopilot_max_variants_per_job}
                                        onChange={e => set('autopilot_max_variants_per_job', Number(e.target.value))} />
                                </Field>
                                <Field label="世代の上限" hint="0 で無制限">
                                    <input type="number" min={0} max={100} value={s.autopilot_max_depth}
                                        onChange={e => set('autopilot_max_depth', Number(e.target.value))} />
                                </Field>
                                <Field label="山登りの我慢の手数" hint="0 で降りない">
                                    <input type="number" min={0} max={100} value={s.autopilot_climb_patience}
                                        disabled={s.autopilot_strategy !== 'climb'}
                                        onChange={e => set('autopilot_climb_patience', Number(e.target.value))} />
                                </Field>
                            </div>
                            <div className="row wrap end">
                                <Field label="探索の方針">
                                    <select value={s.autopilot_strategy} onChange={e => set('autopilot_strategy', e.target.value)}>
                                        <option value="climb">山登り (常に最良から枝分かれ)</option>
                                        <option value="walk">酔歩 (最新からそのまま続ける)</option>
                                    </select>
                                </Field>
                                <Field label="候補の選び方">
                                    <select value={s.autopilot_selection} onChange={e => set('autopilot_selection', e.target.value)}>
                                        <option value="greedy">良さそうな順</option>
                                        <option value="explore">試していない位置を優先</option>
                                    </select>
                                </Field>
                            </div>
                            <p className="field-hint">
                                世代を無制限にするなら変異体数は 1 に。
                                <More>
                                    山登りは子が親を超えなければ捨てて頂点に戻ります。51 世代の実測では酔歩は下り坂に入り、
                                    第 3 世代の 93.37 が第 51 世代には 91.43 まで落ちました。我慢の手数は、頂点から何手試して
                                    超えられなかったら次点に降りるか。野生型ユビキチンのように元が既に良い配列だと、これがないと
                                    1 変異の近傍から永久に出られません。候補の選び方は、実測では全試行の 23% が 6 つの位置に集中しており、
                                    「良さそうな順」は同じ場所を何度も測り直す傾向があります。
                                </More>
                            </p>
                            <Field label="実験名" hint="名前を入れると、その名前のついた予測だけを枝分かれの起点にします。空なら全履歴が対象">
                                <input type="text" placeholder="空欄で全履歴" value={s.autopilot_experiment}
                                    onChange={e => set('autopilot_experiment', e.target.value)} />
                            </Field>

                            <hr className="settings-rule" />
                            <h4>歯止め</h4>
                            <div className="row wrap end">
                                <Field label="24時間あたりの上限" hint="0 で無制限">
                                    <input type="number" min={0} max={2000} value={s.autopilot_daily_budget}
                                        onChange={e => set('autopilot_daily_budget', Number(e.target.value))} />
                                </Field>
                                <Field label="待機ジョブの上限" hint="捌けるまで投入を止める">
                                    <input type="number" min={1} max={100} value={s.autopilot_max_queued}
                                        onChange={e => set('autopilot_max_queued', Number(e.target.value))} />
                                </Field>
                                <Field label="空き容量の下限 (GB)" hint="下回ると投入を止める">
                                    <input type="number" min={0} max={500} step={1} value={s.autopilot_min_disk_gb}
                                        onChange={e => set('autopilot_min_disk_gb', Number(e.target.value))} />
                                </Field>
                            </div>
                            <p className="field-hint">上限はオートパイロットと PDB ウォッチャーの合計です。手動投入はこの枠を消費しません。</p>

                            <hr className="settings-rule" />
                            <h4>提案の絞り込み</h4>
                            <div className="row wrap end">
                                <Field label="1回に出させる提案数" hint="候補の幅">
                                    <input type="number" min={1} max={8} value={s.autopilot_proposals_per_call}
                                        onChange={e => set('autopilot_proposals_per_call', Number(e.target.value))} />
                                </Field>
                                <Field label="ESM-2 スコアの下限" hint="極端な提案だけを落とす安全網">
                                    <input type="number" step={0.5} min={-25} max={5} value={s.autopilot_min_esm_llr}
                                        onChange={e => set('autopilot_min_esm_llr', Number(e.target.value))} />
                                </Field>
                            </div>
                            <p className="field-hint">
                                既定が緩いのは実測の結果です。選抜そのものはスコア順が担います。
                                <More>
                                    ESM-2 と Boltz の一致度は弱く (ユビキチン 9 変異で r=0.34)、-10 に設定すると最良だった
                                    Q40V (LLR -10.0, pLDDT +0.8) を捨てる一方、唯一破壊的だった L67R (LLR -9.7, pLDDT -9.4) は
                                    通してしまいました。値の分布もタンパク質依存で、ユビキチンでは全 1 点変異の中央値が -7.0 です。
                                </More>
                            </p>
                            <Field label="変更を禁止する残基" hint="番号でも K63 のような表記でもかまいません。カンマ区切り">
                                <input type="text" placeholder="例: K48, K63, R72, G75, G76"
                                    value={s.autopilot_protected_residues}
                                    onChange={e => set('autopilot_protected_residues', e.target.value)} />
                            </Field>
                            <p className="field-hint">
                                pLDDT を上げるだけなら、機能に必要な残基を壊すのが一番手っ取り早い近道になります。
                                <More>ユビキチンで 491 件回したときは 91% が C 末端の G76 を、94% が K63 を潰していました。</More>
                            </p>
                            <label className="inline-toggle">
                                <input type="checkbox" checked={s.autopilot_protect_disordered}
                                    onChange={e => set('autopilot_protect_disordered', e.target.checked)} /> 起点で乱れていた残基を自動で保護する
                            </label>
                            <Field label="乱れた残基のしきい値 (pLDDT)" hint="系統の起点でこの値を下回っていた残基を変更禁止にします">
                                <input type="number" min={0} max={100} step={5}
                                    disabled={!s.autopilot_protect_disordered}
                                    value={s.autopilot_disorder_plddt}
                                    onChange={e => set('autopilot_disorder_plddt', Number(e.target.value))} />
                            </Field>
                            <label className="inline-toggle" title="複合体では、どの残基が相手のチェーンに触れているかを Boltz が予測のたびに計算しています。役割がわかっている数少ない部分なので、界面を作り直したいとき以外は触らせません">
                                <input type="checkbox" checked={s.autopilot_protect_interfaces}
                                    onChange={e => set('autopilot_protect_interfaces', e.target.checked)} /> 界面に接している残基を自動で保護する
                            </label>
                            <label className="inline-toggle" title="1世代あたり LLM を2回呼ぶうちの1回。結果の解説文はどこにも保存されません">
                                <input type="checkbox" checked={s.autopilot_explain}
                                    onChange={e => set('autopilot_explain', e.target.checked)} /> 変異提案の前に結果の解説もさせる (1 世代の所要時間の約 27%)
                            </label>

                            <hr className="settings-rule" />
                            <h4>改善の判定</h4>
                            <div className="row wrap end">
                                <Field label="判定に使う指標" hint="自動: 複数チェーンは ipTM、単量体は平均 pLDDT">
                                    <select value={s.autopilot_improvement_metric}
                                        onChange={e => set('autopilot_improvement_metric', e.target.value)}>
                                        <option value="auto">自動 (複合体は ipTM / 単量体は平均 pLDDT)</option>
                                        <option value="mean_plddt">平均 pLDDT</option>
                                        <option value="core_plddt">コア pLDDT (下位10%を除く)</option>
                                        <option value="iptm">ipTM</option>
                                        <option value="ptm">pTM</option>
                                        <option value="confidence_score">信頼度スコア</option>
                                        <option value="complex_plddt">複合体 pLDDT</option>
                                    </select>
                                </Field>
                                <Field label="改善とみなす差" hint="pLDDT 換算 (2.0 = pLDDT +2 / ipTM +0.02)">
                                    <input type="number" step={0.5} min={0} max={50} value={s.autopilot_improvement_delta}
                                        onChange={e => set('autopilot_improvement_delta', Number(e.target.value))} />
                                </Field>
                            </div>
                            <NoiseNote value={s.autopilot_improvement_delta}
                                onAdopt={(v: number) => set('autopilot_improvement_delta', v)} />
                        </section>
                    )}

                    {section === 'watcher' && (
                        <section>
                            <h3>PDB ウォッチャー</h3>
                            <p className="small muted">RCSB PDB を 6 時間ごとに巡回し、新しくリリースされたタンパク質構造を自動取得・予測します。24 時間の上限はオートパイロットと共通です。</p>
                            {!cfg ? <Spinner /> : (
                                <>
                                    <label className="inline-toggle">
                                        <input type="checkbox" checked={cfg.enabled} onChange={e => setCfgKey('enabled', e.target.checked)} /> PDB ウォッチャーを有効にする
                                    </label>
                                    <div className="row wrap end">
                                        <Field label="1回あたりの最大取得件数" hint="ポーリングごとの上限">
                                            <input type="number" min={1} max={20} value={cfg.max_per_poll}
                                                onChange={e => setCfgKey('max_per_poll', Number(e.target.value))} />
                                        </Field>
                                        <Field label="最小配列長 (aa)">
                                            <input type="number" min={10} max={1000} value={cfg.min_seq_len}
                                                onChange={e => setCfgKey('min_seq_len', Number(e.target.value))} />
                                        </Field>
                                        <Field label="最大配列長 (aa)">
                                            <input type="number" min={10} max={5000} value={cfg.max_seq_len}
                                                onChange={e => setCfgKey('max_seq_len', Number(e.target.value))} />
                                        </Field>
                                    </div>
                                    {cfg.last_checked && (
                                        <div className="small muted">最終ポーリング: {new Date(cfg.last_checked).toLocaleString('ja-JP')}</div>
                                    )}
                                    <div className="row">
                                        <Button size="sm" onClick={() => void api.pdbWatcher.pollNow()
                                            .then(() => toast('success', 'ポーリングを開始しました'))
                                            .catch(e => toast('error', errorMessage(e)))}>今すぐポーリング</Button>
                                        <span className="small muted">保存とは別に、押した時点で始まります</span>
                                    </div>
                                </>
                            )}
                        </section>
                    )}

                    {section === 'jobs' && (
                        <section>
                            <h3>失敗時の自動再試行</h3>
                            <p className="small muted">
                                MSA サーバーや回線の一時的な失敗だけを対象に、待ち時間を倍にしながら再投入します。
                                配列の誤りやメモリ不足など、やり直しても同じ結果になる失敗は対象外です。
                            </p>
                            <label className="inline-toggle">
                                <input type="checkbox" checked={s.job_auto_retry}
                                    onChange={e => set('job_auto_retry', e.target.checked)} /> 一時的な失敗を自動で再試行する
                            </label>
                            <Field label="再試行の回数" hint="1回目は60秒後、2回目は120秒後…と間隔が倍になります">
                                <input type="number" min={0} max={10} value={s.job_max_retries}
                                    onChange={e => set('job_max_retries', Number(e.target.value))} />
                            </Field>
                            <hr className="settings-rule" />
                            <h3>通知</h3>
                            <label className="inline-toggle">
                                <input type="checkbox" checked={s.notify_on_finish} onChange={e => set('notify_on_finish', e.target.checked)} /> 計算が終わったら通知する (ウィンドウが裏にあるとき)
                            </label>
                            <label className="inline-toggle" title="ウィンドウを閉じていても、バックエンドから通知します">
                                <input type="checkbox" checked={s.autopilot_notify_improvement}
                                    onChange={e => set('autopilot_notify_improvement', e.target.checked)} /> 親を上回る変異体が出たときに通知する
                            </label>
                        </section>
                    )}

                    {section === 'records' && <LlmLogSection value={s.llm_log_limit} onChange={n => set('llm_log_limit', n)} />}

                    {section === 'system' && (
                        <>
                            <GpuMemorySection />
                            <AppMemorySection />
                            <SsdWearSection applecare={s.applecare} onApplecare={v => set('applecare', v)} />
                        </>
                    )}

                    {section === 'storage' && <StorageSection />}
                </div>
            </div>

            <div className="settings-foot">
                <Button variant="primary" disabled={saving} onClick={() => void save()}>
                    {saving ? <Spinner size={12} /> : '保存'}
                </Button>
                <Button variant="ghost" onClick={close}>閉じる</Button>
                <span className="spacer" />
                <span className="small muted">{dirty ? '未保存の変更があります' : '保存済み'}</span>
            </div>
        </Modal>
    );
}

/** The threshold was picked by hand; the machine's own repeats say what it should be. */
function NoiseNote({ value, onAdopt }: { value: number; onAdopt: (v: number) => void }) {
    const [h, setH] = useState<SearchHistory | null>(null);
    useEffect(() => { api.searchHistory().then(setH).catch(() => setH(null)); }, []);
    if (!h || !h.sigma || !h.suggested_delta) return null;
    const suggested = h.suggested_delta;
    const low = value < suggested;
    return (
        <p className={`small ${low ? 'warn' : 'muted'}`}>
            同じ配列を同じ条件で測り直したときのブレは σ = {h.sigma.toFixed(3)}
            （{h.repeats} 回の測り直しから）。差は 2 回の測定の引き算なので、
            意味があると言えるのは <strong>{suggested.toFixed(2)}</strong> 以上です。
            {low && <> 今の {value} はブレの範囲に入っています。{' '}
                <button type="button" className="link" onClick={() => onAdopt(suggested)}>
                    {suggested.toFixed(2)} にする
                </button></>}
            {h.pairs > 0 && <> これまでの {h.pairs} 回の試行の平均は {h.mean_delta?.toFixed(2)}、
                改善した割合は {((h.improved_rate ?? 0) * 100).toFixed(1)}% です。</>}
        </p>
    );
}

/** The log exists to be looked at later, so it needs a way out of the database. */
function LlmLogSection({ value, onChange }: { value: number; onChange: (n: number) => void }) {
    const { toast } = useStore();
    const [total, setTotal] = useState<number | null>(null);
    const [busy, setBusy] = useState(false);

    useEffect(() => {
        api.llmCalls(1).then(r => setTotal(r.total)).catch(() => setTotal(null));
    }, []);

    return (
        <section>
            <h3>やり取りの記録</h3>
            <p className="small muted">
                LLM に送ったプロンプトと返ってきた本文・提案を、自律ループの分も含めて全部残します。
                あとで「なぜこの提案が出たか」を追ったり、書き出して学習・評価に使ったりするためのものです。
                {total !== null && ` 現在 ${total.toLocaleString('ja-JP')} 件。`}
            </p>
            <Field label="保存件数" hint="古いものから消えます。1 件およそ 12 KB。0 にすると記録しません">
                <input type="number" min={0} max={1000000} step={500} value={value}
                    onChange={e => onChange(Number(e.target.value))} />
            </Field>
            <div className="row">
                <Button disabled={busy || !total} onClick={async () => {
                    setBusy(true);
                    try {
                        const r = await api.exportLlmCalls();
                        toast('success', `${r.count} 件を書き出しました (${formatBytes(r.bytes)}): ${r.path}`);
                    } catch (e) {
                        toast('error', errorMessage(e));
                    } finally {
                        setBusy(false);
                    }
                }}>{busy ? <Spinner /> : 'JSONL で書き出す'}</Button>
                <span className="small muted">保存とは別に、押した時点で書き出します</span>
            </div>
        </section>
    );
}

function StorageSection() {
    const { toast, refreshJobs } = useStore();
    const [info, setInfo] = useState<StorageInfo | null>(null);
    const [busy, setBusy] = useState(false);
    const [deleteFailed, setDeleteFailed] = useState(false);

    useEffect(() => {
        api.storage().then(setInfo).catch(e => toast('error', errorMessage(e)));
    }, [toast]);

    return (
        <section>
            <h3>ストレージ</h3>
            {!info ? <Spinner /> : (
                <>
                    <div className="kv">
                        <span>ジョブ {formatBytes(info.jobs_bytes)}</span>
                        <span>取り込み・比較 {formatBytes(info.imports_bytes)}</span>
                        <span>MSA キャッシュ {formatBytes(info.msa_cache_bytes)}</span>
                        <span>Boltz の重み {formatBytes(info.boltz_cache_bytes)}</span>
                        <span>空き {info.disk_free_gb} GB</span>
                    </div>
                    <div className="small muted mono">{info.home}</div>
                    <label className="inline-toggle small">
                        <input type="checkbox" checked={deleteFailed} onChange={e => setDeleteFailed(e.target.checked)} /> 失敗・キャンセルしたジョブも削除する
                    </label>
                    <div className="row wrap">
                        <Button size="sm" variant={deleteFailed ? 'danger' : 'default'} disabled={busy} onClick={async () => {
                            setBusy(true);
                            try {
                                const r = await api.cleanupStorage({ intermediate: true, aligned_older_than_days: 0, delete_failed_jobs: deleteFailed });
                                setInfo(r.storage);
                                toast('success', `${formatBytes(r.freed_bytes)} を解放しました${r.jobs_deleted ? ` (ジョブ ${r.jobs_deleted} 件を削除)` : ''}`);
                                if (r.jobs_deleted) await refreshJobs();
                            } catch (e) {
                                toast('error', errorMessage(e));
                            } finally {
                                setBusy(false);
                            }
                        }}>{busy ? <Spinner size={11} /> : '不要なファイルを削除'}</Button>
                        <span className="small muted">押した時点で削除します (元に戻せません)。中間ファイルと重ね合わせ表示用のファイルが対象で、予測結果は残ります。</span>
                    </div>
                </>
            )}
        </section>
    );
}

/** SSD wear, and what a swapping run costs it.
 *
 *  This exists because the allocator ceiling was removed: a job that used to die with an OOM
 *  now runs on swap instead, and the price moved from "16 minutes lost" to "writes to a drive
 *  soldered to the logic board". Read-only: nothing here changes anything.
 */
function SsdWearSection({ applecare, onApplecare }: { applecare: boolean; onApplecare: (v: boolean) => void }) {
    const { toast } = useStore();
    const [ssd, setSsd] = useState<SsdInfo | null>(null);

    useEffect(() => {
        api.ssd().then(setSsd).catch(e => toast('error', errorMessage(e)));
    }, [toast]);

    if (!ssd) return <section><h3>SSD の摩耗</h3><Spinner /></section>;
    if (!ssd.available) {
        return (
            <section>
                <h3>SSD の摩耗</h3>
                <p className="small muted">
                    読み取れません。macOS 標準の <code>system_profiler</code> は「S.M.A.R.T. status: Verified」しか返さず、
                    <code>ioreg</code> にも書き込み量の項目がないため、smartmontools が要ります。
                    <code>brew install smartmontools</code> を入れると、スワップする予測が SSD の寿命の何 % を使うかまで出せるようになります。
                    入れなくても書き込み量（TB）の見込みは出ます。
                </p>
            </section>
        );
    }
    const w = ssd.wear;
    const spare = w && w.available_spare !== null;
    return (
        <section>
            <h3>SSD の摩耗</h3>
            {w && (
                <div className="kv">
                    <span>使用 {w.percentage_used}%</span>
                    <span>書き込み {w.written_tb} TB</span>
                    {spare && <span>予備ブロック {w.available_spare}%{w.media_errors ? ` · エラー ${w.media_errors}` : ''}</span>}
                    {w.power_on_hours !== null && <span>稼働 {w.power_on_hours} 時間</span>}
                </div>
            )}
            <p className="small muted">
                {w?.tb_per_percent !== null && w
                    ? (w.tb_per_percent_basis === 'delta'
                        ? <>この個体の実測で <strong>{w.tb_per_percent} TB ＝ 寿命 1%</strong>（{w.tb_per_percent_span}% ぶんの変化から測定、記録 {w.readings} 件）。</>
                        : <>暫定で <strong>{w.tb_per_percent} TB ＝ 寿命 1%</strong>。{w.written_tb} TB で {w.percentage_used}% という 1 点からの外挿なので、切片も直線性も未確認です。</>)
                    : <>使用率がまだ 0% なので、寿命あたりの換算はできません。1% 動いた時点で出ます。</>}
                {' '}スワップする予測は SSD へ約 {ssd.mb_per_sec} MB/s、1 日あたり約 {ssd.tb_per_day} TB を書きます
                {ssd.life_percent_per_day !== null && <>（寿命の約 {ssd.life_percent_per_day}%/日{ssd.life_basis === 'single' ? '、暫定' : ''}）</>}。
            </p>
            <label className="inline-toggle" title="この項目は動作を何も変えません。上の警告の書き方が変わるだけです">
                <input type="checkbox" checked={applecare} onChange={e => onApplecare(e.target.checked)} /> AppleCare+ に加入している
            </label>
            <p className={`small ${applecare ? 'muted' : 'warn'}`}>
                Apple は Mac 内蔵 SSD の TBW を公表しておらず、書き込み量で保証を切ることもしていません。
                代わりに AppleCare+ は「通常の消耗、または通常の経年劣化に起因する故障」を除外しており、摩耗がそれに当たるかは条文からは決まりません。
                Apple Silicon の SSD は基板直付けなので、故障＝ロジックボード交換です。
                {applecare
                    ? ' 加入していても摩耗が通る保証はないので、保証をアテにしない前提で判断してください。'
                    : ' 未加入と設定されています。摩耗で死んだ場合はロジックボード交換の実費になります。'}
            </p>
        </section>
    );
}

/**
 * What this process itself is holding, job after job.
 *
 * Boltz runs in its own process and gives everything back when it exits, so a long batch can
 * only leak here: gemmi structures, PAE matrices, the ESM-2 weights, and torch's MPS pool,
 * which keeps freed blocks instead of returning them to Metal. The footprint is sampled at
 * the end of every job — the number shown is measured, not estimated, and if it is flat there
 * is nothing to fix.
 */
function AppMemorySection() {
    const { toast } = useStore();
    const [mem, setMem] = useState<MemoryTrend | null>(null);
    const [busy, setBusy] = useState(false);
    const load = () => { api.memory().then(setMem).catch(e => toast('error', errorMessage(e))); };
    useEffect(load, [toast]);

    const release = async () => {
        setBusy(true);
        try {
            const r = await api.releaseMemory();
            toast('success', r.freed_gb >= 0.05 || r.esm_unloaded
                ? `${r.freed_gb.toFixed(1)} GB 解放しました${r.esm_unloaded ? ' (ESM-2 も降ろしました)' : ''}`
                : '解放できる分はありませんでした');
            load();
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setBusy(false);
        }
    };

    if (!mem) return <section><h3>アプリ自身のメモリ</h3><Spinner /></section>;
    return (
        <section>
            <h3>アプリ自身のメモリ</h3>
            <div className="kv">
                <span>今の使用量 {mem.current_gb !== null ? `${mem.current_gb.toFixed(2)} GB` : '不明'}</span>
                <span>{mem.samples >= 2 && mem.growth_gb !== null
                    ? `ジョブ ${mem.samples} 件で ${mem.growth_gb >= 0 ? '+' : ''}${mem.growth_gb.toFixed(2)} GB`
                    : `ジョブ ${mem.samples} 件ぶん計測`}</span>
                {mem.torch_mps && <span>torch が Metal から確保 {mem.torch_mps.driver_allocated_gb.toFixed(2)} GB (使用中 {mem.torch_mps.in_use_gb.toFixed(2)} GB)</span>}
            </div>
            <p className={`small ${mem.climbing ? 'warn' : 'muted'}`}>
                {mem.climbing
                    ? `ジョブをまたいで ${mem.growth_gb?.toFixed(1)} GB 増えています。次の予測がその分だけ狭いメモリで走るので、解放するかアプリを再起動してください。`
                    : 'ジョブ終了ごとに torch のキャッシュを返しています。この数字が増え続けていなければ、連続実行でメモリが痩せていくことはありません。'}
            </p>
            <div className="row wrap">
                <Button size="sm" disabled={busy} onClick={() => void release()}>
                    {busy ? <Spinner size={11} /> : '今すぐ解放する'}
                </Button>
                <Button size="sm" variant="ghost" onClick={load}>測り直す</Button>
            </div>
        </section>
    );
}

function GpuMemorySection() {
    const { toast } = useStore();
    const [gpu, setGpu] = useState<GpuState | null>(null);
    const [mb, setMb] = useState<string>('');
    const [busy, setBusy] = useState(false);

    useEffect(() => {
        api.gpu().then(g => { setGpu(g); setMb(String(g.wired_limit_mb)); })
            .catch(e => toast('error', errorMessage(e)));
    }, [toast]);

    const apply = async (value: number) => {
        setBusy(true);
        try {
            const after = await api.setGpuLimit(value);
            setGpu(after);
            setMb(String(after.wired_limit_mb));
            toast('success', value === 0
                ? 'GPU のメモリ上限を既定に戻しました。反映にはアプリの再起動が要ります'
                : `GPU のメモリ上限を ${(value / 1024).toFixed(1)} GB にしました。反映にはアプリの再起動が要ります`);
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setBusy(false);
        }
    };

    if (!gpu) return <section><h3>GPU のメモリ上限</h3><Spinner /></section>;
    const maxGb = gpu.max_mb / 1024;
    const presets = [Math.round(gpu.total_gb * 0.75), Math.round(gpu.total_gb * 0.83), Math.round(maxGb)]
        .filter((v, i, a) => v > 0 && v <= maxGb && a.indexOf(v) === i);

    return (
        <section>
            <h3>GPU のメモリ上限</h3>
            <div className="kv">
                <span>搭載メモリ {gpu.total_gb} GB</span>
                <span>今の上限 {gpu.metal_limit_gb !== null ? `${gpu.metal_limit_gb.toFixed(2)} GB` : '不明'}</span>
                <span>{gpu.is_default ? 'OS の既定値' : `sysctl で ${gpu.wired_limit_mb} MB に設定済み`}</span>
            </div>
            <p className="small muted">
                ユニファイドメモリでも、Metal が 1 プロセスに渡す量には上限があります。上げたぶんは macOS と他のアプリから取り上げることになります。
                <More>
                    実測では 1,696 残基（3 本鎖）の予測が 30 GB まで伸びたところで打ち切られ、16 分ぶんが無駄になりました。
                    609 残基の単量体は 997 秒で完走しています。上限を上げるとその手前で落ちなくなります。
                </More>
            </p>
            <div className="row wrap">
                {presets.map(gb => (
                    <Button key={gb} size="sm" disabled={busy} onClick={() => void apply(gb * 1024)}>{gb} GB</Button>
                ))}
                <Button size="sm" variant="ghost" disabled={busy || gpu.is_default} onClick={() => void apply(0)}>既定に戻す</Button>
                {presets.length === 0 && <span className="small muted">この機体では候補を出せませんでした (下の入力で指定してください)</span>}
            </div>
            <Field label="自分で指定 (MB)" hint={`0 で既定値。上限 ${gpu.max_mb} MB — 残り ${gpu.headroom_gb} GB は macOS に残します`}>
                <input type="number" min={0} max={gpu.max_mb} step={256} value={mb}
                    onChange={e => setMb(e.target.value)} />
            </Field>
            <div className="row wrap">
                <Button size="sm" variant="primary" disabled={busy || mb === String(gpu.wired_limit_mb)}
                    onClick={() => void apply(Number(mb))}>
                    {busy ? <Spinner size={11} /> : '適用する'}
                </Button>
                <span className="small muted">
                    下の「保存」ではなくこのボタンで反映します。macOS の認証ダイアログが出ます (パスワードはこのアプリを経由しません)。
                    設定は再起動すると既定に戻り、反映にはこのアプリの再起動が要ります。
                </span>
            </div>
        </section>
    );
}
