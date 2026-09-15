import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, errorMessage, type AutopilotStatus, type ProtectedSuggestion } from '../api';
import { useStore } from '../store';
import type { JobSummary, Settings } from '../types';
import { uiEvents } from '../uiEvents';
import { Button, Field, Icon, Modal, Spinner } from './ui';

const SEVERITY_LABEL: Record<string, string> = {
    critical: '必須', high: '重要', medium: '参考', low: '参考',
};

/** "K48, 63, G76" -> {48, 63, 76} */
function parsePositions(text: string): Set<number> {
    const out = new Set<number>();
    for (const token of (text || '').replace(/;/g, ',').split(',')) {
        const m = /(\d+)/.exec(token);
        if (m) out.add(Number(m[1]));
    }
    return out;
}

/**
 * What the loop will do next, decided in front of the thing it will do it to.
 *
 * The switch used to be a toggle: it turned on and started rewriting whatever the last result
 * happened to be, under whatever settings were left behind. The prohibition list in particular
 * was a text field in a dialog three clicks away, so in practice it stayed empty — and an empty
 * list is what lets a pLDDT search eat the functional residues.
 */
export function AutopilotDialog({ onClose }: { onClose: () => void }) {
    const { jobs, selectedJobId, toast } = useStore();
    const [status, setStatus] = useState<AutopilotStatus | null>(null);
    const [settings, setSettings] = useState<Settings | null>(null);
    const [saving, setSaving] = useState(false);

    const candidates = useMemo(
        () => jobs.filter(j => j.kind === 'predict' && j.status === 'succeeded').slice(0, 40), [jobs]);
    const [targetId, setTargetId] = useState<string>('');
    const target: JobSummary | undefined = candidates.find(j => j.id === targetId) ?? candidates[0];

    const [chain, setChain] = useState<string>('');
    const [suggestion, setSuggestion] = useState<ProtectedSuggestion | null>(null);
    const [targetChains, setTargetChains] = useState<string[]>([]);
    const [loadingSuggestion, setLoadingSuggestion] = useState(false);
    const [ticked, setTicked] = useState<Set<number>>(new Set());
    const [manual, setManual] = useState<Set<number>>(new Set());

    useEffect(() => {
        api.autopilotStatus().then(setStatus).catch(e => toast('error', errorMessage(e)));
        api.settings().then(s => {
            setSettings(s);
            setManual(parsePositions(s.autopilot_protected_residues));
        }).catch(e => toast('error', errorMessage(e)));
    }, [toast]);

    useEffect(() => {
        if (!targetId && (selectedJobId || candidates[0])) {
            const pick = candidates.find(j => j.id === selectedJobId)?.id ?? candidates[0]?.id ?? '';
            setTargetId(pick);
        }
    }, [selectedJobId, candidates, targetId]);

    const loadSuggestion = useCallback(async (jobId: string, ch: string) => {
        setLoadingSuggestion(true);
        try {
            const s = await api.protectedSuggest(jobId, ch || undefined);
            setSuggestion(s);
            setChain(s.chain);
            setTicked(new Set(s.positions.filter(p => p.default_on).map(p => p.position)));
        } catch (e) {
            toast('error', errorMessage(e));
            setSuggestion(null);
        } finally {
            setLoadingSuggestion(false);
        }
    }, [toast]);

    useEffect(() => {
        if (!target) return;
        void loadSuggestion(target.id, '');
        api.job(target.id)
            .then(full => setTargetChains((full.result && 'chains' in full.result
                ? full.result.chains.filter(c => c.type === 'protein').map(c => c.chain) : [])))
            .catch(() => setTargetChains([]));
        // deliberately keyed on the target only; the chain picker reloads on its own
    }, [target?.id, loadSuggestion]);

    const set = <K extends keyof Settings>(k: K, v: Settings[K]) =>
        setSettings(s => (s ? { ...s, [k]: v } : s));

    const positionsByNumber = useMemo(() => {
        const m = new Map<number, ProtectedSuggestion['positions'][number]>();
        (suggestion?.positions ?? []).forEach(p => m.set(p.position, p));
        return m;
    }, [suggestion]);

    const all = useMemo(() => {
        const nums = new Set<number>([...ticked, ...manual]);
        return [...nums].sort((a, b) => a - b);
    }, [ticked, manual]);

    const listText = useMemo(() => all.map(pos => {
        const p = positionsByNumber.get(pos);
        return `${p?.residue ?? ''}${pos}`;
    }).join(', '), [all, positionsByNumber]);

    const start = async (enable: boolean) => {
        if (!settings) return;
        setSaving(true);
        try {
            const next = await api.updateSettings({
                ...settings,
                autopilot_enabled: enable,
                autopilot_protected_residues: listText,
            });
            setSettings(next);
            uiEvents.emit('autopilotChanged');
            toast('success', enable
                ? `自律ループを開始しました (禁止 ${all.length} 残基)`
                : '自律ループを止めました (実行中のジョブは最後まで走ります)');
            onClose();
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setSaving(false);
        }
    };

    if (!settings) return <Modal title="自律ループ" onClose={onClose}><Spinner /></Modal>;
    const on = !!status?.enabled;
    const chains = targetChains;

    return (
        <Modal title="自律ループ" onClose={onClose} wide className="autopilot-modal">
            <div className="settings ap-body">
                <section>
                    <h3>起点</h3>
                    {candidates.length === 0 ? (
                        <p className="small muted">完了した予測がまだありません。1 件でも予測を終えると、そこから枝分かれできます。</p>
                    ) : (
                        <>
                            <div className="row wrap end">
                                <Field label="この結果から枝分かれする" hint="選んだ結果と同じ系統を伸ばします">
                                    <select value={target?.id ?? ''} onChange={e => setTargetId(e.target.value)}>
                                        {candidates.map(j => <option key={j.id} value={j.id}>{j.title}</option>)}
                                    </select>
                                </Field>
                                <Field label="対象の鎖">
                                    <select value={chain} onChange={e => {
                                        setChain(e.target.value);
                                        if (target) void loadSuggestion(target.id, e.target.value);
                                    }}>
                                        {[chain, ...chains].filter((v, i, a) => v && a.indexOf(v) === i)
                                            .map(c => <option key={c} value={c}>{c}</option>)}
                                    </select>
                                </Field>
                                <Field label="実験名" hint="この名前のついた予測だけを起点にします。空なら全履歴">
                                    <input type="text" placeholder="空欄で全履歴" value={settings.autopilot_experiment}
                                        onChange={e => set('autopilot_experiment', e.target.value)} />
                                </Field>
                            </div>
                            {status && (
                                <p className="small muted">
                                    いまは{on ? '動作中' : '停止中'}。自律枠 {status.used_today}/{status.daily_budget || '∞'} 件 ·
                                    待機 {status.queued}/{status.max_queued} 件 · 空き {status.disk_free_gb} GB
                                    {status.blocked_reason && <span className="warn"> — {status.blocked_reason}</span>}
                                </p>
                            )}
                        </>
                    )}
                </section>

                <section>
                    <h3>進め方</h3>
                    <div className="row wrap end">
                        <Field label="探索の方針">
                            <select value={settings.autopilot_strategy} onChange={e => set('autopilot_strategy', e.target.value)}>
                                <option value="climb">山登り (常に最良から枝分かれ)</option>
                                <option value="walk">酔歩 (最新からそのまま続ける)</option>
                            </select>
                        </Field>
                        <Field label="候補の選び方">
                            <select value={settings.autopilot_selection} onChange={e => set('autopilot_selection', e.target.value)}>
                                <option value="greedy">良さそうな順</option>
                                <option value="explore">試していない位置を優先</option>
                            </select>
                        </Field>
                        <Field label="1回の枝分かれ数" hint="世代無制限なら 1">
                            <input type="number" min={0} max={10} value={settings.autopilot_max_variants_per_job}
                                onChange={e => set('autopilot_max_variants_per_job', Number(e.target.value))} />
                        </Field>
                        <Field label="世代の上限" hint="0 で無制限">
                            <input type="number" min={0} max={100} value={settings.autopilot_max_depth}
                                onChange={e => set('autopilot_max_depth', Number(e.target.value))} />
                        </Field>
                        <Field label="24時間の上限" hint="0 で無制限">
                            <input type="number" min={0} max={2000} value={settings.autopilot_daily_budget}
                                onChange={e => set('autopilot_daily_budget', Number(e.target.value))} />
                        </Field>
                    </div>
                </section>

                <section>
                    <h3>変更を禁止する残基</h3>
                    <p className="small muted">
                        スコアだけを見る探索は、機能を担う残基を潰すのが一番の近道になります。
                        この分子について分かっていること (データベースの注釈・界面・起点で乱れていた部分・ESM-2 の保存度) から
                        候補を出しました。チェックの付いたものが禁止リストに入ります。
                    </p>
                    {loadingSuggestion ? <Spinner /> : !suggestion ? (
                        <p className="small muted">起点を選ぶと候補を出します。</p>
                    ) : (
                        <>
                            <div className="kv small">
                                <span>鎖 {suggestion.chain} · {suggestion.length} 残基</span>
                                <span>注釈 {suggestion.sources.uniprot ?? 'なし'}</span>
                                <span>界面 {suggestion.sources.interface_residues}</span>
                                <span>乱れ {suggestion.sources.disordered_residues}</span>
                                <span>保存 {suggestion.sources.conserved_residues}</span>
                            </div>
                            {suggestion.notes.map(n => <p key={n} className="small warn">{n}</p>)}
                            <div className="row wrap">
                                <Button size="sm" onClick={() => setTicked(new Set(suggestion.positions.filter(p => p.default_on).map(p => p.position)))}>推奨に戻す</Button>
                                <Button size="sm" onClick={() => setTicked(new Set(suggestion.positions.map(p => p.position)))}>すべて選ぶ</Button>
                                <Button size="sm" onClick={() => setTicked(new Set())}>すべて外す</Button>
                                <span className="spacer" />
                                <span className="small muted">{all.length} 残基を禁止</span>
                            </div>
                            <div className="ap-list">
                                {suggestion.positions.length === 0 && (
                                    <p className="small muted">候補は見つかりませんでした。手入力の欄で指定できます。</p>
                                )}
                                {suggestion.positions.map(p => (
                                    <label key={p.position} className={`ap-row sev-${p.severity}`}>
                                        <input type="checkbox" checked={ticked.has(p.position)}
                                            onChange={e => setTicked(prev => {
                                                const next = new Set(prev);
                                                if (e.target.checked) next.add(p.position); else next.delete(p.position);
                                                return next;
                                            })} />
                                        <span className="ap-pos mono">{p.residue}{p.position}</span>
                                        <span className={`ob sev-${p.severity}`}>{SEVERITY_LABEL[p.severity] ?? p.severity}</span>
                                        <span className="ap-why small">{p.reasons.map(r => r.label).join(' · ')}</span>
                                    </label>
                                ))}
                            </div>
                        </>
                    )}
                    <Field label="手入力で足す" hint="番号でも K63 のような表記でもかまいません。カンマ区切り">
                        <input type="text" placeholder="例: K48, K63"
                            defaultValue={[...manual].sort((a, b) => a - b).join(', ')}
                            onChange={e => setManual(parsePositions(e.target.value))} />
                    </Field>
                    <p className="small muted">
                        実際に保存されるリスト: <span className="mono">{listText || '(なし)'}</span>
                    </p>
                </section>
            </div>

            <div className="settings-foot">
                {on ? (
                    <Button variant="danger" disabled={saving} onClick={() => void start(false)}>
                        {saving ? <Spinner size={12} /> : '自律を止める'}
                    </Button>
                ) : (
                    <Button variant="primary" disabled={saving || candidates.length === 0} onClick={() => void start(true)}>
                        {saving ? <Spinner size={12} /> : <><Icon name="play" size={13} /> この設定で自律を開始</>}
                    </Button>
                )}
                {on && (
                    <Button disabled={saving} onClick={() => void start(true)}>
                        {saving ? <Spinner size={12} /> : '設定だけ更新する'}
                    </Button>
                )}
                <Button variant="ghost" onClick={onClose}>閉じる</Button>
                <span className="spacer" />
                <span className="small muted">止めると新規投入だけが止まり、実行中のジョブは最後まで走ります</span>
            </div>
        </Modal>
    );
}
