import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, errorMessage } from '../api';
import { MutationSpectrum, ProgressChart } from './charts';
import type { AutopilotStatus, Leaderboard, LeaderboardMetric, LeaderboardRow } from '../api';
import { useStore } from '../store';
import { Button, Empty, Icon, Spinner } from './ui';

const METRIC_LABEL: Record<LeaderboardMetric, string> = {
    mean_plddt: '平均 pLDDT',
    core_plddt: 'コア pLDDT',
    confidence_score: '信頼度スコア',
    iptm: 'ipTM',
    ptm: 'pTM',
    complex_plddt: '複合体 pLDDT',
};

const ORIGIN_BADGE: Record<string, { label: string; cls: string; title: string }> = {
    user: { label: '手動', cls: 'ob-user', title: 'あなたが投入したジョブ' },
    autopilot_variant: { label: '自動', cls: 'ob-auto', title: 'オートパイロットが提案から自動生成した変異体' },
    pdb_watch: { label: 'PDB', cls: 'ob-pdb', title: 'PDB ウォッチャーが新着構造から自動投入' },
};

/** Metrics where a larger number is better. All current ones are; kept explicit for affinity later. */
const HIGHER_IS_BETTER = true;

function fmtScore(v: number | null, metric: LeaderboardMetric): string {
    if (v === null || v === undefined) return '—';
    return metric === 'mean_plddt' || metric === 'core_plddt' || metric === 'complex_plddt' ? v.toFixed(1) : v.toFixed(3);
}

function Delta({ value, metric, comparable = null }:
    { value: number | null; metric: LeaderboardMetric; comparable?: boolean | null }) {
    if (value === null || value === undefined) return <span className="lb-delta muted">—</span>;
    const good = HIGHER_IS_BETTER ? value > 0 : value < 0;
    const flat = Math.abs(value) < (metric === 'mean_plddt' || metric === 'core_plddt' || metric === 'complex_plddt' ? 0.05 : 0.0005);
    const cls = flat ? 'flat' : good ? 'up' : 'down';
    const sign = value > 0 ? '+' : '';
    const digits = metric === 'mean_plddt' || metric === 'core_plddt' || metric === 'complex_plddt' ? 1 : 3;
    // A delta against a parent computed under other conditions measures the settings
    // change, not the mutation. Shown, because hiding it invites the same comparison by
    // hand — but struck through, so it is never read as a result.
    const stale = comparable === false;
    return (
        <span className={`lb-delta ${stale ? 'stale' : cls}`}
            title={stale ? '親と計算条件が違うため、この差は変異の効果ではありません' : '親ジョブとの差'}>
            {flat ? '±0' : `${sign}${value.toFixed(digits)}`}
        </span>
    );
}

function StatusStrip({ status }: { status: AutopilotStatus | null }) {
    if (!status) return null;
    const unlimited = status.daily_budget === 0;
    const pct = unlimited
        ? Math.min(100, Math.round((status.queued / Math.max(1, status.max_queued)) * 100))
        : Math.min(100, Math.round((status.used_today / Math.max(1, status.daily_budget)) * 100));
    const continuous = status.max_depth === 0;
    return (
        <div className={`lb-status ${status.accepting ? '' : 'blocked'}`}>
            <span className="lb-status-main">
                <Icon name={status.accepting ? 'layers' : 'close'} size={12} />
                {!status.enabled ? 'オートパイロット停止中'
                    : continuous ? '連続稼働中' : 'オートパイロット稼働中'}
            </span>
            {unlimited ? (
                <span className="lb-budget" title={`順番待ち ${status.queued} / 上限 ${status.max_queued} 件`}>
                    待機 {status.queued}/{status.max_queued}
                    <span className="lb-bar"><i style={{ width: `${pct}%` }} /></span>
                </span>
            ) : (
                <span className="lb-budget" title={`本日の自律ジョブ ${status.used_today} / ${status.daily_budget} 件`}>
                    自律枠 {status.used_today}/{status.daily_budget}
                    <span className="lb-bar"><i style={{ width: `${pct}%` }} /></span>
                </span>
            )}
            <span className="small muted">
                {continuous ? '世代 無制限' : `世代 上限 ${status.max_depth}`} · 本日 {status.used_today} 件 · 空き {status.disk_free_gb} GB
            </span>
            {status.blocked_reason && <span className="small warn">{status.blocked_reason}</span>}
        </div>
    );
}

type View = 'rank' | 'tree' | 'progress' | 'pareto';

interface TreeNode {
    row: LeaderboardRow;
    children: TreeNode[];
}

/** Group rows into parent → child trees. A row whose parent is not in the set becomes a root. */
function buildTree(rows: LeaderboardRow[]): TreeNode[] {
    const nodes = new Map<string, TreeNode>(rows.map(r => [r.id, { row: r, children: [] }]));
    const roots: TreeNode[] = [];
    for (const node of nodes.values()) {
        const parent = node.row.parent_id ? nodes.get(node.row.parent_id) : undefined;
        if (parent) parent.children.push(node);
        else roots.push(node);
    }
    const byTime = (a: TreeNode, b: TreeNode) => a.row.created_at - b.row.created_at;
    const sortDeep = (list: TreeNode[]) => {
        list.sort(byTime);
        list.forEach(n => sortDeep(n.children));
    };
    sortDeep(roots);
    // Show the branches that actually have descendants first; lone jobs are less interesting.
    const size = (n: TreeNode): number => 1 + n.children.reduce((a, c) => a + size(c), 0);
    return roots.sort((a, b) => size(b) - size(a) || b.row.created_at - a.row.created_at);
}

export function LeaderboardPanel() {
    const { openJob, selectedJobId, jobs, toast } = useStore();
    const [metric, setMetric] = useState<LeaderboardMetric>('mean_plddt');
    const [view, setView] = useState<View>('rank');
    const [data, setData] = useState<Leaderboard | null>(null);
    const [status, setStatus] = useState<AutopilotStatus | null>(null);
    const [loading, setLoading] = useState(true);
    const [onlyImproved, setOnlyImproved] = useState(false);

    const load = useCallback(async (m: LeaderboardMetric) => {
        setLoading(true);
        try {
            const [lb, st] = await Promise.all([api.leaderboard(m), api.autopilotStatus()]);
            setData(lb);
            setStatus(st);
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setLoading(false);
        }
    }, [toast]);

    useEffect(() => { void load(metric); }, [load, metric]);
    // Refresh when a job finishes: the job list revision changes on every state transition.
    const doneCount = jobs.filter(j => j.status === 'succeeded').length;
    useEffect(() => { void load(metric); }, [doneCount, load, metric]);

    const rows = useMemo(() => {
        if (!data) return [];
        if (!onlyImproved) return data.rows;
        return data.rows.filter(r => (r[`delta_${metric}` as keyof LeaderboardRow] as number | null ?? 0) > 0);
    }, [data, onlyImproved, metric]);

    const tree = useMemo(() => buildTree(rows), [rows]);

    /** Attempts in the order they were run, for the progress chart. */
    const progress = useMemo(() => (data?.rows ?? [])
        .filter(r => (r[metric] as number | null) !== null)
        .slice()
        .sort((a, b) => (a.created_at ?? 0) - (b.created_at ?? 0))
        .map(r => ({ id: r.id, score: r[metric] as number, at: r.created_at ?? 0,
                     title: r.title ?? r.id, depth: r.autopilot_depth })), [data, metric]);

    const improvedCount = useMemo(() => (data?.rows ?? [])
        .filter(r => ((r[`delta_${metric}` as keyof LeaderboardRow] as number | null) ?? 0) > 0).length,
        [data, metric]);

    /** How often each position was rewritten, across every attempt. */
    const spectrum = useMemo(() => {
        const counts = new Map<number, number>();
        let length = 0;
        for (const r of data?.rows ?? []) {
            for (const code of r.mutations ?? []) {
                const m = /(\d+)/.exec(code);
                if (!m) continue;
                const pos = Number(m[1]);
                counts.set(pos, (counts.get(pos) ?? 0) + 1);
                length = Math.max(length, pos);
            }
        }
        const fixed = new Set<number>();
        for (const tok of (status?.protected_residues ?? '').split(/[,;\s]+/)) {
            const m = /(\d+)/.exec(tok);
            if (m) fixed.add(Number(m[1]));
        }
        return { counts, length, fixed };
    }, [data, status]);

    const front = useMemo(
        () => (data?.rows ?? []).filter(r => r.pareto)
            .sort((a, b) => a.mutations.length - b.mutations.length),
        [data]);

    const scoreOf = (r: LeaderboardRow) => r[metric] as number | null;
    const deltaOf = (r: LeaderboardRow) => r[`delta_${metric}` as keyof LeaderboardRow] as number | null;

    const Row = ({ r, depth = 0 }: { r: LeaderboardRow; depth?: number }) => {
        const badge = ORIGIN_BADGE[r.origin] ?? ORIGIN_BADGE.user;
        const best = data?.best_id === r.id;
        return (
            <button
                type="button"
                className={`lb-row ${selectedJobId === r.id ? 'is-selected' : ''} ${best ? 'is-best' : ''}`}
                style={depth ? { paddingLeft: 8 + depth * 16 } : undefined}
                onClick={() => openJob(r.id)}
                title={r.parent_title ? `親: ${r.parent_title}` : undefined}
            >
                {view === 'rank' && <span className="lb-rank">{r.rank}</span>}
                {view === 'tree' && depth > 0 && <span className="lb-branch" aria-hidden>└</span>}
                <span className="lb-title">
                    <span className="lb-name">{r.title ?? r.id}</span>
                    <span className="lb-meta">
                        <span className={`ob ${badge.cls}`} title={badge.title}>{badge.label}</span>
                        {r.mutations.length > 0 && <span className="lb-muts mono">{r.mutations.join(' ')}</span>}
                        {best && <span className="ob ob-best" title="この指標での最良">最良</span>}
                        {data?.conditions?.mixed && r.regime && (
                            <span className="ob ob-regime" title={`計算条件: ${r.regime_label}`}>{r.regime}</span>
                        )}
                    </span>
                </span>
                <span className="lb-score mono">{fmtScore(scoreOf(r), metric)}</span>
                <Delta value={deltaOf(r)} metric={metric} comparable={r.comparable_to_parent} />
            </button>
        );
    };

    const TreeRows = ({ node, depth }: { node: TreeNode; depth: number }) => (
        <>
            <Row r={node.row} depth={depth} />
            {node.children.map(c => <TreeRows key={c.row.id} node={c} depth={depth + 1} />)}
        </>
    );

    return (
        <div className="panel leaderboard">
            <div className="panel-head lb-head">
                <select value={metric} onChange={e => setMetric(e.target.value as LeaderboardMetric)} aria-label="並べ替える指標">
                    {(data?.metrics ?? ['mean_plddt']).map(m => (
                        <option key={m} value={m}>{METRIC_LABEL[m] ?? m}</option>
                    ))}
                </select>
                <div className="seg seg-sm" role="radiogroup" aria-label="表示形式">
                    <button type="button" role="radio" aria-checked={view === 'rank'}
                        className={view === 'rank' ? 'active' : ''} onClick={() => setView('rank')}>順位</button>
                    <button type="button" role="radio" aria-checked={view === 'tree'}
                        className={view === 'tree' ? 'active' : ''} onClick={() => setView('tree')}>系統</button>
                    <button type="button" role="radio" aria-checked={view === 'pareto'}
                        className={view === 'pareto' ? 'active' : ''} onClick={() => setView('pareto')}>費用対効果</button>
                    <button type="button" role="radio" aria-checked={view === 'progress'}
                        className={view === 'progress' ? 'active' : ''} onClick={() => setView('progress')}>推移</button>
                </div>
                <Button size="sm" variant="ghost" onClick={() => void load(metric)} title="再読み込み">
                    <Icon name="retry" size={13} />
                </Button>
            </div>

            <StatusStrip status={status} />

            {data?.conditions?.mixed && (
                <div className="lb-mixed small">
                    <strong>計算条件が混ざっています。</strong>
                    {data.conditions.regimes.map(g => `${g.label} ${g.count} 件`).join(' / ')}。
                    Boltz が返すのは分子の性質ではなく予測の自信なので、MSA の取り方やサンプル数が違うと
                    同じ配列でも点が動きます (この履歴では最大 1.9、改善の判定幅 0.5 の約 4 倍)。
                    条件をまたいだ差は下のように打ち消し線で表示し、改善件数からも除いてあります。
                </div>
            )}

            <label className="inline-toggle small lb-filter">
                <input type="checkbox" checked={onlyImproved} onChange={e => setOnlyImproved(e.target.checked)} />
                親を上回ったものだけ表示
                {data && <span className="muted"> ({data.improved_count} / {data.count}
                    {data.returned < data.count ? ` — 上位 ${data.returned} 件を表示` : ''})</span>}
            </label>

            <div className="panel-scroll lb-body">
                {loading && !data ? <Spinner /> : rows.length === 0 ? (
                    <Empty>
                        {onlyImproved
                            ? '親を上回った変異体はまだありません。'
                            : '完了した構造予測がまだありません。予測を実行すると、ここにスコア順で並びます。'}
                    </Empty>
                ) : view === 'rank' ? (
                    rows.map(r => <Row key={r.id} r={r} />)
                ) : view === 'tree' ? (
                    tree.map(n => <TreeRows key={n.row.id} node={n} depth={0} />)
                ) : view === 'pareto' ? (
                    <div className="lb-pareto">
                        <p className="small muted">
                            変異の数ごとに、そのスコアを超えるものがもっと少ない変異では出ていない、という組み合わせだけを並べています。
                            1 本の順位表では「16 変異で +0.9」と「2 変異で +0.7」が隣に並んでしまい、同じ成果に見えてしまうため。
                            比べるのは同じ系統の中だけです (別の分子と変異数を比べても意味がないので)。
                        </p>
                        {front.length === 0 ? <Empty>まだ前線を引けるだけの結果がありません。</Empty>
                            : front.map(r => (
                                <div key={r.id} className="lb-front-row">
                                    <span className="lb-front-n mono">{r.mutations.length} 変異</span>
                                    <Row r={r} />
                                </div>
                            ))}
                    </div>
                ) : (
                    <div className="lb-progress">
                        <ProgressChart points={progress} label={METRIC_LABEL[metric]}
                            onPick={openJob} selected={selectedJobId} />
                        <h4 className="small">どの残基を書き換えてきたか</h4>
                        <MutationSpectrum counts={spectrum.counts} length={spectrum.length}
                            protected={spectrum.fixed} />
                        <p className="small muted">
                            {progress.length} 件の試行のうち {improvedCount} 件が親を上回りました。
                            最高値は {progress.length ? Math.max(...progress.map(p => p.score)).toFixed(2) : '—'}。
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}
