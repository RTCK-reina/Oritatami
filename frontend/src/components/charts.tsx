import { memo, useEffect, useRef, useState } from 'react';
import { useAppliedTheme } from '../hooks';
import type { PaeSegment } from '../types';
import { AA_INFO, AMINO_ACIDS, llrColor, plddtColor } from '../workbench';
import { t } from '../i18n';

/** Predicted aligned error: low (confident) = dark green, high = white. */
export const PaeHeatmap = memo(function PaeHeatmap({ matrix, factor, segments, size }: { matrix: number[][]; factor: number; segments: PaeSegment[] | null; size: number }) {
    const ref = useRef<HTMLCanvasElement>(null);
    const [tip, setTip] = useState<string | null>(null);
    const n = matrix.length;
    useEffect(() => {
        const canvas = ref.current;
        if (!canvas || !n) return;
        canvas.width = n;
        canvas.height = n;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        const img = ctx.createImageData(n, n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const t = Math.min(1, matrix[i][j] / 30);
                const k = (i * n + j) * 4;
                img.data[k] = Math.round(0 + 255 * t);
                img.data[k + 1] = Math.round(83 + 172 * t);
                img.data[k + 2] = Math.round(34 + 221 * t);
                img.data[k + 3] = 255;
            }
        }
        ctx.putImageData(img, 0, 0);
    }, [matrix, n]);
    const chainAt = (tok: number) => segments?.find(s => tok >= s.start && tok < s.end);
    return (
        <div className="pae">
            <div className="pae-canvas-wrap">
                <canvas ref={ref} className="pae-canvas"
                    onMouseMove={e => {
                        const r = e.currentTarget.getBoundingClientRect();
                        const j = Math.floor(((e.clientX - r.left) / r.width) * n);
                        const i = Math.floor(((e.clientY - r.top) / r.height) * n);
                        if (i < 0 || j < 0 || i >= n || j >= n) return;
                        const ti = i * factor;
                        const tj = j * factor;
                        const ci = chainAt(ti);
                        const cj = chainAt(tj);
                        setTip(`${ci ? `${ci.chain}:${ti - ci.start + 1}` : ti + 1} → ${cj ? `${cj.chain}:${tj - cj.start + 1}` : tj + 1}: ${matrix[i][j].toFixed(1)} Å`);
                    }}
                    onMouseLeave={() => setTip(null)} />
                {segments && segments.slice(1).map(s => (
                    <div key={s.chain}>
                        <div className="pae-line v" style={{ left: `${(s.start / size) * 100}%` }} />
                        <div className="pae-line h" style={{ top: `${(s.start / size) * 100}%` }} />
                    </div>
                ))}
            </div>
            <div className="pae-side small">
                <div className="pae-scale"><span>0 Å</span><i /><span>30 Å</span></div>
                <p className="muted">{t('行の残基を基準に重ねたとき、列の残基の位置が何 Å ずれると予測されるか。チェーン間のブロックが暗いほど、相対配置に自信がある。')}</p>
                {segments && <div className="kv">{segments.map(s => <span key={s.chain}>{s.chain}: {s.end - s.start}</span>)}</div>}
                <div className="mono">{tip ?? ' '}</div>
            </div>
        </div>
    );
});

export const PlddtPlot = memo(function PlddtPlot({ series, mutated, onPick }: {
    series: { chain: string; values: number[] }[];
    mutated: Record<string, number[]>;
    onPick?: (chain: string, pos: number) => void;
}) {
    const W = 900;
    const H = 150;
    const pad = 28;
    return (
        <div className="plddt-plots">
            {series.map(s => {
                const n = s.values.length;
                const x = (i: number) => pad + (i / Math.max(1, n - 1)) * (W - pad - 8);
                const y = (v: number) => 8 + (1 - v / 100) * (H - 30);
                const path = s.values.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
                return (
                    <div key={s.chain} className="plddt-plot">
                        <div className="small">{t('チェーン')} {s.chain} {t('· 平均')} {(s.values.reduce((a, b) => a + b, 0) / Math.max(1, n)).toFixed(1)}</div>
                        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
                            onClick={e => {
                                if (!onPick) return;
                                const r = e.currentTarget.getBoundingClientRect();
                                const px = ((e.clientX - r.left) / r.width) * W;
                                const i = Math.round(((px - pad) / (W - pad - 8)) * (n - 1));
                                if (i >= 0 && i < n) onPick(s.chain, i + 1);
                            }}>
                            {[[90, 100, '#0053d6'], [70, 90, '#65cbf3'], [50, 70, '#ffdb13'], [0, 50, '#ff7d45']].map(([lo, hi, c]) => (
                                <rect key={String(lo)} x={pad} width={W - pad - 8} y={y(Number(hi))} height={y(Number(lo)) - y(Number(hi))} fill={String(c)} opacity={0.08} />
                            ))}
                            {[50, 70, 90].map(v => (
                                <g key={v}>
                                    <line x1={pad} x2={W - 8} y1={y(v)} y2={y(v)} style={{ stroke: 'var(--line-2)' }} strokeDasharray="3 4" />
                                    <text x={4} y={y(v) + 4} fontSize={10} style={{ fill: 'var(--muted)' }}>{v}</text>
                                </g>
                            ))}
                            {(mutated[s.chain] ?? []).map(p => (
                                <line key={p} x1={x(p - 1)} x2={x(p - 1)} y1={8} y2={H - 22} stroke="#ff3df2" strokeWidth={1.5} opacity={0.8} />
                            ))}
                            <path d={path} fill="none" style={{ stroke: 'var(--text)' }} strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
                            {s.values.map((v, i) => (n <= 400 ? <circle key={i} cx={x(i)} cy={y(v)} r={1.6} fill={plddtColor(v)} /> : null))}
                            {Array.from({ length: Math.floor(n / 50) + 1 }, (_, k) => k * 50).filter(v => v > 0 && v <= n).map(v => (
                                <text key={v} x={x(v - 1)} y={H - 6} fontSize={10} style={{ fill: 'var(--muted)' }} textAnchor="middle">{v}</text>
                            ))}
                        </svg>
                    </div>
                );
            })}
        </div>
    );
});

export const ScanHeatmap = memo(function ScanHeatmap({ sequence, matrix, highlight, onPick }: {
    sequence: string;
    matrix: number[][];
    highlight?: Set<number>;
    onPick?: (mutation: string) => void;
}) {
    const ref = useRef<HTMLCanvasElement>(null);
    const [tip, setTip] = useState<string | null>(null);
    const theme = useAppliedTheme();
    const cell = 9;
    const L = sequence.length;
    useEffect(() => {
        const canvas = ref.current;
        if (!canvas) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = L * cell * dpr;
        canvas.height = 20 * cell * dpr;
        canvas.style.width = `${L * cell}px`;
        canvas.style.height = `${20 * cell}px`;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.scale(dpr, dpr);
        const wtFill = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#0d1117';
        for (let i = 0; i < L; i++) {
            for (let j = 0; j < 20; j++) {
                ctx.fillStyle = AMINO_ACIDS[j] === sequence[i] ? wtFill : llrColor(matrix[i][j]);
                ctx.fillRect(i * cell, j * cell, cell - 1, cell - 1);
            }
            if (highlight?.has(i + 1)) {
                ctx.strokeStyle = '#ff3df2';
                ctx.lineWidth = 1.5;
                ctx.strokeRect(i * cell - 0.5, 0, cell, 20 * cell);
            }
        }
    }, [sequence, matrix, highlight, L, theme]);
    return (
        <div className="scan">
            <div className="scan-grid">
                <div className="scan-aa">{[...AMINO_ACIDS].map(a => <span key={a} style={{ height: cell }}>{a}</span>)}</div>
                <div className="scan-scroll">
                    <canvas ref={ref}
                        onMouseMove={e => {
                            const r = e.currentTarget.getBoundingClientRect();
                            const i = Math.floor((e.clientX - r.left) / cell);
                            const j = Math.floor((e.clientY - r.top) / cell);
                            if (i < 0 || j < 0 || i >= L || j >= 20) return;
                            const aa = AMINO_ACIDS[j];
                            setTip(aa === sequence[i] ? `${sequence[i]}${i + 1} ${t('(元の残基)')}` : `${sequence[i]}${i + 1}${aa} ${AA_INFO[aa].name}  LLR ${matrix[i][j] > 0 ? '+' : ''}${matrix[i][j].toFixed(2)}`);
                        }}
                        onMouseLeave={() => setTip(null)}
                        onClick={e => {
                            const r = e.currentTarget.getBoundingClientRect();
                            const i = Math.floor((e.clientX - r.left) / cell);
                            const j = Math.floor((e.clientY - r.top) / cell);
                            if (i < 0 || j < 0 || i >= L || j >= 20 || AMINO_ACIDS[j] === sequence[i]) return;
                            onPick?.(`${sequence[i]}${i + 1}${AMINO_ACIDS[j]}`);
                        }} />
                    <div className="scan-ruler" style={{ width: L * cell }}>
                        {Array.from({ length: Math.floor(L / 10) }, (_, k) => (k + 1) * 10).map(v => (
                            <span key={v} style={{ left: (v - 1) * cell }}>{v}</span>
                        ))}
                    </div>
                </div>
            </div>
            <div className="scan-foot small">
                <span className="scale"><i style={{ background: llrColor(-6) }} />{t('不自然')} <i style={{ background: llrColor(0) }} />0 <i style={{ background: llrColor(6) }} />{t('自然')}</span>
                <span className="mono">{tip ?? t('セルをクリックすると、その変異を作業台に入れます')}</span>
            </div>
        </div>
    );
});

/** How the search moved: every attempt as a dot, the running best as a line.
 *
 * The question this answers is the one a long run makes hard to see from a ranked list —
 * is it still climbing, or has it been flat for two hundred predictions? */
export const ProgressChart = memo(function ProgressChart(
    { points, label, onPick, selected }:
    { points: { id: string; score: number; at: number; title: string; depth: number }[];
      label: string; onPick?: (id: string) => void; selected?: string | null },
) {
    const [tip, setTip] = useState<string | null>(null);
    const W = 520, H = 168, PAD = { l: 44, r: 10, t: 12, b: 34 };
    if (points.length < 2) return <p className="small muted">{t('推移を描くには予測が 2 件以上必要です。')}</p>;

    const lo = Math.min(...points.map(p => p.score));
    const hi = Math.max(...points.map(p => p.score));
    const span = hi - lo || 1;
    const x = (i: number) => PAD.l + (i / (points.length - 1)) * (W - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - (v - lo) / span) * (H - PAD.t - PAD.b);

    let best = -Infinity;
    const bestLine = points.map((p, i) => {
        best = Math.max(best, p.score);
        return `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(best).toFixed(1)}`;
    }).join(' ');

    const ticks = [lo, lo + span / 2, hi];
    return (
        <div className="progress-chart">
            <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label}${t('の推移')}`}>
                {ticks.map(v => (
                    <g key={v}>
                        <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className="grid" />
                        <text x={PAD.l - 6} y={y(v) + 3} className="axis" textAnchor="end">{v.toFixed(1)}</text>
                    </g>
                ))}
                <path d={bestLine} className="best-line" />
                {points.map((p, i) => (
                    <circle key={p.id} cx={x(i)} cy={y(p.score)} r={p.id === selected ? 4 : 2.2}
                        className={`dot${p.id === selected ? ' on' : ''}`}
                        onMouseEnter={() => setTip(`${p.title} — ${p.score.toFixed(2)} ${t('(世代')} ${p.depth})`)}
                        onMouseLeave={() => setTip(null)}
                        onClick={() => onPick?.(p.id)} />
                ))}
                <text x={PAD.l} y={H - 6} className="axis">{t('古い')}</text>
                <text x={W - PAD.r} y={H - 6} className="axis" textAnchor="end">{t('新しい')}</text>
            </svg>
            <div className="small muted mono">{tip ?? `${label} ${t('— 折れ線はその時点での最高値')}`}</div>
        </div>
    );
});

/** Which positions the search keeps rewriting, counted over every attempt.
 *
 * This is the view that made the pLDDT exploit obvious: 91% of one run's attempts had
 * rewritten the same terminal glycine. */
export const MutationSpectrum = memo(function MutationSpectrum(
    { counts, length, protected: fixed }:
    { counts: Map<number, number>; length: number; protected?: Set<number> },
) {
    const [tip, setTip] = useState<string | null>(null);
    if (!length) return null;
    const max = Math.max(1, ...counts.values());
    return (
        <div className="spectrum">
            <div className="spectrum-bars" style={{ gridTemplateColumns: `repeat(${length}, 1fr)` }}>
                {Array.from({ length }, (_, i) => {
                    const pos = i + 1;
                    const n = counts.get(pos) ?? 0;
                    const isFixed = fixed?.has(pos);
                    return (
                        <span key={pos} className={`bar${isFixed ? ' fixed' : ''}`}
                            style={{ ['--h' as string]: `${Math.round((n / max) * 100)}%` }}
                            onMouseEnter={() => setTip(`${pos} ${t('番:')} ${n} ${t('回')}${isFixed ? t(' (変更禁止)') : ''}`)}
                            onMouseLeave={() => setTip(null)} />
                    );
                })}
            </div>
            <div className="small muted mono">{tip ?? `${t('残基ごとの変更回数 (最多')} ${max} ${t('回)')}`}</div>
        </div>
    );
});

/** MSA support per column: the share of hits carrying a residue (area) and how often it
 * matches the query (line). Thin coverage is where the alignment stops backing the model. */
export const MsaCoverage = memo(function MsaCoverage({ coverage, identity }: { coverage: number[]; identity: number[] }) {
    const [tip, setTip] = useState<string | null>(null);
    const n = coverage.length;
    const W = 520, H = 96, PAD = { l: 44, r: 10, t: 10, b: 20 };
    if (!n) return null;
    const x = (i: number) => PAD.l + (i / Math.max(1, n - 1)) * (W - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - Math.min(1, Math.max(0, v))) * (H - PAD.t - PAD.b);
    const area = `M${x(0).toFixed(1)},${y(0).toFixed(1)} ` +
        coverage.map((v, i) => `L${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ') +
        ` L${x(n - 1).toFixed(1)},${y(0).toFixed(1)} Z`;
    const idLine = identity.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
    const mean = (arr: number[]) => Math.round((arr.reduce((a, b) => a + b, 0) / n) * 100);
    return (
        <div className="progress-chart msa-coverage">
            <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={t('MSA の被覆率')}
                onMouseMove={e => {
                    const svg = e.currentTarget;
                    const r = svg.getBoundingClientRect();
                    const i = Math.round((((e.clientX - r.left) / r.width) * W - PAD.l) / (W - PAD.l - PAD.r) * (n - 1));
                    setTip(i >= 0 && i < n ? `${t('列')} ${i + 1}${t(': 被覆')} ${(coverage[i] * 100).toFixed(0)}${t('% · 一致')} ${(identity[i] * 100).toFixed(0)}%` : null);
                }}
                onMouseLeave={() => setTip(null)}>
                {[0.5, 1].map(v => (
                    <g key={v}>
                        <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className="grid" />
                        <text x={PAD.l - 6} y={y(v) + 3} className="axis" textAnchor="end">{v * 100}%</text>
                    </g>
                ))}
                <path d={area} className="msa-fill" />
                <path d={idLine} className="best-line" />
                <text x={PAD.l} y={H - 4} className="axis">1</text>
                <text x={W - PAD.r} y={H - 4} className="axis" textAnchor="end">{n}</text>
            </svg>
            <div className="small muted mono">{tip ?? `${t('平均: 被覆')} ${mean(coverage)}${t('% · 一致')} ${mean(identity)}%`}</div>
        </div>
    );
});
