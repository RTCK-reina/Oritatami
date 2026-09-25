import { useEffect, useRef, useState } from 'react';
import { api, errorMessage } from '../api';
import { useStore } from '../store';
import type { PredictJob, ViewSource } from '../types';
import { uiEvents } from '../uiEvents';
import { viewerBus } from '../viewer/bus';
import type { ColorMode, LoadItem, RepStyle, Viewer } from '../viewer/Viewer';
import { EXAMPLES } from '../examples';
import { Button, Icon, Kbd, Menu, Modal, Spinner } from './ui';
import { t } from '../i18n';

const BACKGROUND = { dark: 0x0d1117, light: 0xffffff };

function mutationHighlights(job: PredictJob): { chain: string; residues: number[] }[] {
    return (job.result?.chains ?? [])
        .filter(c => c.mutations.length)
        .map(c => ({ chain: c.chain, residues: c.mutations.map(m => Number(m.replace(/^[A-Z]/, '').replace(/[A-Z]$/, ''))) }));
}

export function ViewerPanel({ theme, fullscreen, onToggleFullscreen }: { theme: 'dark' | 'light'; fullscreen: boolean; onToggleFullscreen: () => void }) {
    const store = useStore();
    const { view, getJob, toast, setSelectedResidue, openExample, showLeftTab } = store;
    const host = useRef<HTMLDivElement>(null);
    const viewer = useRef<Viewer | null>(null);
    const [ready, setReady] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [hover, setHover] = useState<string | null>(null);
    const [style, setStyle] = useState<RepStyle>('cartoon');
    const [color, setColor] = useState<ColorMode>('plddt');
    const [pocket, setPocket] = useState(true);
    const [spin, setSpin] = useState(false);
    const [title, setTitle] = useState<string>('');
    const [modelIndex, setModelIndex] = useState(0);
    const [modelCount, setModelCount] = useState(1);
    const [shot, setShot] = useState<string | null>(null);
    const [plddtAvailable, setPlddtAvailable] = useState(false);
    // read inside the loader without re-running it: a color change is applied in place (keeps the camera)
    const colorRef = useRef(color);
    colorRef.current = color;
    const themeRef = useRef(theme);
    themeRef.current = theme;

    // Mol* is large: load it after the rest of the UI has painted.
    useEffect(() => {
        let disposed = false;
        let v: Viewer | null = null;
        const offs: (() => void)[] = [];
        const el = host.current;
        if (!el) return;
        import('../viewer/Viewer')
            .then(async mod => {
                if (disposed) return;
                v = new mod.Viewer();
                offs.push(v.onHover(setHover));
                offs.push(v.onError(message => toast('error', `${t('3D ビューア:')} ${message}`)));
                offs.push(v.onResidueClick(ref => {
                    if (ref && ref.isPolymer && ref.structureIndex <= 0) setSelectedResidue({ chain: ref.chain, position: ref.labelSeqId });
                }));
                await v.mount(el, BACKGROUND[themeRef.current]);
                if (disposed) {
                    v.dispose();
                    return;
                }
                viewer.current = v;
                viewerBus.highlight = (chain, ids) => v?.highlightResidues(0, chain, ids);
                viewerBus.focus = (chain, ids) => v?.focusResidues(0, chain, ids);
                setReady(true);
            })
            .catch(e => !disposed && setError(`${t('3D ビューアを起動できませんでした:')} ${errorMessage(e)}`));
        const ro = new ResizeObserver(() => viewer.current?.handleResize());
        ro.observe(el);
        return () => {
            disposed = true;
            ro.disconnect();
            offs.forEach(off => off());
            viewerBus.highlight = () => undefined;
            viewerBus.focus = () => undefined;
            v?.dispose();
            viewer.current = null;
        };
    }, [setSelectedResidue, toast]);

    useEffect(() => {
        if (ready) void viewer.current?.setBackground(BACKGROUND[theme]);
    }, [theme, ready]);
    // ResizeObserver covers this too, but it is delivered late when the window is in the background
    useEffect(() => {
        const t = window.setTimeout(() => viewer.current?.handleResize(), 60);
        return () => window.clearTimeout(t);
    }, [fullscreen]);

    useEffect(() => {
        if (view?.kind === 'job') setModelIndex(view.model);
    }, [view]);

    useEffect(() => {
        const v = viewer.current;
        if (!ready || !v) return;
        if (!view) {
            void v.clear();
            setTitle('');
            return;
        }
        let alive = true;
        const colorNow = colorRef.current;
        const build = async (src: ViewSource): Promise<LoadItem[]> => {
            if (src.kind === 'job') {
                const job = await getJob(src.jobId);
                if (job.kind !== 'predict' || !job.result) throw new Error(t('このジョブには構造がありません'));
                const model = job.result.models.find(m => m.index === modelIndex) ?? job.result.models[0];
                setModelCount(job.result.models.length);
                setTitle(`${job.title}${job.result.models.length > 1 ? ` ${t('· モデル')} ${model.index + 1}` : ''}`);
                setPlddtAvailable(true);
                return [{ url: api.jobFileUrl(job.id, model.file), format: 'mmcif', label: job.title, color: colorNow,
                    highlights: mutationHighlights(job) }];
            }
            if (src.kind === 'import') {
                setModelCount(1);
                setTitle(`${src.item.title} (${src.item.source.db} ${src.item.source.id})`);
                setPlddtAvailable(src.item.plddt_in_bfactor);
                return [{ url: src.item.url, format: src.item.format, label: src.item.title,
                    color: colorNow === 'plddt' && !src.item.plddt_in_bfactor ? 'chain' : colorNow }];
            }
            const [a, b] = await Promise.all([getJob(src.fixed.jobId), getJob(src.moving.jobId)]);
            if (a.kind !== 'predict' || !a.result || b.kind !== 'predict' || !b.result) throw new Error(t('比較できる結果がありません'));
            setModelCount(1);
            setTitle(`${t('比較:')} ${src.fixed.title} ⇄ ${src.moving.title} · RMSD ${src.result.rmsd.toFixed(2)} Å`);
            setPlddtAvailable(false);
            return [
                { url: api.jobFileUrl(a.id, a.result.models[0].file), format: 'mmcif', label: a.title, color: 'uniform', uniformColor: 0x58d5c9, highlights: mutationHighlights(a) },
                { url: src.result.url, format: 'mmcif', label: b.title, color: 'uniform', uniformColor: 0xf59e6b, highlights: mutationHighlights(b) },
            ];
        };
        setLoading(true);
        setError(null);
        build(view)
            .then(items => (alive ? v.load(items) : undefined))
            .catch(e => {
                if (alive) {
                    setError(errorMessage(e));
                    toast('error', `${t('表示に失敗:')} ${errorMessage(e)}`);
                }
            })
            .finally(() => alive && setLoading(false));
        return () => { alive = false; };
    }, [ready, view, modelIndex, toast, getJob]);

    useEffect(() => {
        if (ready && viewer.current?.items.length) void viewer.current.setColor(color);
    }, [color, ready]);
    useEffect(() => {
        if (ready && viewer.current) void viewer.current.setStyle(style);
    }, [style, ready]);
    useEffect(() => {
        if (ready && viewer.current) void viewer.current.setShowPocket(pocket);
    }, [pocket, ready]);

    const selected = store.selectedResidue;
    useEffect(() => {
        if (selected && viewer.current && view?.kind !== 'compare') viewer.current.focusResidues(0, selected.chain, [selected.position]);
    }, [selected, view?.kind]);

    const takeShot = async (resolution: 'viewport' | 'ultra-hd', transparent = false) => {
        try {
            if (!viewer.current) throw new Error(t('3D ビューアの準備ができていません'));
            setShot(await viewer.current.screenshot({ resolution, transparent }));
        } catch (e) {
            toast('error', errorMessage(e));
        }
    };
    const toggleSpin = async () => setSpin(await (viewer.current?.toggleSpin() ?? Promise.resolve(false)));

    useEffect(() => uiEvents.on('viewerAction', action => {
        if (action === 'reset') viewer.current?.resetCamera();
        if (action === 'spin') void toggleSpin();
        if (action === 'screenshot') void takeShot('viewport');
        if (action === 'pocket') setPocket(p => !p);
    }));
    // Keyboard cycling of the two dropdowns. The order matches the <select> options, so
    // pressing the key repeatedly walks the menu the way the eye expects.
    const STYLES: RepStyle[] = ['cartoon', 'surface', 'putty', 'ball-and-stick', 'spacefill'];
    const COLORS: ColorMode[] = ['plddt', 'chain', 'rainbow', 'secondary', 'hydrophobicity'];
    const cycle = <T,>(list: T[], current: T, dir: 'next' | 'prev'): T =>
        list[(list.indexOf(current) + (dir === 'next' ? 1 : list.length - 1)) % list.length];
    useEffect(() => uiEvents.on('viewerStyle', dir => setStyle(c => cycle(STYLES, c, dir))));
    useEffect(() => uiEvents.on('viewerColor', dir => setColor(c => {
        const next = cycle(COLORS, c, dir);
        return next === 'plddt' && !plddtAvailable ? cycle(COLORS, next, dir) : next;
    })));

    const shotName = `${(title || 'oritatami').replace(/[\\/:*?"<>|]+/g, '_').slice(0, 60)}.png`;
    const hasStructure = !!view;

    return (
        <div className="panel viewer-panel">
            <div className="viewer-toolbar">
                <span className="viewer-title" title={title}>{title || t('3D ビューア')}</span>
                {modelCount > 1 && view?.kind === 'job' && (
                    <select value={modelIndex} onChange={e => setModelIndex(Number(e.target.value))} title={t('サンプル (信頼度の高い順)')} aria-label={t('モデル')}>
                        {Array.from({ length: modelCount }, (_, i) => <option key={i} value={i}>{t('モデル')} {i + 1}</option>)}
                    </select>
                )}
                <select value={style} onChange={e => setStyle(e.target.value as RepStyle)} title={t('表示のしかた')} aria-label={t('表示')} disabled={!hasStructure}>
                    <option value="cartoon">{t('リボン')}</option>
                    <option value="surface">{t('表面')}</option>
                    <option value="putty">{t('パテ')}</option>
                    <option value="ball-and-stick">{t('原子 (棒球)')}</option>
                    <option value="spacefill">{t('原子 (空間充填)')}</option>
                </select>
                <select value={color} onChange={e => setColor(e.target.value as ColorMode)} title={t('色分け')} aria-label={t('色')} disabled={!hasStructure || view?.kind === 'compare'}>
                    <option value="plddt" disabled={!plddtAvailable}>{t('pLDDT (信頼度)')}</option>
                    <option value="chain">{t('チェーン')}</option>
                    <option value="rainbow">{t('N→C 虹色')}</option>
                    <option value="secondary">{t('二次構造')}</option>
                    <option value="hydrophobicity">{t('疎水性')}</option>
                </select>
                <label className="inline-toggle small" title={t('リガンドから 4.5 Å 以内の残基を表示')}>
                    <input type="checkbox" checked={pocket} onChange={e => setPocket(e.target.checked)} /> {t('ポケット')}
                </label>
                <span className="toolbar-sep" />
                <button type="button" className={`icon-btn ${spin ? 'on' : ''}`} onClick={() => void toggleSpin()} title={t('自動回転')} aria-pressed={spin} disabled={!hasStructure}><Icon name="rotate" /></button>
                <button type="button" className="icon-btn" onClick={() => viewer.current?.resetCamera()} title={t('視点をリセット (R)')} disabled={!hasStructure}><Icon name="target" /></button>
                <Menu label={<Icon name="camera" />} title={t('画像')} items={[
                    { label: t('画像を撮る (画面の解像度)'), onSelect: () => void takeShot('viewport') },
                    { label: t('高解像度 (4K) で撮る'), onSelect: () => void takeShot('ultra-hd') },
                    { label: t('背景を透明にして撮る (4K)'), onSelect: () => void takeShot('ultra-hd', true) },
                ]} />
                <button type="button" className="icon-btn" onClick={onToggleFullscreen} title={fullscreen ? t('元の配置に戻す (F)') : t('ビューアを大きく表示 (F)')} aria-pressed={fullscreen}>
                    <Icon name={fullscreen ? 'shrink' : 'expand'} />
                </button>
            </div>
            <div className="viewer-host-wrap">
                <div ref={host} className="viewer-host" />
                {(loading || (!ready && !error)) && <div className="viewer-overlay"><Spinner size={22} /></div>}
                {error && <div className="viewer-overlay error">{error}</div>}
                {ready && !view && !error && (
                    <div className="viewer-empty">
                        <p>{t('ここに立体構造が表示されます')}</p>
                        <div className="row wrap center">
                            <Button size="sm" onClick={() => void openExample(EXAMPLES.find(e => e.id === 'gfp-afdb') ?? EXAMPLES[0])}>{t('GFP を眺めてみる')}</Button>
                            <Button size="sm" variant="ghost" onClick={() => showLeftTab('jobs')}>{t('過去の結果を開く')}</Button>
                        </div>
                        <p className="small muted">{t('ドラッグで回転 · スクロールで拡大 · 右ドラッグで移動 · 残基をクリックで選択')}</p>
                    </div>
                )}
                {hover && <div className="hover-label">{hover}</div>}
                {color === 'plddt' && plddtAvailable && view && view.kind !== 'compare' && (
                    <div className="legend">
                        <span className="muted">pLDDT</span>
                        <span><i style={{ background: '#0053d6' }} />&gt;90</span>
                        <span><i style={{ background: '#65cbf3' }} />70–90</span>
                        <span><i style={{ background: '#ffdb13' }} />50–70</span>
                        <span><i style={{ background: '#ff7d45' }} />&lt;50</span>
                    </div>
                )}
                {view?.kind === 'compare' && (
                    <div className="legend">
                        <span><i style={{ background: '#58d5c9' }} />{view.fixed.title}</span>
                        <span><i style={{ background: '#f59e6b' }} />{view.moving.title}</span>
                        <span><i style={{ background: '#ff3df2' }} />{t('変異')}</span>
                    </div>
                )}
                {fullscreen && <div className="fullscreen-hint small"><Kbd combo="f" /> {t('または')} <Kbd combo="escape" /> {t('で戻る')}</div>}
            </div>
            {shot && (
                <Modal title={t('スクリーンショット')} onClose={() => setShot(null)} wide>
                    <img className="shot" src={shot} alt={t('3D ビューアの画像')} />
                    <div className="row wrap">
                        <Button variant="primary" onClick={async () => {
                            try {
                                const r = await api.saveDataUrl(shotName, shot);
                                toast('success', `${t('保存しました:')} ${r.path}`);
                            } catch (e) {
                                toast('error', `${t('保存できませんでした:')} ${errorMessage(e)}`);
                            }
                        }}><Icon name="download" size={14} /> {t('ダウンロードフォルダに保存')}</Button>
                        <Button onClick={async () => {
                            try {
                                const blob = await (await fetch(shot)).blob();
                                await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
                                toast('success', t('クリップボードにコピーしました'));
                            } catch (e) {
                                toast('error', `${t('コピーできませんでした:')} ${errorMessage(e)}`);
                            }
                        }}>{t('クリップボードにコピー')}</Button>
                    </div>
                </Modal>
            )}
        </div>
    );
}
