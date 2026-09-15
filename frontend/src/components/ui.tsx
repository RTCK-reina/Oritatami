import { useCallback, useEffect, useRef, useState, type ButtonHTMLAttributes, type ReactNode, type MouseEvent as ReactMouseEvent } from 'react';
import { comboLabel } from '../hooks';
import { uiEvents } from '../uiEvents';

/** How long a pressed button stays shut after a handler that finishes immediately. Long enough
 *  to swallow the second press of a double-click, short enough not to feel stuck. */
const LOCK_MS = 400;

/**
 * A button that answers to one press.
 *
 * While its handler is running the button is held shut; a press that arrives in the meantime is
 * dropped, the button flinches so the press is visibly *not* lost, and a note is raised so the
 * app can say why. `repeatable` opts out, for the few controls meant to be clicked in a row.
 */
export function Button({ variant = 'default', size = 'md', className = '', onClick, repeatable = false, disabled, ...rest }:
    ButtonHTMLAttributes<HTMLButtonElement> & {
        variant?: 'default' | 'primary' | 'ghost' | 'danger' | 'qwen';
        size?: 'sm' | 'md';
        /** allow rapid repeated presses (steppers, toggles) */
        repeatable?: boolean;
    }) {
    const [busy, setBusy] = useState(false);
    const [flinch, setFlinch] = useState(false);
    const locked = useRef(false);
    const timer = useRef(0);
    useEffect(() => () => window.clearTimeout(timer.current), []);

    const press = useCallback(async (e: ReactMouseEvent<HTMLButtonElement>) => {
        if (!onClick) return;
        if (!repeatable && locked.current) {
            e.preventDefault();
            e.stopPropagation();
            setFlinch(true);
            window.setTimeout(() => setFlinch(false), 600);
            uiEvents.emit('duplicateAction', (e.currentTarget.innerText || '').trim().split('\n')[0]);
            return;
        }
        if (repeatable) {
            onClick(e);
            return;
        }
        locked.current = true;
        setBusy(true);
        try {
            await onClick(e);
        } finally {
            timer.current = window.setTimeout(() => {
                locked.current = false;
                setBusy(false);
            }, LOCK_MS);
        }
    }, [onClick, repeatable]);

    return (
        <button type="button" disabled={disabled}
            aria-busy={busy || undefined}
            className={`btn btn-${variant} btn-${size} ${busy ? 'is-busy' : ''} ${flinch ? 'is-double' : ''} ${className}`}
            onClick={onClick ? press : undefined} {...rest} />
    );
}

export function Spinner({ size = 14 }: { size?: number }) {
    return <span className="spinner" style={{ width: size, height: size }} role="status" aria-label="処理中" />;
}

export function Modal({ title, onClose, children, wide = false, className = '' }:
    { title: string; onClose: () => void; children: ReactNode; wide?: boolean; className?: string }) {
    const ref = useRef<HTMLDivElement>(null);
    const closeRef = useRef(onClose);
    closeRef.current = onClose;
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.key === 'Escape' && !e.isComposing) closeRef.current();
        };
        window.addEventListener('keydown', onKey);
        const previous = document.activeElement as HTMLElement | null;
        // move focus into the dialog unless a child already took it (autoFocus)
        window.requestAnimationFrame(() => {
            if (ref.current && !ref.current.contains(document.activeElement)) ref.current.focus();
        });
        return () => {
            window.removeEventListener('keydown', onKey);
            previous?.focus?.();
        };
    }, []);
    return (
        <div className="modal-backdrop" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
            <div ref={ref} tabIndex={-1} className={`modal ${wide ? 'modal-wide' : ''} ${className}`} role="dialog" aria-modal="true" aria-label={title}>
                <div className="modal-head">
                    <h2>{title}</h2>
                    <button type="button" className="icon-btn" onClick={onClose} aria-label="閉じる"><Icon name="close" /></button>
                </div>
                <div className="modal-body">{children}</div>
            </div>
        </div>
    );
}

export function Tabs<T extends string>({ tabs, value, onChange, small = false }:
    { tabs: { id: T; label: ReactNode; badge?: ReactNode; title?: string }[]; value: T; onChange: (t: T) => void; small?: boolean }) {
    return (
        <div className={`tabs ${small ? 'tabs-sm' : ''}`} role="tablist">
            {tabs.map(t => (
                <button type="button" key={t.id} role="tab" aria-selected={t.id === value} className={`tab ${t.id === value ? 'active' : ''}`}
                    title={t.title} onClick={() => onChange(t.id)}>
                    {t.label}
                    {t.badge !== undefined && t.badge !== null && <span className="tab-badge">{t.badge}</span>}
                </button>
            ))}
        </div>
    );
}

export function Field({ label, hint, children }: { label: ReactNode; hint?: ReactNode; children: ReactNode }) {
    return (
        <label className="field">
            <span className="field-label">{label}</span>
            {children}
            {hint && <span className="field-hint">{hint}</span>}
        </label>
    );
}

export function Empty({ children }: { children: ReactNode }) {
    return <div className="empty">{children}</div>;
}

export function Metric({ label, value, tone, hint, term }: { label: string; value: ReactNode; tone?: 'good' | 'ok' | 'bad'; hint?: string; term?: string }) {
    return (
        <div className={`metric ${tone ? `metric-${tone}` : ''}`} title={hint}>
            <div className="metric-value">{value}</div>
            <div className="metric-label">{label}{term && <InfoTip term={term} />}</div>
        </div>
    );
}

export function StatusDot({ status }: { status: string }) {
    const label: Record<string, string> = { queued: '待機中', running: '実行中', succeeded: '完了', failed: '失敗', cancelled: 'キャンセル' };
    return <span className={`status-dot status-${status}`} role="img" aria-label={label[status] ?? status} />;
}

export function Kbd({ combo }: { combo: string }) {
    return <kbd className="kbd">{comboLabel(combo)}</kbd>;
}

export const TYPE_LABEL: Record<string, string> = { protein: 'タンパク質', dna: 'DNA', rna: 'RNA', ligand: 'リガンド' };

/** Short explanations of the numbers the app shows. Used by InfoTip and the help dialog. */
export const GLOSSARY: Record<string, { title: string; body: string }> = {
    plddt: { title: 'pLDDT', body: '残基ごとの「この部分の形にどれだけ自信があるか」(0–100)。90 以上はとても高く、50 未満はその部分がほどけているか予測できていない可能性が高い。' },
    ptm: { title: 'pTM', body: '全体の折りたたみ方 (トポロジー) の信頼度 (0–1)。0.5 以上なら全体の形はおおむね正しい見込み。' },
    iptm: { title: 'ipTM', body: 'チェーン同士の相対的な位置関係の信頼度 (0–1)。0.8 以上なら結合の仕方に自信あり、0.6 未満は当てにならない。' },
    pae: { title: 'PAE (予測位置誤差)', body: 'ある残基を基準に重ねたとき、別の残基の位置が何 Å ずれると予測されるか。暗い (小さい) ほど相対配置に自信がある。ドメインやチェーンのブロック構造が読み取れる。' },
    confidence: { title: '総合信頼度', body: 'Boltz がサンプルを並べる基準。おおよそ 0.8 × 平均 pLDDT + 0.2 × ipTM (単鎖では pTM)。' },
    affinity: { title: '結合親和性', body: '「結合する確率」は結合する分子かどうかの判別、「IC50」は結合する分子の中での強さの目安。どちらも予測値で、実験値の代わりにはならない。' },
    llr: { title: 'LLR (ESM-2)', body: '言語モデル ESM-2 が、元のアミノ酸と比べてその置換をどれだけ「自然」と見なすかの対数尤度比。プラスほど自然。安定化や機能向上を保証するものではない。' },
    pppl: { title: '疑似パープレキシティ', body: 'ESM-2 から見た配列全体の不自然さ。低いほど天然のタンパク質らしい配列。設計配列の良し悪しの目安に使う。' },
    msa: { title: 'MSA', body: '似た配列を集めた多重配列アラインメント。進化の情報で精度が上がる。ColabFold の公開サーバーに配列が送られる。設計した新しい配列では「単一配列」が向く。' },
    rmsd: { title: 'RMSD', body: '2 つの構造を重ねたときの Cα 原子の平均的なずれ (Å)。1 Å 未満はほぼ同じ形、3 Å を超えると形がはっきり違う。' },
    tokens: { title: 'トークン', body: 'Boltz が扱う単位。タンパク質・核酸は 1 残基 1 トークン、リガンドは重原子 1 個 1 トークン。多いほど時間とメモリを使う。' },
};

export function InfoTip({ term }: { term: string }) {
    const entry = GLOSSARY[term];
    const [open, setOpen] = useState(false);
    if (!entry) return null;
    return (
        <span className="infotip" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
            <button type="button" className="infotip-btn" aria-label={`${entry.title} の説明`} onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}
                onClick={e => { e.stopPropagation(); setOpen(o => !o); }}>?</button>
            {open && <span className="infotip-pop" role="tooltip"><strong>{entry.title}</strong>{entry.body}</span>}
        </span>
    );
}

/** A small dropdown menu anchored to a button. */
export function Menu({ label, items, variant = 'ghost', size = 'sm', title }: {
    label: ReactNode; title?: string; variant?: 'default' | 'primary' | 'ghost'; size?: 'sm' | 'md';
    items: ({ label: ReactNode; onSelect: () => void; disabled?: boolean; hint?: string } | 'separator')[];
}) {
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (!open) return;
        const close = (e: MouseEvent) => {
            if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
        };
        const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
        window.addEventListener('mousedown', close);
        window.addEventListener('keydown', onKey);
        return () => {
            window.removeEventListener('mousedown', close);
            window.removeEventListener('keydown', onKey);
        };
    }, [open]);
    return (
        <div className="menu" ref={ref}>
            {/* opening and closing a menu in quick succession is normal, so this one is not held shut */}
            <Button size={size} variant={variant} title={title} repeatable aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(o => !o)}>
                {label} <Icon name="chevron-down" size={12} />
            </Button>
            {open && (
                <div className="menu-pop" role="menu">
                    {items.map((it, i) => it === 'separator' ? <div key={i} className="menu-sep" /> : (
                        <button type="button" key={i} role="menuitem" className="menu-item" disabled={it.disabled} title={it.hint}
                            onClick={() => { setOpen(false); it.onSelect(); }}>{it.label}</button>
                    ))}
                </div>
            )}
        </div>
    );
}

const ICONS: Record<string, ReactNode> = {
    close: <path d="M6 6l12 12M18 6L6 18" />,
    'chevron-down': <path d="M6 9l6 6 6-6" />,
    'chevron-up': <path d="M6 15l6-6 6 6" />,
    'to-top': <><path d="M5 4h14" /><path d="M12 20V8" /><path d="M7 13l5-5 5 5" /></>,
    'to-bottom': <><path d="M5 20h14" /><path d="M12 4v12" /><path d="M7 11l5 5 5-5" /></>,
    'chevron-left': <path d="M15 6l-6 6 6 6" />,
    'chevron-right': <path d="M9 6l6 6-6 6" />,
    settings: <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 00.3 1.8l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.8-.3 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1.1-1.5 1.7 1.7 0 00-1.8.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00.3-1.8 1.7 1.7 0 00-1.5-1H3a2 2 0 110-4h.1a1.7 1.7 0 001.5-1.1 1.7 1.7 0 00-.3-1.8l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 001.8.3H9a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.8-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.8V9a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z" /></>,
    help: <><circle cx="12" cy="12" r="9" /><path d="M9.5 9a2.5 2.5 0 015 .5c0 1.7-2.5 2-2.5 3.5M12 17h.01" /></>,
    search: <><circle cx="11" cy="11" r="7" /><path d="M20 20l-3.5-3.5" /></>,
    expand: <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />,
    shrink: <path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" />,
    camera: <><path d="M4 8h3l2-3h6l2 3h3v11H4z" /><circle cx="12" cy="13" r="3.5" /></>,
    rotate: <><path d="M20 12a8 8 0 11-2.3-5.7" /><path d="M20 4v5h-5" /></>,
    target: <><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="2" /></>,
    download: <><path d="M12 4v11M7 10l5 5 5-5" /><path d="M5 20h14" /></>,
    folder: <path d="M3 7h6l2 2h10v10H3z" />,
    play: <path d="M8 5l11 7-11 7z" />,
    retry: <><path d="M4 12a8 8 0 0113.7-5.7L20 9" /><path d="M20 4v5h-5" /><path d="M20 12a8 8 0 01-13.7 5.7L4 15" /><path d="M4 20v-5h5" /></>,
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
    moon: <path d="M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z" />,
    'panel-left': <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M9 4v16" /></>,
    'panel-right': <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M15 4v16" /></>,
    sparkles: <path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8zM19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9z" />,
    command: <path d="M9 6a3 3 0 10-3 3h12a3 3 0 10-3-3v12a3 3 0 103-3H6a3 3 0 103 3z" />,
    trash: <><path d="M4 7h16M10 11v6M14 11v6" /><path d="M6 7l1 13h10l1-13M9 7V4h6v3" /></>,
    star: <path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1-4.4-4.3 6.1-.9z" />,
    layers: <><path d="M12 3l9 5-9 5-9-5z" /><path d="M3 13l9 5 9-5" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7.5h.01" /></>,
    check: <path d="M5 12.5l4.5 4.5L19 7" />,
    warning: <><path d="M12 3l10 18H2z" /><path d="M12 10v5M12 18h.01" /></>,
    flask: <><path d="M9 3h6M10 3v6L4.5 19a1.5 1.5 0 001.3 2h12.4a1.5 1.5 0 001.3-2L14 9V3" /><path d="M7 15h10" /></>,
};

export function Icon({ name, size = 16, className = '' }: { name: keyof typeof ICONS | string; size?: number; className?: string }) {
    return (
        <svg className={`icon ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
            strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
            {ICONS[name] ?? ICONS.info}
        </svg>
    );
}
