import { useEffect, useMemo, useRef, useState } from 'react';
import { comboLabel } from '../hooks';

export interface Command {
    id: string;
    label: string;
    section: string;
    run: () => void;
    shortcut?: string;
    /** extra words to match (English names, abbreviations) */
    keywords?: string;
    disabled?: boolean;
}

function normalize(text: string): string {
    // full-width → half-width and katakana → hiragana so either spelling matches
    return text.normalize('NFKC').toLowerCase().replace(/[ァ-ヶ]/g, ch => String.fromCharCode(ch.charCodeAt(0) - 0x60));
}

export function CommandPalette({ commands, onClose }: { commands: Command[]; onClose: () => void }) {
    const [query, setQuery] = useState('');
    const [index, setIndex] = useState(0);
    const listRef = useRef<HTMLDivElement>(null);

    const results = useMemo(() => {
        const tokens = normalize(query).split(/\s+/).filter(Boolean);
        const enabled = commands.filter(c => !c.disabled);
        if (!tokens.length) return enabled;
        return enabled.filter(c => {
            const hay = normalize(`${c.label} ${c.section} ${c.keywords ?? ''}`);
            return tokens.every(t => hay.includes(t));
        });
    }, [commands, query]);

    useEffect(() => setIndex(0), [query]);
    useEffect(() => {
        listRef.current?.querySelector<HTMLElement>(`[data-index="${index}"]`)?.scrollIntoView({ block: 'nearest' });
    }, [index]);

    const run = (c: Command | undefined) => {
        if (!c) return;
        onClose();
        // let the palette unmount before the command opens its own dialog or moves focus
        window.setTimeout(c.run, 0);
    };

    let lastSection = '';
    return (
        <div className="modal-backdrop palette-backdrop" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
            <div className="palette" role="dialog" aria-modal="true" aria-label="コマンド">
                <input autoFocus className="palette-input" value={query} placeholder="やりたいことを入力 (例: 予測, 変異, 画像, テーマ)"
                    aria-label="コマンドを検索" aria-controls="palette-list" aria-activedescendant={results[index] ? `cmd-${results[index].id}` : undefined}
                    onChange={e => setQuery(e.target.value)}
                    onKeyDown={e => {
                        if (e.nativeEvent.isComposing) return;
                        if (e.key === 'ArrowDown') { e.preventDefault(); setIndex(i => Math.min(results.length - 1, i + 1)); }
                        if (e.key === 'ArrowUp') { e.preventDefault(); setIndex(i => Math.max(0, i - 1)); }
                        if (e.key === 'Enter') { e.preventDefault(); run(results[index]); }
                        if (e.key === 'Escape') { e.preventDefault(); onClose(); }
                    }} />
                <div className="palette-list" id="palette-list" role="listbox" ref={listRef}>
                    {results.length === 0 && <div className="palette-empty">該当するコマンドがありません</div>}
                    {results.map((c, i) => {
                        const header = c.section !== lastSection ? c.section : null;
                        lastSection = c.section;
                        return (
                            <div key={c.id}>
                                {header && <div className="palette-section">{header}</div>}
                                <div id={`cmd-${c.id}`} data-index={i} role="option" aria-selected={i === index}
                                    className={`palette-item ${i === index ? 'active' : ''}`}
                                    onMouseMove={() => setIndex(i)} onClick={() => run(c)}>
                                    <span className="grow ellipsis">{c.label}</span>
                                    {c.shortcut && <kbd className="kbd">{comboLabel(c.shortcut)}</kbd>}
                                </div>
                            </div>
                        );
                    })}
                </div>
                <div className="palette-foot small muted">↑↓ で選択 · ↩ で実行 · Esc で閉じる</div>
            </div>
        </div>
    );
}
