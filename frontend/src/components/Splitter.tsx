import { useRef } from 'react';

/**
 * Drag handle between two panes. `value` is the size of the pane the handle controls;
 * `direction` says how pointer movement maps to it (e.g. a handle left of the right column grows it when dragged left).
 */
export function Splitter({ orientation, value, min, max, onChange, onReset, direction = 1, label }: {
    orientation: 'vertical' | 'horizontal';
    value: number;
    min: number;
    max: number;
    onChange: (v: number) => void;
    onReset?: () => void;
    direction?: 1 | -1;
    label: string;
}) {
    const start = useRef<{ pos: number; value: number } | null>(null);
    const clamp = (v: number) => Math.round(Math.min(max, Math.max(min, v)));
    return (
        <div
            className={`splitter splitter-${orientation}`}
            role="separator"
            aria-orientation={orientation}
            aria-label={label}
            aria-valuenow={value}
            aria-valuemin={min}
            aria-valuemax={max}
            tabIndex={0}
            onDoubleClick={onReset}
            onKeyDown={e => {
                const step = e.shiftKey ? 50 : 10;
                const grow = orientation === 'vertical' ? ['ArrowRight', 'ArrowLeft'] : ['ArrowDown', 'ArrowUp'];
                if (e.key === grow[0]) { e.preventDefault(); onChange(clamp(value + step * direction)); }
                if (e.key === grow[1]) { e.preventDefault(); onChange(clamp(value - step * direction)); }
            }}
            onPointerDown={e => {
                e.preventDefault();
                (e.target as HTMLElement).setPointerCapture(e.pointerId);
                start.current = { pos: orientation === 'vertical' ? e.clientX : e.clientY, value };
                document.body.classList.add(orientation === 'vertical' ? 'resizing-col' : 'resizing-row');
            }}
            onPointerMove={e => {
                if (!start.current) return;
                const pos = orientation === 'vertical' ? e.clientX : e.clientY;
                onChange(clamp(start.current.value + (pos - start.current.pos) * direction));
            }}
            onPointerUp={e => {
                start.current = null;
                (e.target as HTMLElement).releasePointerCapture(e.pointerId);
                document.body.classList.remove('resizing-col', 'resizing-row');
            }}
        />
    );
}
