import { useCallback, useEffect, useRef, useState } from 'react';

/** useState that survives restarts (localStorage). Falls back to memory when storage is unavailable. */
export function usePersistentState<T>(key: string, initial: T): [T, (v: T | ((prev: T) => T)) => void] {
    const [value, setValue] = useState<T>(() => {
        try {
            const raw = localStorage.getItem(key);
            return raw === null ? initial : (JSON.parse(raw) as T);
        } catch {
            return initial;
        }
    });
    const set = useCallback((v: T | ((prev: T) => T)) => {
        setValue(prev => {
            const next = typeof v === 'function' ? (v as (p: T) => T)(prev) : v;
            try {
                localStorage.setItem(key, JSON.stringify(next));
            } catch {
                // storage full or disabled: keep the in-memory value
            }
            return next;
        });
    }, [key]);
    return [value, set];
}

export type ThemePref = 'system' | 'dark' | 'light';

function systemTheme(): 'dark' | 'light' {
    return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

/** Applies the theme to <html data-theme> and returns the resolved theme. */
export function useTheme(pref: ThemePref): 'dark' | 'light' {
    const [resolved, setResolved] = useState<'dark' | 'light'>(() => (pref === 'system' ? systemTheme() : pref));
    useEffect(() => {
        const apply = () => setResolved(pref === 'system' ? systemTheme() : pref);
        apply();
        if (pref !== 'system' || !window.matchMedia) return;
        const mq = window.matchMedia('(prefers-color-scheme: light)');
        mq.addEventListener('change', apply);
        return () => mq.removeEventListener('change', apply);
    }, [pref]);
    useEffect(() => {
        document.documentElement.dataset.theme = resolved;
    }, [resolved]);
    return resolved;
}

export interface Hotkey {
    /** e.g. "mod+k", "mod+enter", "?", "f", "escape" — "mod" is ⌘ on macOS and Ctrl elsewhere */
    combo: string;
    handler: (e: KeyboardEvent) => void;
    /** also fire while typing in an input/textarea (for mod+ shortcuts) */
    allowInInputs?: boolean;
}

const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

export function comboLabel(combo: string): string {
    return combo.split('+').map(part => {
        switch (part) {
            case 'mod': return isMac ? '⌘' : 'Ctrl';
            case 'shift': return '⇧';
            case 'alt': return isMac ? '⌥' : 'Alt';
            case 'enter': return '↩';
            case 'escape': return 'Esc';
            default: return part.length === 1 ? part.toUpperCase() : part;
        }
    }).join(isMac ? '' : '+');
}

function matches(e: KeyboardEvent, combo: string): boolean {
    const parts = combo.toLowerCase().split('+');
    const key = parts[parts.length - 1];
    const mod = parts.includes('mod');
    const shift = parts.includes('shift');
    const alt = parts.includes('alt');
    const modPressed = isMac ? e.metaKey : e.ctrlKey;
    if (mod !== modPressed || alt !== e.altKey) return false;
    const pressed = e.key.toLowerCase();
    // "?" arrives as shift+/ on most layouts: compare the produced character and ignore shift for symbols.
    if (key.length === 1 && !/[a-z0-9]/.test(key)) return pressed === key;
    if (shift !== e.shiftKey) return false;
    return pressed === key || (key === 'enter' && pressed === 'enter');
}

function typingTarget(target: EventTarget | null): boolean {
    const el = target as HTMLElement | null;
    if (!el) return false;
    const tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

export function useHotkeys(hotkeys: Hotkey[]): void {
    const ref = useRef(hotkeys);
    ref.current = hotkeys;
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.isComposing) return; // IME conversion in progress (Japanese input)
            for (const hk of ref.current) {
                if (!matches(e, hk.combo)) continue;
                if (typingTarget(e.target) && !hk.allowInInputs) continue;
                e.preventDefault();
                hk.handler(e);
                return;
            }
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, []);
}

/** Debounced value: updates `delay` ms after the input stops changing. */
export function useDebounced<T>(value: T, delay: number): T {
    const [v, setV] = useState(value);
    useEffect(() => {
        const t = window.setTimeout(() => setV(value), delay);
        return () => window.clearTimeout(t);
    }, [value, delay]);
    return v;
}

/** The theme currently applied to <html data-theme>, updated when it changes (for canvas drawings). */
export function useAppliedTheme(): string {
    const [theme, setTheme] = useState(() => document.documentElement.dataset.theme ?? 'dark');
    useEffect(() => {
        const mo = new MutationObserver(() => setTheme(document.documentElement.dataset.theme ?? 'dark'));
        mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
        return () => mo.disconnect();
    }, []);
    return theme;
}

/** Window width, sampled on resize. Used to fold side panels away before the centre collapses. */
export function useWindowWidth(): number {
    const [w, setW] = useState(() => window.innerWidth);
    useEffect(() => {
        let frame = 0;
        const onResize = () => {
            window.cancelAnimationFrame(frame);
            frame = window.requestAnimationFrame(() => setW(window.innerWidth));
        };
        window.addEventListener('resize', onResize);
        return () => {
            window.cancelAnimationFrame(frame);
            window.removeEventListener('resize', onResize);
        };
    }, []);
    return w;
}
