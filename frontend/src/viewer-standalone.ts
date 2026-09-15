/**
 * viewer-standalone.ts
 * Standalone Mol* viewer page — used by the SwiftUI native app.
 * Communication via window.postMessage (Swift → viewer) and
 * window.webkit.messageHandlers.viewer.postMessage (viewer → Swift).
 */

import { Viewer, type ColorMode, type LoadItem, type RepStyle } from './viewer/Viewer';

// ── Swift ↔ JS message types ──────────────────────────────────────────────

interface SwiftMessage {
    cmd:
        | 'load'
        | 'clear'
        | 'style'
        | 'color'
        | 'spin'
        | 'pocket'
        | 'highlight'
        | 'focus'
        | 'theme';
    // load — `color` and `format` may be omitted by the native host; see normalize()
    items?: (Partial<LoadItem> & { url: string })[];
    // style/color/spin/pocket/theme
    value?: string | boolean | number;
    // highlight/focus
    chain?: string;
    residues?: number[];
}

interface ViewerEvent {
    event: 'hover' | 'click' | 'ready' | 'error';
    label?: string;
    chain?: string;
    seqId?: number;
    authSeqId?: number;
    compId?: string;
    message?: string;
}

// ── Globals ───────────────────────────────────────────────────────────────

declare global {
    interface Window {
        webkit?: { messageHandlers?: { viewer?: { postMessage(m: ViewerEvent): void } } };
        oritatami?: {
            load(items: LoadItem[]): Promise<void>;
            clear(): Promise<void>;
        };
    }
}

/** The native host sends a bare {url, label, format}; fill in what Viewer requires. */
function normalize(items: (Partial<LoadItem> & { url: string })[]): LoadItem[] {
    return items.map(i => ({
        ...i,
        format: i.format ?? 'mmcif',
        label: i.label ?? 'structure',
        color: i.color ?? 'plddt',
    }));
}

function send(evt: ViewerEvent) {
    window.webkit?.messageHandlers?.viewer?.postMessage(evt);
    // Also dispatch as DOM event so React preview can listen if embedded
    window.dispatchEvent(new CustomEvent('oritatami:viewer', { detail: evt }));
}

// ── Init ──────────────────────────────────────────────────────────────────

async function init() {
    const target = document.getElementById('viewer') as HTMLDivElement;
    const overlay = document.getElementById('overlay') as HTMLDivElement;
    const hoverLabel = document.getElementById('hover-label') as HTMLDivElement;

    const dark = !document.documentElement.dataset.theme?.includes('light');
    const bg = dark ? 0x0d1117 : 0xffffff;

    const v = new Viewer();

    v.onError(msg => {
        send({ event: 'error', message: msg });
    });

    v.onHover(label => {
        if (label) {
            hoverLabel.textContent = label;
            hoverLabel.style.display = 'block';
        } else {
            hoverLabel.style.display = 'none';
        }
        send({ event: 'hover', label: label ?? undefined });
    });

    v.onResidueClick(ref => {
        if (!ref) { send({ event: 'click' }); return; }
        send({
            event: 'click',
            chain: ref.chain,
            seqId: ref.labelSeqId,
            authSeqId: ref.authSeqId,
            compId: ref.compId,
        });
    });

    await v.mount(target, bg);
    overlay.style.display = 'none';
    send({ event: 'ready' });

    // ── Handle messages from Swift / host ────────────────────────────────
    window.addEventListener('message', async (e: MessageEvent<SwiftMessage>) => {
        const msg = e.data;
        if (!msg?.cmd) return;
        switch (msg.cmd) {
            case 'load':
                if (msg.items?.length) {
                    overlay.style.display = 'none';
                    await v.load(normalize(msg.items));
                }
                break;
            case 'clear':
                await v.clear();
                overlay.style.display = 'flex';
                break;
            case 'style':
                if (msg.value) await v.setStyle(msg.value as RepStyle);
                break;
            case 'color':
                if (msg.value) await v.setColor(msg.value as ColorMode);
                break;
            case 'spin':
                await v.setSpin(!!msg.value);
                break;
            case 'pocket':
                await v.setShowPocket(!!msg.value);
                break;
            case 'highlight':
                if (msg.chain && msg.residues) v.highlightResidues(0, msg.chain, msg.residues);
                break;
            case 'focus':
                if (msg.chain && msg.residues) v.focusResidues(0, msg.chain, msg.residues);
                break;
            case 'theme': {
                const light = msg.value === 'light';
                await v.setBackground(light ? 0xffffff : 0x0d1117);
                break;
            }
        }
    });

    // Convenience global for direct JS calls
    window.oritatami = {
        load: (items) => v.load(items),
        clear: () => v.clear(),
    };
}

init().catch(e => {
    console.error('[viewer-standalone]', e);
    send({ event: 'error', message: String(e) });
});
