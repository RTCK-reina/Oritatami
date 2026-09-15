/** App-wide UI commands (command palette, shortcuts) without threading callbacks through every panel. */
export type AddTab = 'uniprot' | 'pdb' | 'paste' | 'ligand' | 'file' | 'library' | 'history';

export interface UiEvents {
    openAdd: AddTab;
    openSettings: void;
    openHelp: void;
    openPalette: void;
    toggleViewerFullscreen: void;
    viewerAction: 'reset' | 'spin' | 'screenshot' | 'pocket';
    /** cycle the representation / colour dropdowns from the keyboard */
    viewerStyle: 'next' | 'prev';
    viewerColor: 'next' | 'prev';
    /** put the cursor in the LLM composer */
    focusAssistant: void;
    /** the autopilot was switched on or off somewhere: re-read its status */
    autopilotChanged: void;
    /** flip the autopilot from the command palette or a shortcut */
    toggleAutopilot: void;
    /** the same action was triggered again while the first one was still running */
    duplicateAction: string;
    /** open the autopilot dialog (target, plan and the prohibition list) */
    openAutopilot: void;
}

type Handler<T> = (payload: T) => void;
// Typed at the edges (emit/on); stored untyped because a mapped type of Sets cannot be indexed generically.
const handlers = new Map<keyof UiEvents, Set<(payload: unknown) => void>>();

export const uiEvents = {
    emit<K extends keyof UiEvents>(type: K, ...payload: UiEvents[K] extends void ? [] : [UiEvents[K]]): void {
        handlers.get(type)?.forEach(fn => fn(payload[0]));
    },
    on<K extends keyof UiEvents>(type: K, fn: Handler<UiEvents[K]>): () => void {
        const set = handlers.get(type) ?? new Set<(payload: unknown) => void>();
        handlers.set(type, set);
        const wrapped = fn as (payload: unknown) => void;
        set.add(wrapped);
        return () => {
            set.delete(wrapped);
        };
    },
};
