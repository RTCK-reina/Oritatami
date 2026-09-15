/** Lets panels talk to the single Mol* viewer without prop drilling. */
type Highlight = (chain: string, labelSeqIds: number[]) => void;

export const viewerBus: { highlight: Highlight; focus: Highlight } = {
    highlight: () => undefined,
    focus: () => undefined,
};
