import type { ComponentType, PolymerType, PredictParams, SpecOut, WBComponent, Workbench } from './types';
import { t } from './i18n';

export const AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY';
const ALPHABET: Record<PolymerType, string> = { protein: AMINO_ACIDS, dna: 'ACGTN', rna: 'ACGUN' };

export const AA_INFO: Record<string, { name: string; group: string }> = {
    A: { name: t('アラニン'), group: '疎水性' }, V: { name: t('バリン'), group: '疎水性' }, L: { name: t('ロイシン'), group: '疎水性' },
    I: { name: t('イソロイシン'), group: '疎水性' }, M: { name: t('メチオニン'), group: '疎水性' }, F: { name: t('フェニルアラニン'), group: '芳香族' },
    W: { name: t('トリプトファン'), group: '芳香族' }, Y: { name: t('チロシン'), group: '芳香族' }, S: { name: t('セリン'), group: '極性' },
    T: { name: t('トレオニン'), group: '極性' }, N: { name: t('アスパラギン'), group: '極性' }, Q: { name: t('グルタミン'), group: '極性' },
    C: { name: t('システイン'), group: '特殊' }, G: { name: t('グリシン'), group: '特殊' }, P: { name: t('プロリン'), group: '特殊' },
    D: { name: t('アスパラギン酸'), group: '酸性' }, E: { name: t('グルタミン酸'), group: '酸性' }, K: { name: t('リシン'), group: '塩基性' },
    R: { name: t('アルギニン'), group: '塩基性' }, H: { name: t('ヒスチジン'), group: '塩基性' },
};

export const GROUP_COLOR: Record<string, string> = {
    疎水性: '#d8b35a', 芳香族: '#c98bdb', 極性: '#6cc3a0', 特殊: '#9aa4b2', 酸性: '#ef7f7f', 塩基性: '#6fa8ff',
};

let counter = 0;
export function newUid(): string {
    counter += 1;
    return `c${Date.now().toString(36)}${counter}`;
}

export function defaultParams(): PredictParams {
    return { diffusion_samples: 1, recycling_steps: 4, sampling_steps: 200, use_potentials: false, seed: null };
}

export function emptyWorkbench(): Workbench {
    return { name: t('新しい作業台'), components: [], affinityBinderUid: null, params: defaultParams(), parentJobId: null };
}

export function cleanSequence(raw: string, type: PolymerType): { sequence: string; error: string | null } {
    const sequence = raw.replace(/^>.*$/gm, '').replace(/[\s\d*]/g, '').toUpperCase();
    if (!sequence) return { sequence, error: t('配列が空です') };
    const bad = [...new Set([...sequence].filter(c => !ALPHABET[type].includes(c)))];
    if (bad.length) return { sequence, error: `${t('使えない文字:')} ${bad.join('')}` };
    return { sequence, error: null };
}

const CHAIN_IDS = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'];

export function assignChains(components: WBComponent[]): Map<string, string[]> {
    const out = new Map<string, string[]>();
    let i = 0;
    for (const c of components) {
        const ids: string[] = [];
        for (let k = 0; k < Math.max(1, c.copies); k++) {
            // same scheme as the backend: A..Z, then A0..H9
            const j = i - CHAIN_IDS.length;
            ids.push(i < CHAIN_IDS.length ? CHAIN_IDS[i] : `${'ABCDEFGH'[Math.floor(j / 10)]}${j % 10}`);
            i++;
        }
        out.set(c.uid, ids);
    }
    return out;
}

export interface MutationInfo {
    codes: string[];
    lengthChanged: boolean;
}

export function mutationsOf(c: WBComponent): MutationInfo {
    if (c.type !== 'protein' || !c.sequence || !c.baseSequence) return { codes: [], lengthChanged: false };
    if (c.sequence.length !== c.baseSequence.length) return { codes: [], lengthChanged: true };
    const codes: string[] = [];
    for (let i = 0; i < c.sequence.length; i++) {
        if (c.sequence[i] !== c.baseSequence[i]) codes.push(`${c.baseSequence[i]}${i + 1}${c.sequence[i]}`);
    }
    return { codes, lengthChanged: false };
}

const MUT_RE = /^(?:([A-Za-z0-9]{1,4}):)?([A-Za-z])(\d+)([A-Za-z])$/;

/** Apply substitutions to `sequence`. WT letters are checked against the current sequence. */
export function applyMutationCodes(sequence: string, codes: string[]): string {
    const chars = [...sequence];
    for (const raw of codes) {
        const m = MUT_RE.exec(raw.trim());
        if (!m) throw new Error(`${t('変異の書式が不正です:')} ${raw}`);
        const wt = m[2].toUpperCase();
        const pos = Number(m[3]);
        const mt = m[4].toUpperCase();
        if (!AMINO_ACIDS.includes(wt) || !AMINO_ACIDS.includes(mt)) throw new Error(`${t('標準アミノ酸ではありません:')} ${raw}`);
        if (pos < 1 || pos > chars.length) throw new Error(`${raw}${t(': 位置')} ${pos} ${t('は範囲外です (長さ')} ${chars.length})`);
        if (sequence[pos - 1] !== wt) throw new Error(`${raw}${t(': 位置')} ${pos} ${t('は')} ${sequence[pos - 1]} ${t('です')}`);
        chars[pos - 1] = mt;
    }
    return chars.join('');
}

export function buildSpec(wb: Workbench): SpecOut {
    const chains = assignChains(wb.components);
    const binderChains = wb.affinityBinderUid ? chains.get(wb.affinityBinderUid) : undefined;
    return {
        name: wb.name,
        workbench_name: wb.name,
        components: wb.components.map(c => {
            const base = { type: c.type, chains: chains.get(c.uid) ?? [], label: c.label };
            if (c.type === 'ligand') return { ...base, ...(c.smiles ? { smiles: c.smiles } : { ccd: c.ccd }) };
            const muts = mutationsOf(c);
            return {
                ...base,
                sequence: c.sequence,
                ...(c.type === 'protein' ? { msa: c.msa ?? 'server' } : {}),
                ...(c.cyclic ? { cyclic: true } : {}),
                ...(c.source ? { source: c.source } : {}),
                ...(muts.codes.length ? { mutations: muts.codes, parent_sequence: c.baseSequence } : {}),
            };
        }),
        affinity_binder: binderChains && binderChains.length === 1 ? binderChains[0] : null,
        params: wb.params,
    };
}

export function specToWorkbench(spec: SpecOut, jobId: string | null, title: string): Workbench {
    const components: WBComponent[] = spec.components.map(sc => ({
        uid: newUid(),
        type: sc.type as ComponentType,
        label: sc.label,
        copies: Math.max(1, sc.chains.length),
        sequence: sc.sequence,
        baseSequence: sc.parent_sequence ?? sc.sequence,
        smiles: sc.smiles,
        ccd: sc.ccd,
        msa: sc.msa,
        cyclic: sc.cyclic,
        source: sc.source,
    }));
    let binder: string | null = null;
    if (spec.affinity_binder) {
        const idx = spec.components.findIndex(c => c.chains.includes(spec.affinity_binder as string));
        if (idx >= 0) binder = components[idx].uid;
    }
    return {
        name: spec.workbench_name ?? title,
        components,
        affinityBinderUid: binder,
        params: { ...defaultParams(), ...(spec.params ?? {}) },
        parentJobId: jobId,
    };
}

export function plddtColor(v: number): string {
    if (v >= 90) return '#0053d6';
    if (v >= 70) return '#65cbf3';
    if (v >= 50) return '#ffdb13';
    return '#ff7d45';
}

export function llrColor(v: number): string {
    // diverging: negative red, 0 dark, positive blue-green
    const t = Math.max(-1, Math.min(1, v / 6));
    if (t < 0) {
        const a = -t;
        return `rgb(${Math.round(40 + 200 * a)},${Math.round(44 + 30 * a)},${Math.round(56 + 20 * a)})`;
    }
    return `rgb(${Math.round(40 - 10 * t)},${Math.round(44 + 150 * t)},${Math.round(56 + 140 * t)})`;
}

export function fmt(v: number | null | undefined, digits = 2): string {
    return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';
}

export function formatDuration(sec: number): string {
    if (sec < 60) return `${Math.round(sec)} ${t('秒')}`;
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    return `${m} ${t('分')} ${s} ${t('秒')}`;
}
