import { api } from './api';
import type { WBComponent, Workbench } from './types';
import { defaultParams, newUid } from './workbench';

/** Human ubiquitin (76 aa), identical to UniProt P0CG48 residues 1–76. */
export const UBIQUITIN = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG';
/** Human insulin chains as cleaved from preproinsulin (UniProt P01308: A 90–110, B 25–54). */
const INSULIN_A = 'GIVEQCCTSICSLYQLENYCN';
const INSULIN_B = 'FVNQHLCGSHLVEALYLVCGERGFFYTPKT';

export type ExampleAction =
    | { kind: 'workbench'; build: () => Promise<Workbench>; then?: 'predict' | 'scan' }
    | { kind: 'afdb'; accession: string }
    | { kind: 'pdb'; id: string }
    | { kind: 'assistant'; mode: 'design' | 'mutations' | 'complex'; message: string };

export interface Example {
    id: string;
    title: string;
    summary: string;
    /** what the user learns / sees */
    detail: string;
    time: string;
    tags: string[];
    action: ExampleAction;
}

function comp(c: Omit<WBComponent, 'uid' | 'copies' | 'baseSequence'> & { copies?: number }): WBComponent {
    return { copies: 1, ...c, uid: newUid(), baseSequence: c.sequence };
}

function workbench(name: string, components: WBComponent[], binder: WBComponent | null = null): Workbench {
    return { name, components, affinityBinderUid: binder?.uid ?? null, params: defaultParams(), parentJobId: null };
}

export const EXAMPLES: Example[] = [
    {
        id: 'ubiquitin',
        title: 'はじめての構造予測',
        summary: 'ユビキチン (76 残基)',
        detail: '小さく安定なタンパク質を予測して、リボン表示と信頼度 (pLDDT) の色分けを見ます。',
        time: '約 1 分',
        tags: ['予測', '入門'],
        action: {
            kind: 'workbench',
            then: 'predict',
            build: async () => workbench('ユビキチン', [comp({ type: 'protein', label: 'ユビキチン', sequence: UBIQUITIN, msa: 'server' })]),
        },
    },
    {
        id: 'ubiquitin-scan',
        title: '変異の当たりを探す',
        summary: 'ユビキチン × ESM-2 変異スキャン',
        detail: '全 1 残基置換を ESM-2 で採点し、ヒートマップから変異をクリックで入れて予測し比べます。',
        time: '約 1 分',
        tags: ['変異', 'ESM-2'],
        action: {
            kind: 'workbench',
            then: 'scan',
            build: async () => workbench('ユビキチン 変異', [comp({ type: 'protein', label: 'ユビキチン', sequence: UBIQUITIN, msa: 'server' })]),
        },
    },
    {
        id: 'ca2-drug',
        title: '薬が結合する様子',
        summary: '炭酸脱水酵素 II + 亜鉛 + アセタゾラミド',
        detail: '酵素・金属イオン・薬を一緒に予測し、結合ポケットと親和性 (結合する確率・IC50) を見ます。',
        time: '約 3 分',
        tags: ['複合体', '薬', '親和性'],
        action: {
            kind: 'workbench',
            then: 'predict',
            build: async () => {
                const entry = await api.uniprotEntry('P00918');
                const protein = comp({ type: 'protein', label: '炭酸脱水酵素 II', sequence: entry.sequence, msa: 'server',
                    source: { db: 'UniProt', id: entry.accession } });
                const zinc = comp({ type: 'ligand', label: '亜鉛イオン', ccd: 'ZN' });
                const drug = comp({ type: 'ligand', label: 'アセタゾラミド', ccd: 'AZM' });
                return workbench('CA2 + Zn + アセタゾラミド', [protein, zinc, drug], drug);
            },
        },
    },
    {
        id: 'insulin',
        title: '2 本の鎖を組み合わせる',
        summary: 'インスリン A 鎖 + B 鎖',
        detail: '別々の鎖がどう組み合わさるかを予測し、チェーン間の信頼度 (ipTM) と界面の残基を見ます。',
        time: '約 1 分',
        tags: ['複合体'],
        action: {
            kind: 'workbench',
            then: 'predict',
            build: async () => workbench('インスリン', [
                comp({ type: 'protein', label: 'インスリン A 鎖', sequence: INSULIN_A, msa: 'server' }),
                comp({ type: 'protein', label: 'インスリン B 鎖', sequence: INSULIN_B, msa: 'server' }),
            ]),
        },
    },
    {
        id: 'gfp-afdb',
        title: '計算せずに眺める',
        summary: '緑色蛍光タンパク質 (AlphaFold DB)',
        detail: 'AlphaFold DB に公開済みの予測構造を読み込みます。計算は不要ですぐ表示されます。',
        time: '数秒',
        tags: ['閲覧'],
        action: { kind: 'afdb', accession: 'P42212' },
    },
    {
        id: 'hemoglobin-pdb',
        title: '実験で決まった構造',
        summary: 'ヘモグロビン (PDB 4HHB)',
        detail: 'X 線結晶構造を読み込みます。4 本の鎖とヘムが見え、作業台に取り込んで予測と比べることもできます。',
        time: '数秒',
        tags: ['閲覧', 'PDB'],
        action: { kind: 'pdb', id: '4HHB' },
    },
    {
        id: 'qwen-design',
        title: 'AI に設計させる',
        summary: 'LLM が新しい配列を提案',
        detail: '「4 本のヘリックス束」を LLM に設計させ、ESM-2 の評価を見てから構造を予測します (Ollama が必要)。',
        time: '1〜2 分',
        tags: ['設計', 'LLM'],
        action: { kind: 'assistant', mode: 'design', message: '4 本の α ヘリックスが束になった、60〜80 残基の小さな可溶性タンパク質' },
    },
];
