import { useState } from 'react';
import { EXAMPLES, type Example } from '../examples';
import { useStore } from '../store';
import { SetupStatus } from './SetupStatus';
import { Kbd, Spinner } from './ui';
import { t } from '../i18n';

/** Shown on an empty workbench: one-click examples plus what still needs setting up. */
export function Welcome({ onAdd }: { onAdd: (tab: 'uniprot' | 'ligand' | 'paste' | 'pdb') => void }) {
    const { openExample } = useStore();
    const [busy, setBusy] = useState<string | null>(null);

    const open = async (ex: Example) => {
        setBusy(ex.id);
        try {
            await openExample(ex);
        } finally {
            setBusy(null);
        }
    };

    return (
        <div className="welcome">
            <div className="welcome-head">
                <h2>{t('何を折りたたんでみますか？')}</h2>
                <p className="small muted">{t('例を選ぶと、分子の準備から計算の開始まで自動で進みます。自分の分子は下のボタンから追加できます。')}</p>
            </div>
            <div className="example-grid">
                {EXAMPLES.map(ex => (
                    <button type="button" key={ex.id} className="example-card" disabled={busy !== null} onClick={() => void open(ex)}>
                        <span className="example-title">{ex.title}{busy === ex.id && <Spinner size={11} />}</span>
                        <span className="example-summary">{ex.summary}</span>
                        <span className="example-detail">{ex.detail}</span>
                        <span className="example-meta">
                            {ex.tags.map(tag => <span key={tag} className="example-tag">{t(tag)}</span>)}
                            <span className="muted">{ex.time}</span>
                        </span>
                    </button>
                ))}
            </div>
            <div className="add-row welcome-add">
                <button type="button" className="btn btn-sm btn-default" onClick={() => onAdd('uniprot')}>{t('＋ タンパク質を検索')}</button>
                <button type="button" className="btn btn-sm btn-default" onClick={() => onAdd('paste')}>{t('＋ 配列を貼る')}</button>
                <button type="button" className="btn btn-sm btn-default" onClick={() => onAdd('ligand')}>{t('＋ リガンド・薬')}</button>
                <button type="button" className="btn btn-sm btn-default" onClick={() => onAdd('pdb')}>＋ PDB / AlphaFold DB</button>
            </div>
            <SetupStatus compact />
            <p className="small muted welcome-keys">
                <Kbd combo="mod+k" /> {t('コマンド検索 ·')} <Kbd combo="mod+enter" /> {t('予測を開始 ·')} <Kbd combo="?" /> {t('使い方と用語')}
            </p>
        </div>
    );
}
