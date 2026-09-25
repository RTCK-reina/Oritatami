import { useMemo, useState } from 'react';
import { t } from '../i18n';
import { useStore } from '../store';
import { assignChains } from '../workbench';
import { Button, Field, Modal, Spinner } from './ui';

const MAX_VARIANTS = 20;
const TOKEN_RE = /^[A-Za-z]\d+[A-Za-z]$/;

/** Queue one prediction per mutation set: paste K48R-style mutations, one per line. */
export function BatchDialog({ onClose }: { onClose: () => void }) {
    const { workbench, runVariants } = useStore();
    const proteins = useMemo(() => workbench.components.filter(c => c.type === 'protein' && c.sequence), [workbench.components]);
    const chains = useMemo(() => assignChains(workbench.components), [workbench.components]);
    const [uid, setUid] = useState(proteins[0]?.uid ?? '');
    const [text, setText] = useState('');
    const [busy, setBusy] = useState(false);

    const comp = proteins.find(c => c.uid === uid);
    const chain = comp ? (chains.get(comp.uid) ?? [])[0] : undefined;
    const sets = useMemo(
        () => text.split('\n').map(l => l.trim()).filter(Boolean).map(l => l.split(/[\s,;/+]+/).filter(Boolean)),
        [text]);
    const bad = sets.flat().find(tok => !TOKEN_RE.test(tok));
    const tooMany = sets.length > MAX_VARIANTS;
    const valid = !busy && !!chain && sets.length > 0 && !tooMany && !bad;

    const submit = async () => {
        setBusy(true);
        await runVariants(sets.map(mutations => ({ chain: chain ?? '', mutations })));
        setBusy(false);
        onClose();
    };

    return (
        <Modal title={t('変異をまとめて予測')} onClose={onClose}>
            <p className="small muted">
                {t('一行につき 1 変異体 (例:')} <code>K48R</code>{t('、複数なら')} <code>K48R L50F</code>{t(')。')}
                {t('それぞれ別の予測ジョブとしてキューに入り、元の作業台の下にまとまります。')}{MAX_VARIANTS} {t('行まで。')}
            </p>
            <Field label={t('対象のタンパク質')}>
                <select value={uid} onChange={e => setUid(e.target.value)}>
                    {proteins.map(c => <option key={c.uid} value={c.uid}>
                        {c.label || t('タンパク質')} {t('(チェーン')} {(chains.get(c.uid) ?? []).join(', ')})
                    </option>)}
                </select>
            </Field>
            <Field label={t('変異 (一行に 1 変異体)')}>
                <textarea rows={8} value={text} onChange={e => setText(e.target.value)}
                    placeholder={'K48R\nL50F\nK48R L50F'} spellCheck={false} />
            </Field>
            {bad && <p className="warn small">{t('変異の書式が不正です:')} {bad} {t('(例: K48R)')}</p>}
            {tooMany && <p className="warn small">{MAX_VARIANTS} {t('行までです (今')} {sets.length} {t('行)')}</p>}
            <div className="row end">
                <span className="small muted">{sets.length > 0 && t(`${sets.length} ${t('件の予測')}`, `${sets.length} predictions`)}</span>
                <Button variant="primary" disabled={!valid} onClick={() => void submit()}>
                    {busy ? <Spinner /> : t('キューに追加')}
                </Button>
            </div>
        </Modal>
    );
}
