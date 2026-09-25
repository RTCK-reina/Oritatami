import { useStore } from '../store';
import { GLOSSARY, Kbd, Modal } from './ui';
import { t } from '../i18n';

export const SHORTCUTS: { combo: string; label: string; group: string }[] = [
    { group: t('基本'), combo: 'mod+k', label: t('コマンド検索 (なんでもここから)') },
    { group: t('基本'), combo: 'mod+enter', label: t('作業台の構造を予測する') },
    { group: t('基本'), combo: 'mod+i', label: t('分子を追加する') },
    { group: t('基本'), combo: 'mod+,', label: t('設定') },
    { group: t('基本'), combo: '?', label: t('このヘルプ') },
    { group: t('結果を見る'), combo: 'j', label: t('次の予測結果へ (] も同じ)') },
    { group: t('結果を見る'), combo: 'k', label: t('前の予測結果へ ([ も同じ)') },
    { group: t('結果を見る'), combo: 'mod+1', label: t('左側: 作業台') },
    { group: t('結果を見る'), combo: 'mod+2', label: t('左側: ジョブ') },
    { group: t('結果を見る'), combo: 'mod+3', label: t('左側: 成果 (順位・系統)') },
    { group: t('3D ビューア'), combo: 'f', label: t('大きく表示 / 戻す') },
    { group: t('3D ビューア'), combo: 'r', label: t('視点をリセット') },
    { group: t('3D ビューア'), combo: 'v', label: t('表示のしかたを次へ (⇧V で前へ)') },
    { group: t('3D ビューア'), combo: 'c', label: t('色分けを次へ (⇧C で前へ)') },
    { group: t('3D ビューア'), combo: 's', label: t('自動回転') },
    { group: t('3D ビューア'), combo: 'o', label: t('ポケット表示') },
    { group: t('3D ビューア'), combo: 'p', label: t('画像を撮る') },
    { group: 'LLM', combo: '/', label: t('質問を書く (入力欄にカーソル)') },
    { group: t('基本'), combo: 'mod+shift+a', label: t('自律ループを動かす / 止める') },
    { group: 'LLM', combo: 'mod+enter', label: t('(入力欄で) 送信') },
    { group: 'LLM', combo: 'mod+shift+enter', label: t('(入力欄で) じっくり答えるモデルで送信') },
    { group: 'LLM', combo: 'mod+shift+b', label: t('パネルの表示 / 非表示') },
    { group: 'LLM', combo: 'mod+b', label: t('左パネルの表示 / 非表示') },
];

const TROUBLE: { q: string; a: string }[] = [
    { q: t('予測がなかなか終わらない'), a: t('ジョブの「ログを表示」で進み具合を確認できます。初回は重みのダウンロード (約 6 GB) で時間がかかります。「MSA 検索」で止まって見えるときは公開サーバーの順番待ちです。') },
    { q: t('メモリ不足で失敗した'), a: t('失敗画面の「サンプル 1 で再実行」を試してください。それでも駄目なら構成要素やコピー数を減らすか、長い配列を必要なドメインだけに切り出します。「CPU で再実行」は遅いですが動くことがあります。') },
    { q: t('MSA サーバーのエラーで失敗した'), a: t('時間をおいて再実行するか、「MSA なしで再実行」を選びます。設計した新しい配列はもともと MSA なし (単一配列) が向いています。') },
    { q: t('LLM が応答しない'), a: t('設定 → 準備状況で llama-server とモデルの状態を確認し、「起動する」「ダウンロード」を押してください。予測の実行中は Boltz がメモリを優先するため、LLM の応答が遅くなることがあります。') },
    { q: t('結果を論文やスライドに使いたい'), a: t('結果の「書き出し」から構造 (mmCIF / PDB) とスコアを zip で保存できます。3D の画像はビューア右上のカメラから 4K で保存できます。値はすべて予測で、実験値ではない点に注意してください。') },
];

export function HelpDialog({ onClose }: { onClose: () => void }) {
    const { health } = useStore();
    return (
        <Modal title={t('使い方と用語')} onClose={onClose} wide>
            <div className="help">
                <section>
                    <h3>{t('基本の流れ')}</h3>
                    <p>{t('左の')}<strong>{t('作業台')}</strong>{t('にタンパク質・核酸・リガンド (薬や補因子) を並べ、')}<strong>{t('構造を予測する')}</strong>{t('を押します。計算は Mac の中で行われ、終わると中央の 3D ビューアと結果パネルに表示されます。')}
                        {t('結果から「作業台で改変」を選ぶと、変異を入れて予測し直し、元の結果と重ねて比べられます。右の')} <strong>LLM</strong> {t('は変異・結合相手・新しい配列を提案し、提案は配列との照合やデータベース検索で検証されてから表示されます。')}</p>
                </section>
                <section>
                    <h3>{t('キーボードショートカット')}</h3>
                    <table className="shortcut-table">
                        <tbody>
                            {[...new Set(SHORTCUTS.map(x => x.group))].flatMap(group => [
                                <tr key={group} className="shortcut-group"><td colSpan={2}>{group}</td></tr>,
                                ...SHORTCUTS.filter(x => x.group === group).map(x => (
                                    <tr key={group + x.combo + x.label}><td><Kbd combo={x.combo} /></td><td>{x.label}</td></tr>
                                )),
                            ])}
                        </tbody>
                    </table>
                </section>
                <section>
                    <h3>{t('用語')}</h3>
                    <dl className="glossary">
                        {Object.entries(GLOSSARY).map(([key, g]) => (
                            <div key={key}><dt>{g.title}</dt><dd>{g.body}</dd></div>
                        ))}
                    </dl>
                </section>
                <section>
                    <h3>{t('困ったとき')}</h3>
                    {TROUBLE.map(t => (
                        <details key={t.q} className="faq">
                            <summary>{t.q}</summary>
                            <p>{t.a}</p>
                        </details>
                    ))}
                </section>
                {health && (
                    <section>
                        <h3>{t('データの保存場所')}</h3>
                        <p className="small">{t('ジョブ・履歴:')} <span className="mono">{health.home}</span></p>
                        <p className="small">{t('書き出し:')} <span className="mono">{health.exports_dir}</span></p>
                    </section>
                )}
            </div>
        </Modal>
    );
}
