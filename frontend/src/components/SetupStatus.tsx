import { useEffect, useState } from 'react';
import { api, errorMessage } from '../api';
import { useStore } from '../store';
import { Button, Icon, Spinner } from './ui';
import { t } from '../i18n';

type Level = 'ok' | 'warn' | 'bad' | 'pending';

/** llama-server can be the copy inside this app, one it fetched, or the machine's own. */
const SOURCE_LABEL: Record<string, string> = {
    bundled: t(' (アプリ内蔵)'),
    downloaded: t(' (アプリが取得したもの)'),
    system: t(' (この Mac にインストール済みのもの)'),
};

const pct = (done?: number, total?: number) => (total ? Math.round(((done ?? 0) / total) * 100) : 0);

interface Item {
    key: string;
    label: string;
    level: Level;
    detail: string;
    action?: { label: string; run: () => Promise<void> };
}

/** What works right now and what to do about what doesn't. */
export function SetupStatus({ compact = false }: { compact?: boolean }) {
    const { health, refreshHealth, toast } = useStore();
    const [busy, setBusy] = useState<string | null>(null);
    const pull = health?.llm.pull;
    const install = health?.llm.install;

    // while a download runs — the model, or the runtime itself — refresh often enough to show progress
    useEffect(() => {
        if (!pull?.active && !install?.active) return;
        const t = window.setInterval(() => void refreshHealth(), 2000);
        return () => window.clearInterval(t);
    }, [pull?.active, install?.active, refreshHealth]);

    if (!health) return <div className="setup"><Spinner /> {t('状態を確認中…')}</div>;

    const run = (key: string, fn: () => Promise<unknown>) => async () => {
        setBusy(key);
        try {
            await fn();
            await refreshHealth();
        } catch (e) {
            toast('error', errorMessage(e));
        } finally {
            setBusy(null);
        }
    };

    const items: Item[] = [
        {
            key: 'boltz',
            label: t('構造予測 (Boltz-2)'),
            level: health.boltz.bin ? 'ok' : 'bad',
            detail: health.boltz.bin
                ? `${t('インストール済み')}${health.boltz.version ? ` (v${health.boltz.version})` : ''}`
                : t('未インストール。ターミナルでフォルダを開き ./scripts/setup.sh を実行してください'),
        },
        {
            key: 'weights',
            label: t('Boltz の重み'),
            level: health.boltz.weights && health.boltz.ccd ? 'ok' : 'warn',
            detail: health.boltz.weights && health.boltz.ccd
                ? `${t('ダウンロード済み')}${health.boltz.affinity_weights ? t(' (親和性モデルを含む)') : ''}`
                : t('初回の予測時に自動でダウンロードします (約 6 GB、数分)'),
        },
        {
            key: 'gpu',
            label: 'GPU (Apple Silicon)',
            level: !health.torch_probed ? 'pending' : health.mps ? 'ok' : 'warn',
            detail: !health.torch_probed ? t('確認中…') : health.mps ? `${t('MPS で計算します (PyTorch')} ${health.torch})`
                : health.torch ? t('MPS が使えないため CPU で計算します (遅くなります)') : t('PyTorch を読み込めません'),
        },
        {
            key: 'esm',
            label: t('変異スコア (ESM-2)'),
            level: health.esm.cached || health.esm.loaded ? 'ok' : 'warn',
            detail: health.esm.loaded ? `${t('メモリに読み込み済み (')}${health.esm.device})`
                : health.esm.cached ? t('ダウンロード済み') : t('初回の変異スキャン時に自動でダウンロードします (約 2.5 GB)'),
        },
        {
            key: 'ollama',
            label: t('AI アシスタント (llama.cpp)'),
            level: health.llm.server ? 'ok' : install?.active ? 'pending' : 'bad',
            detail: health.llm.server
                ? `${t('起動しています')}${SOURCE_LABEL[health.llm.source ?? ''] ?? ''}`
                : install?.active
                    ? `${install.status ?? t('取得中')} ${pct(install.completed, install.total)}%`
                    : install?.error
                        ? `${t('取得に失敗しました:')} ${install.error}`
                        : health.llm.binary
                            ? `${t('見つかっています')}${SOURCE_LABEL[health.llm.source ?? ''] ?? ''}${t('が、起動していません')}`
                            : `${t('この Mac にありません。押すとアプリ用に取得します (約')} ${health.llm.download_mb ?? 150} ${t('MB、システムには入れません)')}`,
            action: health.llm.server || install?.active ? undefined : {
                label: health.llm.binary ? t('起動する') : t('用意する'),
                run: run('ollama', () => api.llmStart().then(r => {
                    if (!r.running && !r.installing) throw new Error(r.error ?? t('llama-server を起動できませんでした'));
                })),
            },
        },
        {
            key: 'model',
            label: `${t('LLM モデル (')}${health.llm.model})`,
            // Availability is read from the GGUF files themselves, so it is known before
            // the server is up — and the download works without a server too.
            level: health.llm.model_available ? 'ok' : pull?.active ? 'pending' : 'warn',
            detail: health.llm.model_available ? t('ダウンロード済み')
                : pull?.active ? `${t('ダウンロード中')} ${pull.total ? Math.round(((pull.completed ?? 0) / pull.total) * 100) : 0}%`
                    : pull?.error ? `${t('ダウンロード失敗:')} ${pull.error}` : t('まだダウンロードされていません (数 GB)'),
            action: !health.llm.model_available && !pull?.active
                ? { label: t('ダウンロード'), run: run('model', () => api.llmPull(health.llm.model)) } : undefined,
        },
        {
            key: 'disk',
            label: t('空きディスク'),
            level: health.disk_free_gb >= 20 ? 'ok' : health.disk_free_gb >= 8 ? 'warn' : 'bad',
            detail: `${health.disk_free_gb} GB${health.disk_free_gb < 20 ? t(' (モデルの保存に 10 GB 以上あると安心です)') : ''}`,
        },
    ];

    const problems = items.filter(i => i.level === 'bad' || i.level === 'warn');
    if (compact && problems.length === 0) {
        return (
            <div className="setup setup-compact setup-allok">
                <Icon name="check" /> {t('準備完了: 予測・変異スコア・AI アシスタントがすべて使えます')}
                {health.machine.memory_gb && <span className="muted small"> · {health.machine.chip} / {health.machine.memory_gb} GB</span>}
            </div>
        );
    }
    const shown = compact ? problems : items;
    return (
        <div className={`setup ${compact ? 'setup-compact' : ''}`}>
            {compact && <div className="setup-title"><Icon name="info" /> {t('準備状況')}</div>}
            {shown.map(item => (
                <div key={item.key} className={`setup-item setup-${item.level}`}>
                    <span className="setup-mark" aria-hidden>{item.level === 'ok' ? <Icon name="check" size={14} />
                        : item.level === 'pending' ? <Spinner size={11} /> : <Icon name="warning" size={14} />}</span>
                    <span className="setup-label">{item.label}</span>
                    <span className="setup-detail">{item.detail}</span>
                    {item.action && (
                        <Button size="sm" disabled={busy !== null} onClick={() => void item.action?.run()}>
                            {busy === item.key ? <Spinner size={11} /> : item.action.label}
                        </Button>
                    )}
                    {item.key === 'ollama' && install?.active && (
                        <div className="progress setup-progress"><i style={{ width: `${pct(install.completed, install.total)}%` }} /></div>
                    )}
                </div>
            ))}
            {!compact && (
                <div className="small muted">
                    {health.machine.os} · {health.machine.chip}{health.machine.memory_gb ? ` ${t('· メモリ')} ${health.machine.memory_gb} GB` : ''} · Python {health.machine.python} · Oritatami {health.version}
                </div>
            )}
        </div>
    );
}
