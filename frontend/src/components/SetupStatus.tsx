import { useEffect, useState } from 'react';
import { api, errorMessage } from '../api';
import { useStore } from '../store';
import { Button, Icon, Spinner } from './ui';

type Level = 'ok' | 'warn' | 'bad' | 'pending';

/** Ollama can be the copy inside this app, one it fetched, or the machine's own. */
const SOURCE_LABEL: Record<string, string> = {
    bundled: ' (アプリ内蔵)',
    downloaded: ' (アプリが取得したもの)',
    system: ' (この Mac にインストール済みのもの)',
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

    // while a download runs — the model, or Ollama itself — refresh often enough to show progress
    useEffect(() => {
        if (!pull?.active && !install?.active) return;
        const t = window.setInterval(() => void refreshHealth(), 2000);
        return () => window.clearInterval(t);
    }, [pull?.active, install?.active, refreshHealth]);

    if (!health) return <div className="setup"><Spinner /> 状態を確認中…</div>;

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
            label: '構造予測 (Boltz-2)',
            level: health.boltz.bin ? 'ok' : 'bad',
            detail: health.boltz.bin
                ? `インストール済み${health.boltz.version ? ` (v${health.boltz.version})` : ''}`
                : '未インストール。ターミナルでフォルダを開き ./scripts/setup.sh を実行してください',
        },
        {
            key: 'weights',
            label: 'Boltz の重み',
            level: health.boltz.weights && health.boltz.ccd ? 'ok' : 'warn',
            detail: health.boltz.weights && health.boltz.ccd
                ? `ダウンロード済み${health.boltz.affinity_weights ? ' (親和性モデルを含む)' : ''}`
                : '初回の予測時に自動でダウンロードします (約 6 GB、数分)',
        },
        {
            key: 'gpu',
            label: 'GPU (Apple Silicon)',
            level: !health.torch_probed ? 'pending' : health.mps ? 'ok' : 'warn',
            detail: !health.torch_probed ? '確認中…' : health.mps ? `MPS で計算します (PyTorch ${health.torch})`
                : health.torch ? 'MPS が使えないため CPU で計算します (遅くなります)' : 'PyTorch を読み込めません',
        },
        {
            key: 'esm',
            label: '変異スコア (ESM-2)',
            level: health.esm.cached || health.esm.loaded ? 'ok' : 'warn',
            detail: health.esm.loaded ? `メモリに読み込み済み (${health.esm.device})`
                : health.esm.cached ? 'ダウンロード済み' : '初回の変異スキャン時に自動でダウンロードします (約 2.5 GB)',
        },
        {
            key: 'ollama',
            label: 'AI アシスタント (Ollama)',
            level: health.llm.server ? 'ok' : install?.active ? 'pending' : 'bad',
            detail: health.llm.server
                ? `起動しています${SOURCE_LABEL[health.llm.source ?? ''] ?? ''}`
                : install?.active
                    ? `${install.status ?? '取得中'} ${pct(install.completed, install.total)}%`
                    : install?.error
                        ? `取得に失敗しました: ${install.error}`
                        : health.llm.binary
                            ? `見つかっています${SOURCE_LABEL[health.llm.source ?? ''] ?? ''}が、起動していません`
                            : `この Mac にありません。押すとアプリ用に取得します (約 ${health.llm.download_mb ?? 150} MB、システムには入れません)`,
            action: health.llm.server || install?.active ? undefined : {
                label: health.llm.binary ? '起動する' : '用意する',
                run: run('ollama', () => api.llmStart().then(r => {
                    if (!r.running && !r.installing) throw new Error(r.error ?? 'Ollama を起動できませんでした');
                })),
            },
        },
        {
            key: 'model',
            label: `LLM モデル (${health.llm.model})`,
            level: !health.llm.server ? 'pending' : health.llm.model_available ? 'ok' : pull?.active ? 'pending' : 'warn',
            detail: !health.llm.server ? 'Ollama の起動後に確認します'
                : health.llm.model_available ? 'ダウンロード済み'
                    : pull?.active ? `ダウンロード中 ${pull.total ? Math.round(((pull.completed ?? 0) / pull.total) * 100) : 0}%`
                        : pull?.error ? `ダウンロード失敗: ${pull.error}` : 'まだダウンロードされていません (数 GB)',
            action: health.llm.server && !health.llm.model_available && !pull?.active
                ? { label: 'ダウンロード', run: run('model', () => api.llmPull(health.llm.model)) } : undefined,
        },
        {
            key: 'disk',
            label: '空きディスク',
            level: health.disk_free_gb >= 20 ? 'ok' : health.disk_free_gb >= 8 ? 'warn' : 'bad',
            detail: `${health.disk_free_gb} GB${health.disk_free_gb < 20 ? ' (モデルの保存に 10 GB 以上あると安心です)' : ''}`,
        },
    ];

    const problems = items.filter(i => i.level === 'bad' || i.level === 'warn');
    if (compact && problems.length === 0) {
        return (
            <div className="setup setup-compact setup-allok">
                <Icon name="check" /> 準備完了: 予測・変異スコア・AI アシスタントがすべて使えます
                {health.machine.memory_gb && <span className="muted small"> · {health.machine.chip} / {health.machine.memory_gb} GB</span>}
            </div>
        );
    }
    const shown = compact ? problems : items;
    return (
        <div className={`setup ${compact ? 'setup-compact' : ''}`}>
            {compact && <div className="setup-title"><Icon name="info" /> 準備状況</div>}
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
                    {health.machine.os} · {health.machine.chip}{health.machine.memory_gb ? ` · メモリ ${health.machine.memory_gb} GB` : ''} · Python {health.machine.python} · Oritatami {health.version}
                </div>
            )}
        </div>
    );
}
