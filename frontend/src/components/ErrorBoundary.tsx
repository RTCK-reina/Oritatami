import { Component, type ErrorInfo, type ReactNode } from 'react';
import { t } from '../i18n';

interface Props {
    /** shown in the fallback so the user knows which part failed */
    area: string;
    children: ReactNode;
}

interface State {
    error: Error | null;
}

/** Keeps a rendering bug in one panel from blanking the whole window. */
export class ErrorBoundary extends Component<Props, State> {
    state: State = { error: null };

    static getDerivedStateFromError(error: Error): State {
        return { error };
    }

    componentDidCatch(error: Error, info: ErrorInfo): void {
        console.error(`[Oritatami] ${this.props.area} ${t('の表示でエラー')}`, error, info.componentStack);
    }

    render(): ReactNode {
        if (!this.state.error) return this.props.children;
        return (
            <div className="panel boundary" role="alert">
                <strong>{this.props.area}{t('を表示できませんでした')}</strong>
                <pre className="error-box">{this.state.error.message}</pre>
                <div className="row">
                    <button type="button" className="btn btn-md btn-primary" onClick={() => this.setState({ error: null })}>{t('もう一度表示する')}</button>
                    <button type="button" className="btn btn-md btn-ghost" onClick={() => window.location.reload()}>{t('画面を再読み込み')}</button>
                </div>
                <p className="small muted">{t('作業台の内容は保存されています。再読み込みしても消えません。')}</p>
            </div>
        );
    }
}
