/** Two-language UI: Japanese source literals double as dictionary keys.
 *
 * `t('概要')` returns the English string when the UI language is English, and the
 * Japanese literal itself otherwise — untranslated keys degrade gracefully. The
 * language is chosen in Settings and persisted in localStorage; switching reloads
 * the app so module-level constants re-evaluate in the new language.
 */
import { EN } from './locales/en';

export type Lang = 'ja' | 'en';
const KEY = 'oritatami.lang';

const lang: Lang = localStorage.getItem(KEY) === 'en' ? 'en' : 'ja';
document.documentElement.lang = lang;

export function getLang(): Lang {
    return lang;
}

export function setLang(next: Lang) {
    if (next === lang) return;
    localStorage.setItem(KEY, next);
    location.reload();
}

/** Translate a Japanese UI literal: the inline `en` arg wins, then the dictionary,
 * then the Japanese itself. */
export function t(ja: string, en?: string): string {
    return lang === 'ja' ? ja : en ?? EN[ja] ?? ja;
}
