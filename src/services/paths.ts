/**
 * Absolute URL for a file in `public/`, safe under the GitHub Pages sub-path
 * (Vite's BASE_URL is "/attentionscope/"). Workers receive absolute URLs so
 * that resolution never depends on the worker script's own location.
 */
export function assetUrl(relative: string): string {
  const base = import.meta.env.BASE_URL.endsWith('/')
    ? import.meta.env.BASE_URL
    : `${import.meta.env.BASE_URL}/`;
  return new URL(base + relative.replace(/^\/+/, ''), window.location.origin).href;
}

export const REPO_URL = 'https://github.com/aashikshahriar/attentionscope';
export const LIVE_URL = 'https://aashikshahriar.github.io/attentionscope/';
