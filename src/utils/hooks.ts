import { useEffect, useRef, useState, useSyncExternalStore, type RefObject } from 'react';
import { useSettings } from '@/stores/settings';

/** Observed content-box width of an element. */
export function useElementWidth<T extends HTMLElement>(): [RefObject<T | null>, number] {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setWidth(el.clientWidth);
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setWidth(Math.floor(e.contentRect.width));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

export interface ThemeColors {
  fg: string;
  muted: string;
  accent: string;
  teal: string;
  grid: string;
  line: string;
  panel: string;
  bg: string;
}

function readColors(): ThemeColors {
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string, fb: string) => cs.getPropertyValue(n).trim() || fb;
  return {
    fg: v('--fg', '#f4f7fb'),
    muted: v('--muted', '#a1aec2'),
    accent: v('--accent', '#8b8cff'),
    teal: v('--teal', '#38d9c5'),
    grid: v('--grid-bg', '#0d1320'),
    line: v('--line', '#243149'),
    panel: v('--panel', '#111827'),
    bg: v('--bg', '#080b12'),
  };
}

/** Current theme colors for canvas drawing; updates when the theme attribute changes. */
export function useThemeColors(): ThemeColors {
  const [colors, setColors] = useState<ThemeColors>(() => readColors());
  useEffect(() => {
    const mo = new MutationObserver(() => setColors(readColors()));
    mo.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme', 'data-contrast'],
    });
    return () => mo.disconnect();
  }, []);
  return colors;
}

const REDUCED_QUERY = '(prefers-reduced-motion: reduce)';
function subscribeReduced(cb: () => void) {
  const mq = window.matchMedia?.(REDUCED_QUERY);
  mq?.addEventListener?.('change', cb);
  return () => mq?.removeEventListener?.('change', cb);
}

export function useReducedMotion(): boolean {
  const motion = useSettings((s) => s.motion);
  const system = useSyncExternalStore(
    subscribeReduced,
    () => !!window.matchMedia?.(REDUCED_QUERY).matches,
    () => false,
  );
  return motion === 'system' ? system : motion === 'reduce';
}

export function fmt(v: number, digits = 4): string {
  if (!Number.isFinite(v)) return v === Number.NEGATIVE_INFINITY ? '−∞' : '—';
  if (v !== 0 && Math.abs(v) < 10 ** -digits) return v.toExponential(2);
  return v.toFixed(digits);
}

export function pct(v: number, digits = 1): string {
  return `${(v * 100).toFixed(digits)}%`;
}
