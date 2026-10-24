import { useMemo, useRef, useState, type KeyboardEvent } from 'react';
import type { TokenRecord } from '@/types/contracts';
import { TokenText, cn } from '@/components/ui/primitives';
import { Switch } from '@/components/ui/controls';
import { fmt, useElementWidth } from '@/utils/hooks';

const ROW = 26;

/**
 * Bipartite attention diagram: queries on the left, keys on the right.
 * Line width and opacity are proportional to the attention probability.
 */
export function ArcView({
  values,
  T,
  tokens,
  selectedQuery,
  selectedKey,
  onSelect,
}: {
  values: ArrayLike<number>;
  T: number;
  tokens: TokenRecord[];
  selectedQuery: number;
  selectedKey: number | null;
  onSelect: (q: number, k: number | null) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const [threshold, setThreshold] = useState(0.15);
  const [wrapRef, width] = useElementWidth<HTMLDivElement>();
  const listRef = useRef<HTMLDivElement>(null);
  const colW = Math.min(170, Math.max(96, width * 0.28));
  const midW = Math.max(80, width - 2 * colW);
  const H = T * ROW + 8;
  const y = (i: number) => 4 + i * ROW + ROW / 2;

  const lines = useMemo(() => {
    const out: { q: number; k: number; w: number; sel: boolean }[] = [];
    for (let i = 0; i < T; i++) {
      const sel = i === selectedQuery;
      if (!sel && !showAll) continue;
      for (let j = 0; j <= i; j++) {
        const w = values[i * T + j]!;
        if (sel ? w >= 0.003 : w >= threshold) out.push({ q: i, k: j, w, sel });
      }
    }
    return out.sort((a, b) => Number(a.sel) - Number(b.sel) || a.w - b.w);
  }, [values, T, selectedQuery, showAll, threshold]);

  const onKey = (e: KeyboardEvent) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    const q = Math.max(0, Math.min(T - 1, selectedQuery + (e.key === 'ArrowDown' ? 1 : -1)));
    onSelect(q, null);
    listRef.current?.querySelector<HTMLButtonElement>(`[data-q="${q}"]`)?.focus();
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <div className="w-60">
          <Switch
            id="arc-all"
            label="Show all queries"
            description="Draw every query's strong links, not only the selected one."
            checked={showAll}
            onChange={setShowAll}
          />
        </div>
        {showAll && (
          <label className="flex items-center gap-2 text-[12px] text-muted">
            Threshold
            <input
              type="range"
              min={0.02}
              max={0.9}
              step={0.01}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="accent-[var(--accent)]"
              aria-valuetext={`${threshold.toFixed(2)}`}
            />
            <span className="w-10 font-mono text-fg tabular">{threshold.toFixed(2)}</span>
          </label>
        )}
      </div>
      <div ref={wrapRef} className="w-full">
        <div className="flex" style={{ height: H }}>
          <div
            ref={listRef}
            role="listbox"
            aria-label="Query tokens"
            onKeyDown={onKey}
            className="shrink-0"
            style={{ width: colW }}
          >
            {tokens.map((t, i) => (
              <button
                key={i}
                id={`arc-q-${i}`}
                data-q={i}
                role="option"
                aria-selected={i === selectedQuery}
                tabIndex={i === selectedQuery ? 0 : -1}
                onClick={() => onSelect(i, null)}
                className={cn(
                  'flex w-full items-center justify-end gap-1.5 rounded px-1.5 text-right text-[12px]',
                  i === selectedQuery ? 'bg-accent-soft text-accent' : 'text-muted hover:text-fg',
                )}
                style={{ height: ROW }}
              >
                <TokenText text={t.text} className="truncate" />
                <span className="w-5 shrink-0 font-mono text-[10px] opacity-70">{i}</span>
              </button>
            ))}
          </div>
          <svg width={midW} height={H} aria-hidden className="shrink-0 overflow-visible">
            {lines.map(({ q, k, w, sel }) => (
              <path
                key={`${q}-${k}`}
                d={`M 0 ${y(q)} C ${midW * 0.5} ${y(q)}, ${midW * 0.5} ${y(k)}, ${midW} ${y(k)}`}
                fill="none"
                stroke={sel ? (k === selectedKey ? 'var(--teal)' : 'var(--accent)') : 'var(--muted)'}
                strokeOpacity={sel ? 0.18 + 0.82 * w : 0.12 + 0.6 * w}
                strokeWidth={sel ? 0.75 + 7 * w : 0.5 + 3 * w}
                strokeLinecap="round"
              />
            ))}
          </svg>
          <ul aria-label={`Attention from query ${selectedQuery} to each key`} className="min-w-0 flex-1">
            {tokens.map((t, j) => {
              const w = j <= selectedQuery ? values[selectedQuery * T + j]! : null;
              return (
                <li key={j} style={{ height: ROW }} className="flex items-center gap-1.5 pl-1.5 text-[12px]">
                  <button
                    type="button"
                    onClick={() => onSelect(selectedQuery, j)}
                    className={cn(
                      'flex min-w-0 items-center gap-1.5 rounded px-1 text-left',
                      j === selectedKey ? 'bg-teal-soft text-teal' : w === null ? 'text-muted/50' : 'text-fg',
                    )}
                  >
                    <span className="w-5 shrink-0 font-mono text-[10px] text-muted">{j}</span>
                    <TokenText text={t.text} className="truncate" />
                  </button>
                  {w === null ? (
                    <span className="ml-auto pr-1 text-[10.5px] text-muted/60">masked</span>
                  ) : (
                    <span className="ml-auto flex items-center gap-1.5 pr-1">
                      <span
                        aria-hidden
                        className="hidden h-1.5 w-12 overflow-hidden rounded bg-elevated sm:block"
                      >
                        <span className="block h-full bg-accent" style={{ width: `${w * 100}%` }} />
                      </span>
                      <span className="w-14 text-right font-mono text-[11px] tabular text-fg">
                        {fmt(w, 4)}
                      </span>
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </div>
  );
}
