import { useCallback, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import type { TokenRecord } from '@/types/contracts';
import { useSettings } from '@/stores/settings';
import { fmt, useElementWidth, useThemeColors } from '@/utils/hooks';
import { colorFor, gradientCss } from './colormaps';

export interface HeatmapProps {
  values: ArrayLike<number>;
  T: number;
  tokens: TokenRecord[];
  mode: 'probabilities' | 'scores';
  selectedQuery: number;
  selectedKey: number | null;
  onSelect: (query: number, key: number | null) => void;
  label: string;
}

const label = (s: string, n = 14) => {
  const v = s.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '⇥');
  return v.length > n ? `${v.slice(0, n - 1)}…` : v || '∅';
};

/**
 * Canvas heatmap of a [query, key] matrix: rows are queries (who attends),
 * columns are keys (attended to). Masked cells (key > query) are hatched.
 */
export function Heatmap({
  values,
  T,
  tokens,
  mode,
  selectedQuery,
  selectedKey,
  onSelect,
  label: aria,
}: HeatmapProps) {
  const [wrapRef, width] = useElementWidth<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const colors = useThemeColors();
  const { colormap, colorScale } = useSettings();
  const [hover, setHover] = useState<{ q: number; k: number; x: number; y: number } | null>(null);
  const [announce, setAnnounce] = useState('');
  const descId = useId();

  const { min, max } = useMemo(() => {
    if (mode === 'probabilities') return { min: 0, max: 1 };
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < T; i++)
      for (let j = 0; j <= i; j++) {
        const v = values[i * T + j]!;
        if (Number.isFinite(v)) {
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
      }
    return { min: lo, max: hi === lo ? lo + 1 : hi };
  }, [values, T, mode]);

  const longest = useMemo(
    () => Math.max(4, ...tokens.map((t) => Math.min(14, label(t.text).length))),
    [tokens],
  );
  const left = Math.min(132, 22 + longest * 7.2);
  const top = Math.min(104, 26 + longest * 5.2);
  const cell = Math.max(10, Math.min(46, Math.floor((Math.max(width, 200) - left - 6) / T)));
  const W = left + cell * T + 4;
  const H = top + cell * T + 4;

  const draw = useCallback(() => {
    const c = canvasRef.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(W * dpr);
    c.height = Math.round(H * dpr);
    c.style.width = `${W}px`;
    c.style.height = `${H}px`;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const norm = (v: number) => (mode === 'probabilities' ? v : (v - min) / (max - min));
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        const x = left + j * cell;
        const y = top + i * cell;
        if (j > i) {
          ctx.fillStyle = colors.grid;
          ctx.fillRect(x, y, cell - 1, cell - 1);
          ctx.strokeStyle = colors.line;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x + 2, y + cell - 3);
          ctx.lineTo(x + cell - 3, y + 2);
          ctx.stroke();
          continue;
        }
        const v = values[i * T + j]!;
        ctx.fillStyle = colorFor(colormap, norm(v), colorScale);
        ctx.fillRect(x, y, cell - 1, cell - 1);
        if (cell >= 38) {
          const n = norm(v);
          ctx.fillStyle = n > 0.55 ? '#0b0f17' : '#e8edf6';
          ctx.font = '10px ui-monospace, monospace';
          ctx.textAlign = 'center';
          ctx.fillText(
            mode === 'probabilities' ? v.toFixed(2) : v.toFixed(1),
            x + cell / 2,
            y + cell / 2 + 3.5,
          );
        }
      }
    }
    // selection outlines
    ctx.lineWidth = 2;
    ctx.strokeStyle = colors.accent;
    ctx.strokeRect(left - 1, top + selectedQuery * cell - 1, cell * T + 1, cell + 1);
    if (selectedKey !== null) {
      ctx.strokeStyle = colors.teal;
      ctx.strokeRect(left + selectedKey * cell - 1, top - 1, cell + 1, cell * T + 1);
    }
    if (hover) {
      ctx.strokeStyle = colors.fg;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(left + hover.k * cell - 0.5, top + hover.q * cell - 0.5, cell, cell);
    }
    // labels
    const fontPx = Math.max(9, Math.min(12, cell * 0.6 + 4));
    ctx.font = `${fontPx}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
    for (let i = 0; i < T; i++) {
      const t = tokens[i]!;
      const isQ = i === selectedQuery;
      ctx.fillStyle = isQ ? colors.accent : t.special ? colors.teal : colors.muted;
      ctx.textAlign = 'right';
      ctx.fillText(label(t.text), left - 6, top + i * cell + cell / 2 + fontPx / 3);
      ctx.save();
      ctx.translate(left + i * cell + cell / 2 + fontPx / 3, top - 6);
      ctx.rotate(-Math.PI / 3.2);
      ctx.textAlign = 'left';
      ctx.fillStyle = i === selectedKey ? colors.teal : t.special ? colors.teal : colors.muted;
      ctx.fillText(label(t.text), 0, 0);
      ctx.restore();
    }
    ctx.fillStyle = colors.muted;
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('query ↓  key →', 4, 12);
  }, [
    W,
    H,
    T,
    cell,
    left,
    top,
    values,
    tokens,
    colors,
    colormap,
    colorScale,
    mode,
    min,
    max,
    selectedQuery,
    selectedKey,
    hover,
  ]);

  useEffect(() => {
    draw();
  }, [draw]);

  const cellAt = (clientX: number, clientY: number) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const k = Math.floor((x - left) / cell);
    const q = Math.floor((y - top) / cell);
    if (q < 0 || k < 0 || q >= T || k >= T) return null;
    return { q, k, x, y };
  };

  const describe = (q: number, k: number) => {
    const qt = JSON.stringify(tokens[q]!.text);
    const kt = JSON.stringify(tokens[k]!.text);
    if (k > q) return `Query ${q} ${qt}, key ${k} ${kt}: masked by the causal mask (future position).`;
    const v = values[q * T + k]!;
    return `Query ${q} ${qt}, key ${k} ${kt}: ${mode === 'probabilities' ? 'attention probability' : 'pre-softmax score'} ${fmt(v, 6)}.`;
  };

  const onKey = (e: KeyboardEvent) => {
    const k0 = selectedKey ?? selectedQuery;
    let q = selectedQuery;
    let k = k0;
    switch (e.key) {
      case 'ArrowUp':
        q = Math.max(0, q - 1);
        break;
      case 'ArrowDown':
        q = Math.min(T - 1, q + 1);
        break;
      case 'ArrowLeft':
        k = Math.max(0, k - 1);
        break;
      case 'ArrowRight':
        k = Math.min(T - 1, k + 1);
        break;
      case 'Home':
        k = 0;
        break;
      case 'End':
        k = q;
        break;
      case 'Escape':
        onSelect(q, null);
        setAnnounce(`Key selection cleared. Query ${q}.`);
        e.preventDefault();
        return;
      default:
        return;
    }
    e.preventDefault();
    onSelect(q, k);
    setAnnounce(describe(q, k));
  };

  const hv = hover && hover.k <= hover.q ? values[hover.q * T + hover.k]! : null;

  return (
    <div className="space-y-3">
      <div ref={wrapRef} className="relative w-full overflow-x-auto scrollbar-thin">
        <div
          role="group"
          aria-roledescription="heatmap"
          aria-label={aria}
          aria-describedby={descId}
          tabIndex={0}
          onKeyDown={onKey}
          className="relative inline-block rounded-md"
        >
          <canvas
            ref={canvasRef}
            aria-hidden
            className="block cursor-crosshair"
            onMouseMove={(e) => setHover(cellAt(e.clientX, e.clientY))}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => {
              const c = cellAt(e.clientX, e.clientY);
              if (c) {
                onSelect(c.q, c.k);
                setAnnounce(describe(c.q, c.k));
              }
            }}
          />
          {hover && (
            <div
              role="presentation"
              className="pointer-events-none absolute z-10 w-max max-w-64 rounded-md border border-line-strong bg-panel/95 px-2.5 py-1.5 text-xs shadow-xl"
              style={{
                left: Math.min(hover.x + 14, W - 200),
                top: hover.y + 14,
              }}
            >
              <div className="font-mono text-muted">
                q{hover.q} <span className="text-fg">{label(tokens[hover.q]!.text, 18)}</span> → k{hover.k}{' '}
                <span className="text-fg">{label(tokens[hover.k]!.text, 18)}</span>
              </div>
              {hv === null ? (
                <div className="text-muted">masked (future key) · probability exactly 0</div>
              ) : (
                <div className="font-mono text-fg tabular">
                  {mode === 'probabilities' ? 'p = ' : 'score = '}
                  {fmt(hv, 6)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      <p id={descId} className="sr-only">
        {T} by {T} matrix. Rows are query tokens, columns are key tokens. Entries above the diagonal are
        masked by the causal mask. Use arrow keys to move between cells; values are announced. A table view
        with every value is available.
      </p>
      <div aria-live="polite" className="sr-only">
        {announce}
      </div>
      <Legend mode={mode} min={min} max={max} />
    </div>
  );
}

function Legend({ mode, min, max }: { mode: 'probabilities' | 'scores'; min: number; max: number }) {
  const { colormap, colorScale } = useSettings();
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[11.5px] text-muted">
      <div className="flex items-center gap-2">
        <span className="font-mono tabular">{mode === 'probabilities' ? '0' : fmt(min, 2)}</span>
        <span
          aria-hidden
          className="h-2.5 w-32 rounded-sm sm:w-44"
          style={{ background: gradientCss(colormap) }}
        />
        <span className="font-mono tabular">{mode === 'probabilities' ? '1' : fmt(max, 2)}</span>
        <span>
          {mode === 'probabilities'
            ? 'attention probability'
            : 'pre-softmax score q·k/√d (color normalized to visible range)'}
          {colorScale === 'sqrt' && ' · √ color scale'}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <span aria-hidden className="hatch inline-block size-3 rounded-sm border border-line bg-grid" />{' '}
        masked (key after query)
      </div>
    </div>
  );
}
