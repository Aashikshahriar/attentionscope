import { useEffect, useMemo, useRef, useState } from 'react';
import type { AttentionRun } from '@/types/contracts';
import type { HeadStats } from '@/core/metrics/headMetrics';
import { headMatrix } from '@/core/attention/tensor';
import { dataWorker } from '@/services/workers';
import { useWorkspace } from '@/stores/workspace';
import { useSettings } from '@/stores/settings';
import { log } from '@/stores/log';
import { Callout, Panel, Spinner, TokenText, cn } from '@/components/ui/primitives';
import { Select, Switch } from '@/components/ui/controls';
import { Tex } from '@/components/ui/Tex';
import { Heatmap } from '@/features/heatmap/Heatmap';
import { colorFor } from '@/features/heatmap/colormaps';
import { fmt, pct, useThemeColors } from '@/utils/hooks';

type Metric =
  | 'normalizedEntropy'
  | 'meanDistance'
  | 'previousToken'
  | 'firstToken'
  | 'selfAttention'
  | 'simJs'
  | 'simPearson';

const METRICS: Record<Metric, { label: string; description: string }> = {
  normalizedEntropy: {
    label: 'Normalized entropy',
    description: 'Mean over rows of H_i / log(visible keys). 0 = focused, 1 = uniform.',
  },
  meanDistance: {
    label: 'Mean attention distance',
    description: 'Mean over rows of Σ_j a_ij (i − j), in tokens.',
  },
  previousToken: { label: 'Previous-token score', description: 'Mean attention a_{i,i−1} over i ≥ 1.' },
  firstToken: { label: 'First-token (sink) score', description: 'Mean attention to position 0 over i ≥ 1.' },
  selfAttention: { label: 'Self-attention score', description: 'Mean attention a_{i,i} over i ≥ 1.' },
  simJs: {
    label: 'Similarity to selected head (1 − JSD)',
    description: '1 − mean Jensen–Shannon divergence (bits) between rows; 1 = identical patterns.',
  },
  simPearson: {
    label: 'Similarity to selected head (Pearson r)',
    description:
      'Correlation of all causally visible entries (rows i ≥ 1). Sensitive to sequence length; use with care.',
  },
};

function MiniMap({ m, T, size }: { m: Float32Array; T: number; size: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { colormap, colorScale } = useSettings();
  const colors = useThemeColors();
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = size * dpr;
    c.height = size * dpr;
    const ctx = c.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const s = size / T;
    ctx.fillStyle = colors.grid;
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < T; i++)
      for (let j = 0; j <= i; j++) {
        ctx.fillStyle = colorFor(colormap, m[i * T + j]!, colorScale);
        ctx.fillRect(j * s, i * s, Math.ceil(s), Math.ceil(s));
      }
  }, [m, T, size, colormap, colorScale, colors]);
  return <canvas ref={ref} aria-hidden style={{ width: size, height: size }} className="block rounded-sm" />;
}

function DotPlot({
  values,
  selected,
  onSelect,
  label,
}: {
  values: number[][];
  selected: [number, number];
  onSelect: (l: number, h: number) => void;
  label: string;
}) {
  const L = values.length;
  const flat = values.flat().filter(Number.isFinite);
  const lo = Math.min(0, ...flat);
  const hi = Math.max(...flat, lo + 1e-9);
  const W = 640;
  const H = 220;
  const pad = { l: 44, r: 12, t: 12, b: 30 };
  const x = (l: number) => pad.l + ((l + 0.5) / L) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
  const [hover, setHover] = useState<{ l: number; h: number } | null>(null);
  const ticks = [lo, (lo + hi) / 2, hi];
  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={`${label} by layer; one dot per head`}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeDasharray="2 4" />
            <text
              x={pad.l - 6}
              y={y(t) + 3.5}
              textAnchor="end"
              fontSize="10"
              fill="var(--muted)"
              className="font-mono"
            >
              {fmt(t, 2)}
            </text>
          </g>
        ))}
        {values.map((_row, l) => (
          <text
            key={l}
            x={x(l)}
            y={H - 10}
            textAnchor="middle"
            fontSize="10"
            fill="var(--muted)"
            className="font-mono"
          >
            {l}
          </text>
        ))}
        <text x={pad.l} y={H - 10} textAnchor="end" fontSize="10" fill="var(--muted)" dx={-8}>
          layer
        </text>
        {values.map((row, l) =>
          row.map((v, h) => {
            if (!Number.isFinite(v)) return null;
            const sel = selected[0] === l && selected[1] === h;
            const jitter = (h / Math.max(1, row.length - 1) - 0.5) * ((W - pad.l - pad.r) / L) * 0.55;
            return (
              <circle
                key={`${l}-${h}`}
                cx={x(l) + jitter}
                cy={y(v)}
                r={sel ? 6 : 4}
                fill={sel ? 'var(--teal)' : 'var(--accent)'}
                fillOpacity={sel ? 1 : 0.75}
                stroke="var(--panel)"
                strokeWidth={2}
                className="cursor-pointer"
                onMouseEnter={() => setHover({ l, h })}
                onMouseLeave={() => setHover(null)}
                onClick={() => onSelect(l, h)}
              >
                <title>{`L${l} H${h}: ${fmt(v, 4)}`}</title>
              </circle>
            );
          }),
        )}
      </svg>
      {hover && (
        <div className="pointer-events-none absolute right-2 top-2 rounded-md border border-line-strong bg-panel px-2 py-1 font-mono text-[11.5px] text-fg shadow">
          L{hover.l} H{hover.h}: {fmt(values[hover.l]![hover.h]!, 4)}
        </div>
      )}
    </div>
  );
}

export function CompareView({ run }: { run: AttentionRun }) {
  const { layer, head, setView, queryIndex } = useWorkspace();
  const { colormap } = useSettings();
  const [statsState, setStatsState] = useState<{ key: string; stats: HeadStats[][] } | null>(null);
  const [similarity, setSimilarity] = useState<{ key: string; values: number[][] } | null>(null);
  const [metric, setMetric] = useState<Metric>('normalizedEntropy');
  const [residualWeight, setResidualWeight] = useState(0.5);
  const [rollout, setRollout] = useState<Float64Array | null>(null);
  const [showTable, setShowTable] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const { layers: L, heads: H, queryLength: T } = run.attention.shape;
  const key = run.id;
  const stats = statsState?.key === key ? statsState.stats : null;

  useEffect(() => {
    let cancelled = false;
    const t0 = performance.now();
    dataWorker()
      .call('registerTensor', { key, shape: run.attention.shape, data: run.attention.data.slice() })
      .then(() => dataWorker().call('headStats', { key }))
      .then((s) => {
        if (cancelled) return;
        setStatsState({ key, stats: s });
        log({
          level: 'info',
          scope: 'metrics',
          message: `Computed entropy, distance and pattern scores for ${L * H} heads`,
          durationMs: performance.now() - t0,
        });
      })
      .catch((e: unknown) => !cancelled && setErr(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [key, run, L, H]);

  const simMetric = metric === 'simJs' ? 'js' : metric === 'simPearson' ? 'pearson' : null;
  useEffect(() => {
    if (!simMetric || !stats) return;
    let cancelled = false;
    dataWorker()
      .call('similarity', { key, layer, head, metric: simMetric })
      .then((v) => !cancelled && setSimilarity({ key: `${key}:${layer}:${head}:${simMetric}`, values: v }))
      .catch((e: unknown) => !cancelled && setErr(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [simMetric, key, layer, head, stats]);

  useEffect(() => {
    if (!stats) return;
    let cancelled = false;
    dataWorker()
      .call('rollout', { key, residualWeight })
      .then((r) => !cancelled && setRollout(r))
      .catch((e: unknown) => !cancelled && setErr(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [key, residualWeight, stats]);

  const values: number[][] | null = useMemo(() => {
    if (!stats) return null;
    if (simMetric) {
      return similarity?.key === `${key}:${layer}:${head}:${simMetric}` ? similarity.values : null;
    }
    return stats.map((row) => row.map((s) => s[metric as keyof HeadStats] as number));
  }, [stats, metric, simMetric, similarity, key, layer, head]);

  const range = useMemo(() => {
    const f = (values ?? []).flat().filter(Number.isFinite);
    return { lo: Math.min(...f), hi: Math.max(...f) };
  }, [values]);
  const norm = (v: number) => (range.hi > range.lo ? (v - range.lo) / (range.hi - range.lo) : 0.5);
  const cellSize = Math.max(34, Math.min(64, Math.floor(560 / H)));

  if (err)
    return (
      <Callout tone="danger" title="Comparison failed">
        {err}
      </Callout>
    );

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end gap-4">
        <div className="w-full max-w-xs">
          <label htmlFor="metric" className="mb-1.5 block text-[12px] font-medium text-muted">
            Metric
          </label>
          <Select<Metric>
            id="metric"
            label="Metric"
            value={metric}
            onChange={setMetric}
            options={(Object.keys(METRICS) as Metric[]).map((k) => ({ value: k, label: METRICS[k].label }))}
          />
        </div>
        <p className="max-w-md text-[12px] leading-snug text-muted">{METRICS[metric].description}</p>
        <div className="ml-auto w-40">
          <Switch id="cmp-table" label="Table view" checked={showTable} onChange={setShowTable} />
        </div>
      </div>

      {!values ? (
        <div className="flex items-center gap-2 py-10 text-sm text-muted" role="status">
          <Spinner /> Computing in a Web Worker…
        </div>
      ) : showTable ? (
        <div className="overflow-auto rounded-lg border border-line">
          <table className="text-[12px]">
            <caption className="sr-only">{METRICS[metric].label} for every layer and head</caption>
            <thead className="bg-elevated text-muted">
              <tr>
                <th scope="col" className="px-2 py-1 text-left">
                  layer \ head
                </th>
                {Array.from({ length: H }, (_, h) => (
                  <th key={h} scope="col" className="px-2 py-1 text-right font-mono">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {values.map((row, l) => (
                <tr key={l} className="border-t border-line">
                  <th scope="row" className="px-2 py-1 text-left font-mono text-muted">
                    {l}
                  </th>
                  {row.map((v, h) => (
                    <td
                      key={h}
                      className={cn(
                        'px-2 py-1 text-right font-mono tabular',
                        l === layer && h === head && 'bg-accent-soft text-accent',
                      )}
                    >
                      {fmt(v, 3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="overflow-x-auto scrollbar-thin">
          <div
            role="grid"
            aria-label={`Attention patterns for all ${L * H} heads with ${METRICS[metric].label}`}
            className="inline-grid gap-1.5"
            style={{ gridTemplateColumns: `2rem repeat(${H}, ${cellSize}px)` }}
          >
            <div role="row" className="contents">
              <span role="columnheader" className="text-[10px] text-muted" />
              {Array.from({ length: H }, (_, h) => (
                <span role="columnheader" key={h} className="text-center font-mono text-[10px] text-muted">
                  H{h}
                </span>
              ))}
            </div>
            {Array.from({ length: L }, (_, l) => (
              <div role="row" key={l} className="contents">
                <span role="rowheader" className="self-center font-mono text-[10px] text-muted">
                  L{l}
                </span>
                {Array.from({ length: H }, (_, h) => {
                  const v = values[l]![h]!;
                  const sel = l === layer && h === head;
                  const shown = Number.isFinite(v) ? v.toFixed(Math.abs(v) >= 10 ? 1 : 2) : '—';
                  return (
                    <button
                      key={h}
                      role="gridcell"
                      type="button"
                      aria-selected={sel}
                      aria-label={`${shown} — layer ${l} head ${h}, ${METRICS[metric].label}`}
                      onClick={() => setView({ layer: l, head: h })}
                      className={cn(
                        'group rounded-md p-0.5 ring-offset-1 ring-offset-[var(--panel)]',
                        sel ? 'ring-2 ring-teal' : 'hover:ring-1 hover:ring-line-strong',
                      )}
                    >
                      <MiniMap m={headMatrix(run.attention, l, h)} T={T} size={cellSize - 6} />
                      <span
                        className="mt-0.5 block rounded-sm text-center font-mono text-[9.5px] tabular"
                        style={{
                          boxShadow: `inset 0 -2px 0 ${Number.isFinite(v) ? colorFor(colormap, norm(v)) : 'transparent'}`,
                        }}
                      >
                        {shown}
                      </span>
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      )}

      {values && (
        <Panel title={`${METRICS[metric].label} by layer`} id="cmp-dots">
          <DotPlot
            values={values}
            selected={[layer, head]}
            onSelect={(l, h) => setView({ layer: l, head: h })}
            label={METRICS[metric].label}
          />
          <p className="mt-1 text-[11.5px] text-muted">
            Each dot is one head (jittered horizontally within its layer). Teal = selected head. Click a dot
            to select it.
          </p>
        </Panel>
      )}

      {(simMetric || metric === 'normalizedEntropy') && (
        <Callout tone="info" title="Reading these metrics">
          Metrics summarize a single input. Entropy and similarity depend on sequence length and on the
          first-token sink; two heads with similar patterns can still write very different information (their
          values and W<sub>O</sub> differ). Similarity is not a test of functional equivalence.
        </Callout>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Attention rollout" id="cmp-rollout">
          <div className="space-y-3">
            <Tex
              block
            >{String.raw`\hat A_\ell = \operatorname{rownorm}\!\big(\alpha I + (1-\alpha)\tfrac{1}{H}\textstyle\sum_h A_{\ell,h}\big)`}</Tex>
            <Tex block>{String.raw`R = \hat A_{L-1}\cdots \hat A_1 \hat A_0`}</Tex>
            <label className="flex items-center gap-2 text-[12px] text-muted">
              Residual weight α
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={residualWeight}
                onChange={(e) => setResidualWeight(Number(e.target.value))}
                className="accent-[var(--accent)]"
              />
              <span className="font-mono text-fg">{residualWeight.toFixed(2)}</span>
            </label>
            {rollout ? (
              <Heatmap
                values={rollout}
                T={T}
                tokens={run.input.tokens}
                mode="probabilities"
                selectedQuery={queryIndex}
                selectedKey={null}
                onSelect={(q) => setView({ queryIndex: q })}
                label="Attention rollout across all layers"
              />
            ) : (
              <Spinner />
            )}
            <Callout tone="warn" title="Assumptions (Abnar & Zuidema, 2020)">
              Heads are averaged equally, the residual stream is modelled as a fixed identity mix α, and
              values, W<sub>O</sub>, MLPs and LayerNorm are ignored. Rollout is a heuristic estimate of
              information flow, not a measurement of causal influence.
            </Callout>
          </div>
        </Panel>
        <div className="space-y-4">
          {run.logitLens && (
            <Panel title={`Logit lens · position ${run.logitLens.position}`} id="cmp-lens">
              <p className="mb-2 text-[12px] text-muted">
                Decoding the residual stream after each layer with the final LayerNorm and unembedding. Early
                layers were not trained to be read this way; treat as a rough probe.
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-[12px]">
                  <thead className="text-muted">
                    <tr>
                      <th scope="col" className="py-1 pr-2 text-left font-medium">
                        after layer
                      </th>
                      <th scope="col" className="py-1 pr-2 text-left font-medium">
                        top-1
                      </th>
                      <th scope="col" className="py-1 pr-2 text-right font-medium">
                        p
                      </th>
                      {run.logitLens.layers[0]?.targetProb !== undefined && (
                        <th scope="col" className="py-1 text-right font-medium">
                          p(target {JSON.stringify(run.predictions?.target?.text)})
                        </th>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {run.logitLens.layers.map((l) => (
                      <tr key={l.layer} className="border-t border-line">
                        <td className="py-1 pr-2 font-mono text-muted">{l.layer}</td>
                        <td className="py-1 pr-2">
                          <TokenText text={l.top[0]?.text ?? ''} className="text-fg" />
                        </td>
                        <td className="py-1 pr-2 text-right font-mono tabular">{pct(l.top[0]?.prob ?? 0)}</td>
                        {l.targetProb !== undefined && (
                          <td className="py-1 text-right font-mono tabular">{pct(l.targetProb, 2)}</td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
          {run.residualNorms && (
            <Panel title={`Residual stream norm · token ${queryIndex}`} id="cmp-resid">
              <ResidualBars values={run.residualNorms.values.map((r) => r[queryIndex] ?? 0)} />
              <p className="mt-2 text-[11.5px] text-muted">{run.residualNorms.method}.</p>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}

function ResidualBars({ values }: { values: number[] }) {
  const max = Math.max(...values, 1e-9);
  return (
    <ol className="space-y-1">
      {values.map((v, i) => (
        <li key={i} className="grid grid-cols-[4.5rem_minmax(0,1fr)_4rem] items-center gap-2 text-[11.5px]">
          <span className="font-mono text-muted">{i === 0 ? 'embed' : `layer ${i - 1}`}</span>
          <span aria-hidden className="h-2 overflow-hidden rounded-sm bg-elevated">
            <span className="block h-full rounded-sm bg-accent" style={{ width: `${(v / max) * 100}%` }} />
          </span>
          <span className="text-right font-mono tabular text-fg">{fmt(v, 1)}</span>
        </li>
      ))}
    </ol>
  );
}
