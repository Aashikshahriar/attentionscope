import { useEffect, useMemo, useState } from 'react';
import { Dices, Pause, Play, RotateCcw } from 'lucide-react';
import { runToyAttention } from '@/core/attention/toyAttention';
import { initToyWeights } from './toyInit';
import type { Matrix } from '@/core/math/matrix';
import type { ToyWeights as ToyW } from '@/core/attention/toyAttention';
import { Badge, Button, Callout, Field, Panel, cn } from '@/components/ui/primitives';
import { Segmented, Stepper, Switch } from '@/components/ui/controls';
import { Tex } from '@/components/ui/Tex';
import { colorFor, textOn } from '@/features/heatmap/colormaps';
import { useSettings } from '@/stores/settings';
import { fmt, useReducedMotion } from '@/utils/hooks';

function MatrixView({
  m,
  title,
  rows,
  cols,
  heat,
  highlightRow,
  highlightCol,
  highlightCell,
  masked,
  onCell,
  digits = 2,
}: {
  m: Matrix;
  title: string;
  rows?: string[];
  cols?: string[];
  heat?: boolean;
  highlightRow?: number;
  highlightCol?: number;
  highlightCell?: [number, number];
  masked?: boolean[][];
  onCell?: (i: number, j: number) => void;
  digits?: number;
}) {
  const { colormap } = useSettings();
  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 text-[11.5px] text-muted">{title}</figcaption>
      <div className="overflow-x-auto scrollbar-thin">
        <table className="border-separate border-spacing-[2px] font-mono text-[11px]">
          {cols && (
            <thead>
              <tr>
                {rows && <th />}
                {cols.map((c, j) => (
                  <th
                    key={j}
                    scope="col"
                    className={cn('px-1 font-normal text-muted', highlightCol === j && 'text-teal')}
                  >
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
          )}
          <tbody>
            {m.map((row, i) => (
              <tr key={i}>
                {rows && (
                  <th
                    scope="row"
                    className={cn(
                      'pr-1 text-right font-normal text-muted',
                      highlightRow === i && 'text-accent',
                    )}
                  >
                    {rows[i]}
                  </th>
                )}
                {row.map((v, j) => {
                  const isMasked = masked?.[i]?.[j] === false;
                  const hl = highlightRow === i || highlightCol === j;
                  const cellHl = highlightCell && highlightCell[0] === i && highlightCell[1] === j;
                  const style =
                    heat && !isMasked && Number.isFinite(v)
                      ? { background: colorFor(colormap, v), color: textOn(colormap, v) }
                      : undefined;
                  const Tag = onCell ? 'button' : 'span';
                  return (
                    <td key={j} className="p-0">
                      <Tag
                        type={onCell ? 'button' : undefined}
                        onClick={onCell ? () => onCell(i, j) : undefined}
                        style={style}
                        aria-label={
                          onCell
                            ? `${title} row ${i} column ${j}: ${isMasked ? 'masked' : fmt(v, 4)}`
                            : undefined
                        }
                        className={cn(
                          'block min-w-[3.1rem] rounded-[4px] px-1.5 py-1 text-right tabular',
                          !style && 'bg-elevated text-fg',
                          isMasked && 'hatch text-center text-muted',
                          hl && !style && 'bg-accent-soft',
                          cellHl && 'ring-2 ring-teal',
                        )}
                      >
                        {isMasked ? '−∞' : fmt(v, digits)}
                      </Tag>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}

function MatrixEditor({
  m,
  onChange,
  title,
  rowLabel = 'row',
}: {
  m: Matrix;
  onChange: (m: Matrix) => void;
  title: string;
  rowLabel?: string;
}) {
  return (
    <fieldset className="min-w-0">
      <legend className="mb-1 text-[11.5px] text-muted">{title}</legend>
      <div className="overflow-x-auto scrollbar-thin">
        <div
          className="inline-grid gap-[2px]"
          style={{ gridTemplateColumns: `repeat(${m[0]?.length ?? 1}, 3.6rem)` }}
        >
          {m.map((row, i) =>
            row.map((v, j) => (
              <input
                key={`${i}-${j}`}
                type="number"
                step={0.1}
                value={v}
                aria-label={`${title}, ${rowLabel} ${i}, column ${j}`}
                onChange={(e) => {
                  const x = Number(e.target.value);
                  if (!Number.isFinite(x)) return;
                  onChange(m.map((r, a) => r.map((c, b) => (a === i && b === j ? x : c))));
                }}
                className="h-7 rounded-[4px] border border-line bg-elevated px-1 text-right font-mono text-[11px] text-fg tabular"
              />
            )),
          )}
        </div>
      </div>
    </fieldset>
  );
}

export function QkvLab() {
  const [tokens, setTokens] = useState(['The', 'cat', 'sat', 'down']);
  const [tokenDraft, setTokenDraft] = useState('The cat sat down');
  const [d, setD] = useState(4);
  const [dk, setDk] = useState(2);
  const [heads, setHeads] = useState(1);
  const [seed, setSeed] = useState(7);
  const [temperature, setTemperature] = useState(1);
  const [useScaling, setUseScaling] = useState(true);
  const [causal, setCausal] = useState(true);
  const [paddingState, setPadding] = useState<boolean[]>([]);
  const T = tokens.length;
  const padding = useMemo(
    () => Array.from({ length: T }, (_, i) => paddingState[i] ?? false),
    [paddingState, T],
  );
  // Edits apply to one configuration; changing dimensions or seed starts from fresh seeded weights.
  const cfgKey = `${seed}:${T}:${d}:${dk}:${heads}`;
  const [edited, setEdited] = useState<{ key: string; w: ToyW } | null>(null);
  const fresh = useMemo(() => initToyWeights(seed, T, d, dk, heads), [seed, T, d, dk, heads]);
  const weights = edited?.key === cfgKey ? edited.w : fresh;
  const setWeights = (w: ToyW) => setEdited({ key: cfgKey, w });
  const [viewHeadState, setViewHead] = useState(0);
  const viewHead = Math.min(viewHeadState, heads - 1);
  const [queryState, setQuery] = useState(2);
  const [keyState, setKey] = useState(1);
  const query = Math.min(queryState, T - 1);
  const key = Math.min(keyState, T - 1);
  const [step, setStep] = useState<number | null>(null);
  const reduced = useReducedMotion();

  const result = useMemo(() => {
    try {
      return runToyAttention(weights, { temperature, useScaling, causal, padding });
    } catch (e) {
      return e instanceof Error ? e : new Error(String(e));
    }
  }, [weights, temperature, useScaling, causal, padding]);

  // dot-product animation
  useEffect(() => {
    if (step === null) return;
    if (step >= dk) {
      const t = setTimeout(() => setStep(null), 1400);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => setStep(step + 1), 700);
    return () => clearTimeout(t);
  }, [step, dk]);

  if (result instanceof Error) return <Callout tone="danger">{result.message}</Callout>;
  const h = result.heads[viewHead]!;
  const qLabels = tokens.map((t, i) => `${i}:${t}`);
  const dimLabels = (n: number, p: string) => Array.from({ length: n }, (_, i) => `${p}${i}`);
  const qi = h.Q[query]!;
  const kj = h.K[key]!;
  const visibleTerms = step === null ? dk : Math.min(step, dk);
  const partial = qi.slice(0, visibleTerms).reduce((s, v, c) => s + v * kj[c]!, 0);
  const allowedRow = h.allowed[query]!;
  const exps = h.scaledScores[query]!.map((s, j) => (allowedRow[j] ? Math.exp(s - h.rowMax[query]!) : 0));

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="max-w-2xl">
          <h2 className="text-[15px] font-semibold text-fg">
            Q/K/V calculator — a real, tiny self-attention layer
          </h2>
          <p className="text-[12.5px] leading-relaxed text-muted">
            Every number below is recomputed from your inputs. Toy embeddings and weights only: this is an
            educational module, <strong className="text-fg">not a pretrained language model</strong>. Verified
            against an independent NumPy implementation in the test-suite.
          </p>
        </div>
        <Badge tone="accent">educational · toy weights</Badge>
      </div>

      <div className="flex flex-wrap items-center gap-x-10">
        <Tex block>{String.raw`Q = XW_Q,\; K = XW_K,\; V = XW_V`}</Tex>
        <Tex
          block
        >{String.raw`A = \operatorname{softmax}\!\Big(\frac{QK^\top}{\tau\sqrt{d_k}} + M\Big),\quad O = AV`}</Tex>
        <Tex block>{String.raw`Y = [O_1\,\|\,\dots\,\|\,O_H]\,W_O`}</Tex>
      </div>

      <Panel title="Configuration" id="qkv-config">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <Field label="Tokens (space-separated, 2–6)" htmlFor="qkv-tokens">
            <input
              id="qkv-tokens"
              value={tokenDraft}
              onChange={(e) => {
                setTokenDraft(e.target.value);
                const t = e.target.value.trim().split(/\s+/).filter(Boolean).slice(0, 6);
                if (t.length >= 2) setTokens(t);
              }}
              className="h-9 w-full rounded-lg border border-line bg-elevated px-3 text-[13px] text-fg"
            />
          </Field>
          <Field label="d_model" htmlFor="qkv-d">
            <Stepper id="qkv-d" label="d_model" value={d} min={2} max={6} onChange={setD} />
          </Field>
          <Field label="d_k = d_v (per head)" htmlFor="qkv-dk">
            <Stepper id="qkv-dk" label="d_k" value={dk} min={1} max={4} onChange={setDk} />
          </Field>
          <Field label="Heads" htmlFor="qkv-h">
            <Stepper id="qkv-h" label="Heads" value={heads} min={1} max={2} onChange={setHeads} />
          </Field>
          <Field
            label={`Temperature τ = ${temperature.toFixed(2)}`}
            htmlFor="qkv-temp"
            hint="Divides the scores; τ < 1 sharpens, τ > 1 flattens."
          >
            <input
              id="qkv-temp"
              type="range"
              min={0.25}
              max={4}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-full accent-[var(--accent)]"
            />
          </Field>
          <div className="space-y-3">
            <Switch id="qkv-scale" label="Scale by 1/√d_k" checked={useScaling} onChange={setUseScaling} />
            <Switch id="qkv-causal" label="Causal mask" checked={causal} onChange={setCausal} />
          </div>
          <Field
            label="Padding mask (masked keys)"
            hint="Padded keys receive −∞ for every query. A row with no visible key is undefined; it outputs zeros and is flagged."
          >
            <div className="flex flex-wrap gap-1.5">
              {tokens.map((t, i) => (
                <button
                  key={i}
                  type="button"
                  aria-pressed={padding[i]}
                  onClick={() => setPadding(padding.map((v, j) => (j === i ? !v : v)))}
                  className={cn(
                    'rounded-md border px-2 py-1 font-mono text-[11.5px]',
                    padding[i] ? 'hatch border-warn text-warn' : 'border-line bg-elevated text-fg',
                  )}
                >
                  {t}
                </button>
              ))}
            </div>
          </Field>
          <Field label={`Seed ${seed}`} hint="Same seed → same toy weights on every device.">
            <div className="flex gap-2">
              <Button size="sm" onClick={() => setSeed((s) => (s * 1103515245 + 12345) % 2147483647)}>
                <Dices aria-hidden className="size-3.5" /> New weights
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setEdited(null)}>
                <RotateCcw aria-hidden className="size-3.5" /> Reset edits
              </Button>
            </div>
          </Field>
        </div>
      </Panel>

      <Panel title="1 · Inputs and projection weights (editable)" id="qkv-inputs">
        <div className="flex flex-wrap gap-6">
          <MatrixEditor
            title={`X (T×d = ${T}×${d})`}
            rowLabel="token"
            m={weights.X}
            onChange={(X) => setWeights({ ...weights, X })}
          />
          {heads > 1 && (
            <Segmented<string>
              label="Head shown"
              value={String(viewHead)}
              onChange={(v) => setViewHead(Number(v))}
              options={Array.from({ length: heads }, (_, i) => ({ value: String(i), label: `Head ${i}` }))}
              size="sm"
            />
          )}
          {(['WQ', 'WK', 'WV'] as const).map((k) => (
            <MatrixEditor
              key={k}
              title={`${k.replace('W', 'W_')}^${viewHead} (${d}×${dk})`}
              m={weights[k][viewHead]!}
              onChange={(m) =>
                setWeights({ ...weights, [k]: weights[k].map((w, i) => (i === viewHead ? m : w)) })
              }
            />
          ))}
        </div>
      </Panel>

      <Panel title={`2 · Projections for head ${viewHead}`} id="qkv-proj">
        <div className="flex flex-wrap gap-6">
          <MatrixView
            title={`Q = X W_Q  (${T}×${dk})`}
            m={h.Q}
            rows={qLabels}
            cols={dimLabels(dk, 'd')}
            highlightRow={query}
          />
          <MatrixView
            title={`K = X W_K  (${T}×${dk})`}
            m={h.K}
            rows={qLabels}
            cols={dimLabels(dk, 'd')}
            highlightRow={key}
          />
          <MatrixView title={`V = X W_V  (${T}×${dk})`} m={h.V} rows={qLabels} cols={dimLabels(dk, 'd')} />
        </div>
      </Panel>

      <Panel title="3 · Scores, scaling, mask and softmax" id="qkv-scores">
        <p className="mb-3 text-[12.5px] text-muted">
          Click any score cell to choose the query (row) and key (column) used in the worked example below.
          Scale factor s = {useScaling ? `1/(τ·√${dk})` : '1/τ'} ={' '}
          <span className="font-mono text-fg">{fmt(h.scale, 4)}</span>.
        </p>
        <div className="flex flex-wrap gap-6">
          <MatrixView
            title="QKᵀ (raw)"
            m={h.rawScores}
            rows={qLabels}
            cols={tokens}
            highlightCell={[query, key]}
            onCell={(i, j) => {
              setQuery(i);
              setKey(j);
            }}
          />
          <MatrixView
            title="QKᵀ · s + M (masked)"
            m={h.maskedScores}
            rows={qLabels}
            cols={tokens}
            masked={h.allowed}
            highlightCell={[query, key]}
          />
          <MatrixView
            title="A = softmax(rows)"
            m={h.attention}
            rows={qLabels}
            cols={tokens}
            heat
            masked={h.allowed}
            highlightCell={[query, key]}
            digits={3}
          />
          <MatrixView
            title="row sums"
            m={h.attention.map((r) => [r.reduce((a, b) => a + b, 0)])}
            digits={6}
          />
        </div>
        {h.fullyMaskedRows.length > 0 && (
          <Callout tone="warn" className="mt-3" title="Fully masked rows">
            Row(s) {h.fullyMaskedRows.join(', ')} have no visible key (e.g. the first token is padding under a
            causal mask). Softmax over an empty set is undefined; this module outputs zeros for those rows.
            Real implementations differ (NaN, uniform, or a large finite negative instead of −∞), which is why
            padding is usually placed on the right.
          </Callout>
        )}
      </Panel>

      <Panel
        title={`4 · Worked example: query ${query} (“${tokens[query]}”) → key ${key} (“${tokens[key]}”)`}
        id="qkv-worked"
      >
        <div className="grid gap-6 xl:grid-cols-2">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <h3 className="text-[13px] font-semibold text-fg">Dot product q·k</h3>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setStep(step === null ? (reduced ? dk : 0) : null)}
                aria-label={step === null ? 'Animate the dot product' : 'Stop animation'}
              >
                {step === null ? (
                  <Play aria-hidden className="size-3.5" />
                ) : (
                  <Pause aria-hidden className="size-3.5" />
                )}
                {step === null ? 'Step through' : 'Stop'}
              </Button>
            </div>
            <ol className="space-y-1 font-mono text-[12px]" aria-live="polite">
              {qi.map((v, c) => (
                <li
                  key={c}
                  className={cn(
                    'flex gap-2 rounded px-1.5 py-0.5 transition-colors',
                    c < visibleTerms ? 'text-fg' : 'text-muted/40',
                    step !== null && c === visibleTerms - 1 && 'bg-accent-soft',
                  )}
                >
                  <span className="text-muted">d{c}:</span>
                  <span>
                    {fmt(v, 2)} × {fmt(kj[c]!, 2)} = {fmt(v * kj[c]!, 4)}
                  </span>
                </li>
              ))}
            </ol>
            <p className="font-mono text-[12.5px] text-fg">
              q·k = {fmt(partial, 4)}
              {visibleTerms === dk && (
                <>
                  {' '}
                  → × s = {fmt(h.scaledScores[query]![key]!, 4)}
                  {!allowedRow[key] && ' → masked (−∞)'}
                </>
              )}
            </p>
          </div>
          <div className="space-y-2">
            <h3 className="text-[13px] font-semibold text-fg">Softmax of row {query}</h3>
            <Tex>{String.raw`a_{ij} = \frac{e^{s_{ij} - m_i}}{\sum_{j'} e^{s_{ij'} - m_i}},\quad m_i = \max_{j\ \text{visible}} s_{ij}`}</Tex>
            <table className="w-full font-mono text-[11.5px]">
              <thead className="text-muted">
                <tr>
                  <th scope="col" className="text-left font-normal">
                    key
                  </th>
                  <th scope="col" className="text-right font-normal">
                    s_ij
                  </th>
                  <th scope="col" className="text-right font-normal">
                    e^(s−m)
                  </th>
                  <th scope="col" className="text-right font-normal">
                    a_ij
                  </th>
                </tr>
              </thead>
              <tbody>
                {tokens.map((t, j) => (
                  <tr key={j} className={cn('border-t border-line', j === key && 'bg-teal-soft')}>
                    <td className="py-0.5">
                      {j}:{t}
                    </td>
                    <td className="text-right">
                      {allowedRow[j] ? fmt(h.scaledScores[query]![j]!, 3) : '−∞'}
                    </td>
                    <td className="text-right">{fmt(exps[j]!, 4)}</td>
                    <td className="text-right text-fg">{fmt(h.attention[query]![j]!, 4)}</td>
                  </tr>
                ))}
                <tr className="border-t border-line-strong text-muted">
                  <td>Σ</td>
                  <td />
                  <td className="text-right">{fmt(h.rowSumExp[query]!, 4)}</td>
                  <td className="text-right text-fg">
                    {fmt(
                      h.attention[query]!.reduce((a, b) => a + b, 0),
                      6,
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
            <h3 className="pt-2 text-[13px] font-semibold text-fg">Weighted value sum o = Σ a_ij v_j</h3>
            <p className="font-mono text-[11.5px] leading-relaxed text-muted">
              {tokens
                .map((_, j) =>
                  allowedRow[j]
                    ? `${fmt(h.attention[query]![j]!, 3)}·[${h.V[j]!.map((v) => fmt(v, 2)).join(', ')}]`
                    : null,
                )
                .filter(Boolean)
                .join(' + ')}{' '}
              = <span className="text-fg">[{h.output[query]!.map((v) => fmt(v, 3)).join(', ')}]</span>
            </p>
          </div>
        </div>
      </Panel>

      <Panel title="5 · Head outputs, concatenation and output projection" id="qkv-out">
        <div className="flex flex-wrap gap-6">
          {result.heads.map((hh, i) => (
            <MatrixView key={i} title={`O_${i} = A_${i} V_${i} (${T}×${dk})`} m={hh.output} rows={qLabels} />
          ))}
          <MatrixView title={`concat (${T}×${heads * dk})`} m={result.concat} rows={qLabels} />
          <MatrixEditor
            title={`W_O (${heads * dk}×${d})`}
            m={weights.WO}
            onChange={(WO) => setWeights({ ...weights, WO })}
          />
          <MatrixView title={`Y = concat · W_O (${T}×${d})`} m={result.projected} rows={qLabels} />
        </div>
        <p className="mt-3 text-[12px] text-muted">
          In a full Transformer, Y is added to the residual stream. Real models also apply LayerNorm before
          attention and may share K/V across heads (grouped-query or multi-query attention); this module uses
          one K/V per head and no dropout.
        </p>
      </Panel>
    </div>
  );
}
