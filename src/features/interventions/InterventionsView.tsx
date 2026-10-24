import { useMemo, useState } from 'react';
import { FlaskConical, Play, Square, Grid3x3 } from 'lucide-react';
import type { AttentionRun, ComparisonResult, InterventionKind, Prediction } from '@/types/contracts';
import { INTERVENTIONS, interventionDefinition } from '@/core/model-adapters/interventions';
import { useWorkspace } from '@/stores/workspace';
import { modelWorker } from '@/services/workers';
import {
  Badge,
  Button,
  Callout,
  Field,
  Panel,
  ProgressBar,
  Stat,
  TokenText,
  cn,
} from '@/components/ui/primitives';
import { Select, Stepper } from '@/components/ui/controls';
import { divergingColor, readableOn } from '@/features/heatmap/colormaps';
import { fmt, pct } from '@/utils/hooks';
import type { SweepResult } from '@/workers/modelApi';

function EffectGrid({
  values,
  selected,
  onSelect,
  format,
  label,
  legend,
}: {
  values: number[][];
  selected: [number, number] | null;
  onSelect: (l: number, h: number) => void;
  format: (v: number) => string;
  label: (l: number, h: number, v: number) => string;
  legend: [string, string];
}) {
  const maxAbs = Math.max(
    1e-12,
    ...values
      .flat()
      .map((v) => Math.abs(v))
      .filter(Number.isFinite),
  );
  const H = values[0]?.length ?? 0;
  return (
    <div className="space-y-2">
      <div className="overflow-x-auto scrollbar-thin">
        <div
          className="inline-grid gap-[2px]"
          style={{ gridTemplateColumns: `2.25rem repeat(${H}, minmax(2.6rem, 1fr))` }}
        >
          <span />
          {Array.from({ length: H }, (_, h) => (
            <span key={h} className="text-center font-mono text-[10px] text-muted">
              H{h}
            </span>
          ))}
          {values.map((row, l) => (
            <div key={l} className="contents">
              <span className="self-center font-mono text-[10px] text-muted">L{l}</span>
              {row.map((v, h) => {
                const sel = selected?.[0] === l && selected[1] === h;
                const bg = divergingColor(v, maxAbs);
                const text = format(v);
                return (
                  <button
                    key={h}
                    type="button"
                    onClick={() => onSelect(l, h)}
                    aria-label={`${text} — ${label(l, h, v)}`}
                    aria-pressed={sel}
                    className={cn(
                      'h-8 rounded-[4px] font-mono text-[10px] tabular',
                      sel && 'ring-2 ring-accent ring-offset-1 ring-offset-[var(--panel)]',
                    )}
                    style={{ background: bg, color: readableOn(bg) }}
                  >
                    {text}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-2 text-[11px] text-muted">
        <span>{legend[0]}</span>
        <span
          aria-hidden
          className="h-2 w-40 rounded-sm"
          style={{
            background: `linear-gradient(90deg, ${divergingColor(-1, 1)}, ${divergingColor(0, 1)}, ${divergingColor(1, 1)})`,
          }}
        />
        <span>{legend[1]}</span>
        <span className="ml-2">scale ±{fmt(maxAbs, 3)}</span>
      </div>
    </div>
  );
}

function PrecomputedAblation({ run }: { run: AttentionRun }) {
  const ab = run.ablation!;
  const { setView, layer, head } = useWorkspace();
  const [sel, setSel] = useState<[number, number]>([layer, head]);
  const delta = ab.heads.map((row) => row.map((c) => c.targetProb - ab.baseline.targetProb));
  const cell = ab.heads[sel[0]]?.[sel[1]];
  const ranked = ab.heads
    .flatMap((row, l) =>
      row.map((c, h) => ({ l, h, d: c.targetProb - ab.baseline.targetProb, kl: c.klFromBaseline })),
    )
    .sort((a, b) => a.d - b.d)
    .slice(0, 6);
  return (
    <div className="space-y-4">
      <Callout tone="warn" title="Precomputed offline — not executed in your browser">
        These results come from {ab.heads.length * (ab.heads[0]?.length ?? 0)} real forward passes of{' '}
        {run.model.displayName} run by{' '}
        <code className="font-mono text-[11.5px]">{run.provenance.generator.split(' ')[0]}</code>. For
        interventions you can run yourself, switch to the live Pythia-14m model.
      </Callout>
      <dl className="grid gap-3 sm:grid-cols-3">
        <Stat label="intervention" value="zero head output (pre-W_O)" mono={false} />
        <Stat
          label="measured at"
          value={`position ${ab.measuredAt.position}, target ${JSON.stringify(ab.measuredAt.target.text)}`}
          mono={false}
        />
        <Stat label="baseline P(target)" value={pct(ab.baseline.targetProb, 2)} />
      </dl>
      <EffectGrid
        values={delta}
        selected={sel}
        onSelect={(l, h) => {
          setSel([l, h]);
          setView({ layer: l, head: h });
        }}
        format={(v) => (Math.abs(v) < 0.0005 ? '0.0' : (v * 100).toFixed(1))}
        label={(l, h, v) =>
          `Layer ${l} head ${h}: change in target probability ${(v * 100).toFixed(2)} percentage points`
        }
        legend={['P(target) falls', 'rises']}
      />
      <p className="text-[11.5px] text-muted">
        Cells show the change in P(target) in percentage points after zeroing that head.
      </p>
      <div className="grid gap-4 lg:grid-cols-2">
        {cell && (
          <Panel title={`Head ${sel[0]}.${sel[1]} ablated`} id="abl-cell">
            <dl className="grid grid-cols-2 gap-3">
              <Stat label="P(target) after" value={pct(cell.targetProb, 2)} />
              <Stat
                label="Δ P(target)"
                value={`${((cell.targetProb - ab.baseline.targetProb) * 100).toFixed(2)} pp`}
              />
              <Stat
                label="log P(target)"
                value={`${fmt(Math.log(ab.baseline.targetProb), 3)} → ${fmt(Math.log(cell.targetProb), 3)}`}
              />
              <Stat label="KL(base ‖ ablated)" value={`${fmt(cell.klFromBaseline, 4)} nats`} />
              <Stat label="new top-1" value={<TokenText text={cell.top1.text} />} />
              <Stat label="top-1 prob" value={pct(cell.top1.prob, 2)} />
            </dl>
            <p className="mt-3 text-[11.5px] text-muted">
              Raw logits are not compared: GPT-2&apos;s logits share a large input-dependent offset (softmax
              is shift-invariant), so they are reported only in exports.
            </p>
          </Panel>
        )}
        <Panel title="Largest drops in P(target)" id="abl-rank">
          <ol className="space-y-1 text-[12.5px]">
            {ranked.map((r) => (
              <li key={`${r.l}.${r.h}`}>
                <button
                  type="button"
                  className="flex w-full justify-between rounded px-1 hover:bg-elevated"
                  onClick={() => setSel([r.l, r.h])}
                >
                  <span className="font-mono">
                    L{r.l} H{r.h}
                  </span>
                  <span className="font-mono tabular text-muted">
                    {(r.d * 100).toFixed(2)} pp · KL {fmt(r.kl, 3)}
                  </span>
                </button>
              </li>
            ))}
          </ol>
        </Panel>
      </div>
      <Callout title="What this does and does not show">
        {ab.caveat} A large effect shows the head matters for this output under this intervention; it does not
        identify what the head computes.
      </Callout>
    </div>
  );
}

function ComparisonTable({ c }: { c: ComparisonResult }) {
  const union = useMemo(() => {
    const ids = new Map<number, string>();
    [...c.baselineTop, ...c.intervenedTop].forEach((p) => ids.set(p.id, p.text));
    const b = new Map(c.baselineTop.map((p) => [p.id, p.prob]));
    const i = new Map(c.intervenedTop.map((p) => [p.id, p.prob]));
    return [...ids.entries()]
      .map(([id, text]) => ({ id, text, b: b.get(id), i: i.get(id) }))
      .sort((x, y) => (y.b ?? 0) - (x.b ?? 0))
      .slice(0, 12);
  }, [c]);
  const max = Math.max(...union.flatMap((u) => [u.b ?? 0, u.i ?? 0]), 1e-9);
  return (
    <table className="w-full text-[12px]">
      <caption className="sr-only">Next-token probabilities before and after the intervention</caption>
      <thead className="text-muted">
        <tr>
          <th scope="col" className="py-1 pr-2 text-left font-medium">
            token
          </th>
          <th scope="col" className="py-1 pr-2 text-left font-medium">
            baseline vs intervened
          </th>
          <th scope="col" className="py-1 pr-2 text-right font-medium">
            before
          </th>
          <th scope="col" className="py-1 text-right font-medium">
            after
          </th>
        </tr>
      </thead>
      <tbody>
        {union.map((u) => (
          <tr key={u.id} className={cn('border-t border-line', u.id === c.target.id && 'bg-accent-soft')}>
            <td className="py-1 pr-2">
              <TokenText text={u.text} className="text-fg" />
            </td>
            <td className="w-1/2 py-1 pr-2">
              <div aria-hidden className="space-y-0.5">
                <div
                  className="h-1.5 rounded-sm bg-muted/50"
                  style={{ width: `${((u.b ?? 0) / max) * 100}%` }}
                />
                <div
                  className="h-1.5 rounded-sm bg-accent"
                  style={{ width: `${((u.i ?? 0) / max) * 100}%` }}
                />
              </div>
            </td>
            <td className="py-1 pr-2 text-right font-mono tabular text-muted">
              {u.b === undefined ? '<top-10' : pct(u.b, 2)}
            </td>
            <td className="py-1 text-right font-mono tabular text-fg">
              {u.i === undefined ? '<top-10' : pct(u.i, 2)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function LiveInterventions({ run }: { run: AttentionRun }) {
  const ws = useWorkspace();
  const {
    model,
    comparison,
    comparisonStatus,
    comparisonError,
    sweep,
    sweepProgress,
    sweepError,
    runComparison,
    runSweep,
    cancelSweep,
  } = ws;
  const [kind, setKind] = useState<InterventionKind>('zero-head-output');
  const [targetMode, setTargetMode] = useState<string>('top1');
  const [custom, setCustom] = useState('');
  const [customId, setCustomId] = useState<{ id: number; text: string } | null>(null);
  const [customErr, setCustomErr] = useState<string | null>(null);
  const shape = run.attention.shape;
  const top: Prediction[] = run.predictions?.top ?? [];
  const ready = model.state === 'ready';
  const stale =
    comparison && comparison.inputTokenIds.join(',') !== run.input.tokens.map((t) => t.id).join(',');

  const targetId =
    targetMode === 'top1' ? null : targetMode === 'custom' ? (customId?.id ?? null) : Number(targetMode);

  if (!ready) {
    return (
      <Callout tone="warn" title="Load the model to run interventions">
        Interventions re-run the forward pass, so the Pythia-14m weights must be loaded in this tab.
      </Callout>
    );
  }
  const def = interventionDefinition(kind);
  return (
    <div className="space-y-4">
      <Callout tone="success" title="Executed in your browser">
        Each intervention modifies the actual forward pass of Pythia-14m at the stated location and re-runs
        every downstream layer. Settings are deterministic: no dropout, no sampling, same input ids.
      </Callout>
      <div className="grid gap-3 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_minmax(0,1fr)]">
        <Field
          label="Intervention"
          htmlFor="iv-kind"
          hint={
            def.family === 'head-output-ablation' ? 'Head-output ablation' : 'Attention-matrix replacement'
          }
        >
          <Select<InterventionKind>
            id="iv-kind"
            label="Intervention"
            value={kind}
            onChange={setKind}
            options={INTERVENTIONS.map((i) => ({ value: i.kind, label: i.label }))}
          />
        </Field>
        <Field label="Layer" htmlFor="iv-layer">
          <Stepper
            id="iv-layer"
            label="Intervention layer"
            value={ws.layer}
            min={0}
            max={shape.layers - 1}
            onChange={(v) => ws.setView({ layer: v })}
          />
        </Field>
        <Field label="Head" htmlFor="iv-head">
          <Stepper
            id="iv-head"
            label="Intervention head"
            value={ws.head}
            min={0}
            max={shape.heads - 1}
            onChange={(v) => ws.setView({ head: v })}
          />
        </Field>
      </div>
      <p className="text-[12.5px] leading-relaxed text-muted">{def.description}</p>
      <div className="grid gap-3 md:grid-cols-2">
        <Field label="Target token (measured at the last position)" htmlFor="iv-target">
          <Select
            id="iv-target"
            label="Target token"
            value={targetMode}
            onChange={setTargetMode}
            options={[
              { value: 'top1', label: 'Baseline top-1 prediction' },
              ...top
                .slice(0, 10)
                .map((p) => ({ value: String(p.id), label: `${JSON.stringify(p.text)} (${pct(p.prob)})` })),
              { value: 'custom', label: 'Custom token…' },
            ]}
          />
        </Field>
        {targetMode === 'custom' && (
          <Field
            label="Custom target text (must be one token)"
            htmlFor="iv-custom"
            hint={
              customErr ??
              (customId
                ? `token id ${customId.id}`
                : 'Include a leading space for word-initial tokens, e.g. " Mary".')
            }
          >
            <div className="flex gap-2">
              <input
                id="iv-custom"
                value={custom}
                onChange={(e) => {
                  setCustom(e.target.value);
                  setCustomId(null);
                }}
                className="h-9 min-w-0 flex-1 rounded-lg border border-line bg-elevated px-3 font-mono text-[13px] text-fg"
              />
              <Button
                size="md"
                onClick={async () => {
                  const r = await modelWorker().call('resolveToken', { text: custom });
                  if (r.ids.length !== 1) {
                    setCustomErr(
                      `"${custom}" is ${r.ids.length} tokens; choose text that is exactly one token.`,
                    );
                    setCustomId(null);
                  } else {
                    setCustomErr(null);
                    setCustomId({ id: r.ids[0]!, text: custom });
                  }
                }}
              >
                Check
              </Button>
            </div>
          </Field>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        <Button
          variant="primary"
          disabled={comparisonStatus === 'running' || (targetMode === 'custom' && !customId)}
          onClick={() => void runComparison({ kind, layer: ws.layer, head: ws.head }, targetId)}
        >
          <Play aria-hidden className="size-4" />{' '}
          {comparisonStatus === 'running' ? 'Running…' : `Run on L${ws.layer} H${ws.head}`}
        </Button>
        {sweepProgress ? (
          <Button onClick={cancelSweep}>
            <Square aria-hidden className="size-3.5" /> Cancel sweep
          </Button>
        ) : (
          <Button
            disabled={targetMode === 'custom' && !customId}
            onClick={() => void runSweep(kind, targetId)}
          >
            <Grid3x3 aria-hidden className="size-4" /> Sweep all {shape.layers * shape.heads} heads
          </Button>
        )}
      </div>
      {sweepProgress && (
        <div className="max-w-md space-y-1" aria-live="polite">
          <ProgressBar value={sweepProgress.done / sweepProgress.total} label="Sweep progress" />
          <p className="font-mono text-[11.5px] text-muted">
            {sweepProgress.done} / {sweepProgress.total} forward passes
          </p>
        </div>
      )}
      {comparisonError && <Callout tone="danger">{comparisonError}</Callout>}
      {sweepError && <Callout tone="danger">{sweepError}</Callout>}
      {comparison && !stale && <ComparisonPanel c={comparison} />}
      {sweep && sweep.inputTokenIds.join(',') === run.input.tokens.map((t) => t.id).join(',') && (
        <SweepPanel s={sweep} />
      )}
    </div>
  );
}

function ComparisonPanel({ c }: { c: ComparisonResult }) {
  return (
    <Panel
      title={
        <span className="inline-flex items-center gap-2">
          <FlaskConical aria-hidden className="size-4 text-accent" /> Result ·{' '}
          {interventionDefinition(c.spec.kind).label} @ L{c.spec.layer} H{c.spec.head}
        </span>
      }
      id="iv-result"
    >
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="space-y-3">
          <dl className="grid grid-cols-2 gap-3">
            <Stat label="target" value={<TokenText text={c.target.text} />} />
            <Stat label="Δ log P(target)" value={fmt(c.target.deltaLogProb, 4)} />
            <Stat
              label="P(target)"
              value={`${pct(c.target.baselineProb, 2)} → ${pct(c.target.intervenedProb, 2)}`}
            />
            <Stat
              label="log P(target)"
              value={`${fmt(Math.log(c.target.baselineProb), 3)} → ${fmt(Math.log(c.target.intervenedProb), 3)}`}
            />
            <Stat label="KL(base ‖ intervened)" value={`${fmt(c.klDivergence, 5)} nats`} />
            <Stat label="compute" value={`${c.durationMs.toFixed(0)} ms`} />
          </dl>
          <div className="rounded-lg border border-line bg-elevated p-2.5 text-[11.5px] leading-relaxed text-muted">
            <div>
              <span className="text-fg">Location:</span> {c.location}
            </div>
            <div>
              <span className="text-fg">Model:</span> {c.model.id}@{c.model.revision.slice(0, 10)} · position{' '}
              {c.position}
            </div>
            {c.validation ? (
              <div className="mt-1 flex flex-wrap items-center gap-1.5">
                <Badge tone={c.validation.passed ? 'teal' : 'danger'}>
                  {c.validation.passed ? 'replacement valid' : 'invalid'}
                </Badge>
                rows sum to 1 (max error {c.validation.maxRowSumError.toExponential(1)}),{' '}
                {c.validation.maskViolations} mask violations, {c.validation.negativeEntries} negative
                entries, shape {c.validation.shape.join('×')}
              </div>
            ) : (
              <div className="mt-1">No replacement matrix (head-output ablation).</div>
            )}
          </div>
        </div>
        <div>
          <ComparisonTable c={c} />
          <p className="mt-1 text-[11px] text-muted">
            Grey bar = baseline, accent bar = after intervention. Highlighted row = target.
          </p>
        </div>
      </div>
      <p className="mt-3 text-[11.5px] text-muted">
        This measures the effect of this specific intervention on this input. It is evidence about the
        head&apos;s role here, not a general causal explanation.
      </p>
    </Panel>
  );
}

function SweepPanel({ s }: { s: SweepResult }) {
  const ws = useWorkspace();
  return (
    <Panel
      title={`Sweep · ${interventionDefinition(s.kind).label} · target ${JSON.stringify(s.target.text)}`}
      id="iv-sweep"
    >
      <p className="mb-3 text-[12px] text-muted">
        Δ log P(target) for each head (baseline P = {pct(s.target.baselineProb, 2)}).{' '}
        {s.deltaLogProb.flat().length} forward passes in {(s.durationMs / 1000).toFixed(1)} s. Click a cell to
        inspect that head.
      </p>
      <EffectGrid
        values={s.deltaLogProb}
        selected={[ws.layer, ws.head]}
        onSelect={(l, h) => ws.setView({ layer: l, head: h })}
        format={(v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2))}
        label={(l, h, v) => `Layer ${l} head ${h}: change in log probability ${v.toFixed(3)}`}
        legend={['log P falls', 'rises']}
      />
    </Panel>
  );
}

export function InterventionsView({ run }: { run: AttentionRun }) {
  if (run.source.kind === 'precomputed') {
    return run.ablation ? (
      <PrecomputedAblation run={run} />
    ) : (
      <Callout>No intervention data for this artifact.</Callout>
    );
  }
  return <LiveInterventions run={run} />;
}
