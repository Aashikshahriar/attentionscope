import { Link } from 'react-router-dom';
import type { AttentionRun } from '@/types/contracts';
import { attentionRow, headScores } from '@/core/attention/tensor';
import { normalizedRowEntropy, rowEntropy, rowMeanDistance } from '@/core/metrics/headMetrics';
import { Badge, Panel, Stat, TokenText, cn } from '@/components/ui/primitives';
import { fmt, pct } from '@/utils/hooks';
import { useWorkspace } from '@/stores/workspace';

export function Inspector({ run }: { run: AttentionRun }) {
  const { layer, head, queryIndex, keyIndex, valueMode, setView } = useWorkspace();
  const T = run.attention.shape.queryLength;
  const tok = run.input.tokens[queryIndex]!;
  const row = attentionRow(run.attention, layer, head, queryIndex);
  const scores = headScores(run.attention, layer, head);
  const ranked = Array.from({ length: queryIndex + 1 }, (_, j) => j).sort((a, b) => row[b]! - row[a]!);
  const top = ranked.slice(0, 8);
  const sum = Array.from(row).reduce((a, b) => a + b, 0);
  // row-level metrics operate on a single row embedded at index queryIndex
  const rowMatrix = new Float64Array(T * T);
  rowMatrix.set(row, queryIndex * T);
  const H = rowEntropy(rowMatrix, T, queryIndex);
  const Hn = normalizedRowEntropy(rowMatrix, T, queryIndex);
  const dist = rowMeanDistance(rowMatrix, T, queryIndex);
  const pred = run.predictions?.perPosition?.[queryIndex];
  const a = run.model.architecture;

  return (
    <div className="space-y-3 lg:grid lg:grid-cols-2 lg:items-start lg:gap-3 lg:space-y-0 2xl:block 2xl:space-y-3">
      <Panel title="Selected query token" id="insp-token">
        <div className="space-y-3">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <div className="text-lg leading-tight text-fg">
                <TokenText text={tok.text} />
              </div>
              <div className="mt-1 font-mono text-[11.5px] text-muted">
                position {queryIndex} · id {tok.id} · piece <span className="bidi-isolate">{tok.piece}</span>
              </div>
            </div>
            <div className="flex flex-col items-end gap-1">
              {tok.special && <Badge tone="teal">special</Badge>}
              {tok.partialCharacter && <Badge tone="warn">partial char</Badge>}
            </div>
          </div>
          {pred && (
            <div>
              <div className="mb-1 text-[11px] uppercase tracking-wide text-muted">
                Model's next-token prediction here
              </div>
              <ol className="space-y-0.5">
                {pred.top.map((p) => (
                  <li key={p.id} className="flex items-center justify-between gap-2 text-[12.5px]">
                    <TokenText text={p.text} className="truncate text-fg" />
                    <span className="font-mono tabular text-muted">{pct(p.prob)}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      </Panel>

      <Panel
        title={
          <span>
            Attention row · L{layer} H{head}
          </span>
        }
        id="insp-row"
      >
        <p className="mb-2 text-[12px] text-muted">
          Where position {queryIndex} attends (post-softmax, top {top.length} of {queryIndex + 1} visible
          keys).
        </p>
        <ol className="space-y-1">
          {top.map((j) => (
            <li key={j}>
              <button
                type="button"
                onClick={() => setView({ keyIndex: j })}
                className={cn(
                  'grid w-full grid-cols-[2rem_minmax(0,1fr)_4.5rem] items-center gap-2 rounded px-1 py-0.5 text-left text-[12.5px] hover:bg-elevated',
                  keyIndex === j && 'bg-teal-soft',
                )}
              >
                <span className="font-mono text-[11px] text-muted">{j}</span>
                <span className="flex min-w-0 items-center gap-2">
                  <TokenText text={run.input.tokens[j]!.text} className="truncate text-fg" />
                  <span aria-hidden className="h-1 min-w-4 flex-1 overflow-hidden rounded bg-elevated">
                    <span className="block h-full bg-accent" style={{ width: `${row[j]! * 100}%` }} />
                  </span>
                </span>
                <span className="text-right font-mono tabular text-fg">{fmt(row[j]!, 4)}</span>
              </button>
            </li>
          ))}
        </ol>
        {keyIndex !== null && keyIndex <= queryIndex && (
          <div className="mt-3 rounded-lg border border-line bg-elevated p-2.5 text-[12px]">
            <div className="text-muted">
              Selected cell q{queryIndex} → k{keyIndex} <TokenText text={run.input.tokens[keyIndex]!.text} />
            </div>
            <dl className="mt-1 grid grid-cols-2 gap-2">
              <Stat label="probability" value={fmt(row[keyIndex]!, 6)} />
              <Stat
                label="pre-softmax score"
                value={scores ? fmt(scores[queryIndex * T + keyIndex]!, 4) : 'not stored'}
              />
            </dl>
          </div>
        )}
        {keyIndex !== null && keyIndex > queryIndex && (
          <p className="mt-3 text-[12px] text-muted">
            Key {keyIndex} is after the query: masked by the causal mask (score −∞, probability exactly 0).
          </p>
        )}
        <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-2 border-t border-line pt-3">
          <Stat label="entropy (nats)" value={fmt(H, 3)} />
          <Stat label="normalized" value={Number.isNaN(Hn) ? 'n/a (1 key)' : fmt(Hn, 3)} />
          <Stat label="mean distance" value={`${fmt(dist, 2)} tok`} />
          <Stat label="row sum" value={fmt(sum, 6)} />
        </dl>
      </Panel>

      <Panel title="Tensor & model" id="insp-tensor">
        <dl className="grid grid-cols-2 gap-x-3 gap-y-2">
          <Stat
            label="attention shape"
            value={`[${run.attention.shape.batch}, ${run.attention.shape.layers}, ${run.attention.shape.heads}, ${T}, ${T}]`}
          />
          <Stat label="layout" value="batch·layer·head·q·k" />
          <Stat label="attention" value={`${a.attentionType}, ${a.numKeyValueHeads} KV heads`} mono={false} />
          <Stat label="d_head · scale" value={`${a.headDim} · ${a.attentionScale}`} />
          <Stat label="positional" value={a.positional} mono={false} />
          <Stat
            label="values shown"
            value={valueMode === 'scores' ? 'pre-softmax' : 'post-softmax'}
            mono={false}
          />
        </dl>
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          {run.provenance.precomputed ? (
            <Badge tone="warn">precomputed artifact</Badge>
          ) : (
            <Badge tone="teal">live in-browser run</Badge>
          )}
          <Badge>{run.model.license}</Badge>
          <Badge>{run.provenance.storedPrecision}</Badge>
        </div>
        <p className="mt-2 break-words font-mono text-[10.5px] leading-relaxed text-muted">
          {run.model.repo}@{run.model.revision.slice(0, 12)}
        </p>
      </Panel>

      <Panel title="Interpretation caveats" id="insp-caveats">
        <ul className="list-disc space-y-1.5 pl-4 text-[12.5px] leading-relaxed text-muted">
          <li>
            Attention weights show where information is read from, not why the model predicts what it does.
          </li>
          <li>
            A high weight does not prove that token caused the prediction; use the intervention tools to test
            effects.
          </li>
          <li>Heavy attention to the first token is often an “attention sink” used as a no-op.</li>
        </ul>
        <Link to="/docs/methodology" className="mt-2 inline-block text-[12.5px] text-accent underline">
          Methodology &amp; limits
        </Link>
      </Panel>
    </div>
  );
}
