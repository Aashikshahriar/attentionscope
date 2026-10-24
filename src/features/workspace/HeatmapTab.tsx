import { useState } from 'react';
import type { AttentionRun } from '@/types/contracts';
import { headMatrix, headScores } from '@/core/attention/tensor';
import { useWorkspace } from '@/stores/workspace';
import { Heatmap } from '@/features/heatmap/Heatmap';
import { AttentionTable } from '@/features/heatmap/AttentionTable';
import { Switch } from '@/components/ui/controls';
import { Badge } from '@/components/ui/primitives';

export function HeatmapTab({ run }: { run: AttentionRun }) {
  const { layer, head, queryIndex, keyIndex, setView, valueMode } = useWorkspace();
  const [table, setTable] = useState(false);
  const T = run.attention.shape.queryLength;
  const scores = headScores(run.attention, layer, head);
  const mode = valueMode === 'scores' && scores ? 'scores' : 'probabilities';
  const values = mode === 'scores' ? scores! : headMatrix(run.attention, layer, head);
  const caption = `${mode === 'scores' ? 'Pre-softmax scores' : 'Attention probabilities'} for ${run.model.displayName}, layer ${layer}, head ${head}`;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-[15px] font-semibold text-fg">
            Layer {layer} · Head {head}
          </h2>
          <p className="text-[12.5px] text-muted">
            {mode === 'scores' ? (
              <>
                Scaled dot-product scores q·k/√d<sub>head</sub> before masking and softmax.
              </>
            ) : (
              <>Post-softmax attention probabilities A = softmax(QKᵀ/√d + M). Each row sums to 1.</>
            )}
          </p>
        </div>
        <div className="flex items-center gap-4">
          {run.provenance.precomputed ? (
            <Badge tone="warn">precomputed</Badge>
          ) : (
            <Badge tone="teal">live</Badge>
          )}
          <div className="w-36">
            <Switch id="table-view" label="Table view" checked={table} onChange={setTable} />
          </div>
        </div>
      </div>
      {table ? (
        <AttentionTable
          values={values}
          T={T}
          tokens={run.input.tokens}
          mode={mode}
          caption={caption}
          selectedQuery={queryIndex}
          onSelectQuery={(q) => setView({ queryIndex: q })}
        />
      ) : (
        <Heatmap
          values={values}
          T={T}
          tokens={run.input.tokens}
          mode={mode}
          selectedQuery={queryIndex}
          selectedKey={keyIndex}
          onSelect={(q, k) => setView({ queryIndex: q, keyIndex: k })}
          label={caption}
        />
      )}
    </div>
  );
}
