import type { TokenRecord } from '@/types/contracts';
import { TokenText, cn } from '@/components/ui/primitives';
import { fmt } from '@/utils/hooks';

/** Accessible table alternative to the heatmap: every value, with row/column headers. */
export function AttentionTable({
  values,
  T,
  tokens,
  mode,
  caption,
  selectedQuery,
  onSelectQuery,
}: {
  values: ArrayLike<number>;
  T: number;
  tokens: TokenRecord[];
  mode: 'probabilities' | 'scores';
  caption: string;
  selectedQuery: number;
  onSelectQuery: (q: number) => void;
}) {
  return (
    <div className="max-h-[60vh] overflow-auto rounded-lg border border-line scrollbar-thin">
      <table className="border-collapse text-[11.5px]">
        <caption className="sr-only">{caption}</caption>
        <thead className="sticky top-0 z-10 bg-panel">
          <tr>
            <th
              scope="col"
              className="sticky left-0 z-20 border-b border-r border-line bg-panel px-2 py-1 text-left text-muted"
            >
              query \ key
            </th>
            {tokens.map((t, j) => (
              <th
                key={j}
                scope="col"
                className="whitespace-nowrap border-b border-line px-2 py-1 text-left font-normal text-muted"
              >
                <span className="font-mono">{j}</span> <TokenText text={t.text} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {tokens.map((t, i) => (
            <tr key={i} className={cn(i === selectedQuery && 'bg-accent-soft')}>
              <th
                scope="row"
                className="sticky left-0 whitespace-nowrap border-r border-line bg-panel px-2 py-1 text-left font-normal"
              >
                <button
                  type="button"
                  className="text-left text-fg hover:text-accent"
                  onClick={() => onSelectQuery(i)}
                >
                  <span className="font-mono text-muted">{i}</span> <TokenText text={t.text} />
                </button>
              </th>
              {Array.from({ length: T }, (_, j) =>
                j > i ? (
                  <td key={j} className="px-2 py-1 text-right text-muted/60" aria-label="masked">
                    —
                  </td>
                ) : (
                  <td key={j} className="px-2 py-1 text-right font-mono tabular text-fg">
                    {fmt(values[i * T + j]!, mode === 'probabilities' ? 4 : 3)}
                  </td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
