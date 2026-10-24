import { useEffect, useState } from 'react';
import type { AttentionRun } from '@/types/contracts';
import { fetchDemoManifest, loadDemoRun } from '@/services/runs';
import { headMatrix } from '@/core/attention/tensor';
import { useReducedMotion } from '@/utils/hooks';

const LAYER = 5;
const HEAD = 5;
const STEP = 50;

/** Lightweight arc diagram of real, precomputed GPT-2 attention (induction head 5.5). */
export function HeroAttention() {
  const [run, setRun] = useState<AttentionRun | null>(null);
  const [failed, setFailed] = useState(false);
  const [q, setQ] = useState(18);
  const reduced = useReducedMotion();

  useEffect(() => {
    let cancelled = false;
    fetchDemoManifest()
      .then((m) => {
        const e = m.demos.find((d) => d.id === 'induction');
        if (!e) throw new Error('missing demo');
        return loadDemoRun(e);
      })
      .then((r) => !cancelled && setRun(r.run))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, []);

  const T = run?.attention.shape.queryLength ?? 0;
  useEffect(() => {
    if (!run || reduced) return;
    const start = 11;
    const id = setInterval(() => setQ((x) => (x + 1 >= T ? start : Math.max(start, x + 1))), 1500);
    return () => clearInterval(id);
  }, [run, reduced, T]);

  if (failed) return null;
  const W = Math.max(1, T) * STEP;
  const baseY = 200;
  const m = run ? headMatrix(run.attention, LAYER, HEAD) : null;
  const x = (i: number) => i * STEP + STEP / 2;

  return (
    <figure className="relative overflow-hidden rounded-2xl border border-line bg-panel/70 p-4 shadow-[0_0_80px_-30px_var(--accent)]">
      {!run ? (
        <div className="h-[200px] animate-pulse-soft rounded-xl bg-elevated" aria-hidden />
      ) : (
        <svg
          viewBox={`0 0 ${W} 215`}
          className="w-full"
          role="img"
          aria-label={`Attention from token ${q} in GPT-2 small layer ${LAYER} head ${HEAD}`}
        >
          <defs>
            <linearGradient id="hero-arc" x1="0" x2="1">
              <stop offset="0" stopColor="var(--teal)" />
              <stop offset="1" stopColor="var(--accent)" />
            </linearGradient>
          </defs>
          {Array.from({ length: q + 1 }, (_, k) => {
            const w = m![q * T + k]!;
            if (w < 0.02 || k === q) return null;
            const x1 = x(q);
            const x2 = x(k);
            const h = Math.min(170, Math.abs(x1 - x2) * 0.5 + 18);
            return (
              <path
                key={`${q}-${k}`}
                d={`M ${x1} ${baseY - 20} C ${x1} ${baseY - 20 - h}, ${x2} ${baseY - 20 - h}, ${x2} ${baseY - 20}`}
                fill="none"
                stroke="url(#hero-arc)"
                strokeOpacity={0.25 + 0.75 * w}
                strokeWidth={1 + 9 * w}
                strokeLinecap="round"
                className="transition-all duration-500"
              />
            );
          })}
          {run.input.tokens.map((t, i) => {
            const w = i <= q ? m![q * T + i]! : 0;
            return (
              <g key={i}>
                <circle
                  cx={x(i)}
                  cy={baseY - 20}
                  r={3 + 6 * w}
                  fill={i === q ? 'var(--accent)' : 'var(--teal)'}
                  fillOpacity={i === q ? 1 : 0.2 + 0.8 * w}
                />
                <text
                  x={x(i)}
                  y={baseY + 6}
                  textAnchor="middle"
                  fontSize="15"
                  className="font-mono"
                  fill={i === q ? 'var(--accent)' : i <= q ? 'var(--fg)' : 'var(--muted)'}
                  fillOpacity={i <= q ? 1 : 0.45}
                >
                  {t.special ? '⟨bos⟩' : t.text.trim() || '·'}
                </text>
              </g>
            );
          })}
        </svg>
      )}
      <figcaption className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[11.5px] text-muted">
        <span>
          Real attention · GPT-2 small, layer {LAYER}, head {HEAD} · precomputed, SHA-256 verified
        </span>
        <span>On the repeat, each token attends to what followed it the first time (an induction head).</span>
      </figcaption>
    </figure>
  );
}
