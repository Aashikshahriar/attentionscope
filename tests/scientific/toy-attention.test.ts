import { describe, expect, it } from 'vitest';
import { runToyAttention, type ToyConfig } from '@/core/attention/toyAttention';
import type { Matrix } from '@/core/math/matrix';
import { flat, maxAbsDiff, readJson } from '../helpers';

interface HeadRef {
  Q: Matrix;
  K: Matrix;
  V: Matrix;
  rawScores: Matrix;
  scaledScores: Matrix;
  allowed: boolean[][];
  attention: Matrix;
  output: Matrix;
  scale: number;
}
interface ToyCase {
  name: string;
  config: ToyConfig;
  X: Matrix;
  WQ: Matrix[];
  WK: Matrix[];
  WV: Matrix[];
  WO: Matrix;
  expected: { heads: HeadRef[]; concat: Matrix; projected: Matrix };
}

const { cases } = readJson<{ cases: ToyCase[] }>('tests/fixtures/toy-attention.json');
const TOL = 1e-12;

describe('educational attention vs independent NumPy reference', () => {
  for (const c of cases) {
    it(c.name, () => {
      const r = runToyAttention({ X: c.X, WQ: c.WQ, WK: c.WK, WV: c.WV, WO: c.WO }, c.config);
      expect(r.heads).toHaveLength(c.expected.heads.length);
      r.heads.forEach((h, i) => {
        const e = c.expected.heads[i]!;
        expect(h.scale).toBeCloseTo(e.scale, 15);
        for (const key of ['Q', 'K', 'V', 'rawScores', 'scaledScores', 'attention', 'output'] as const) {
          expect(maxAbsDiff(flat(h[key]), flat(e[key])), key).toBeLessThan(TOL);
        }
        expect(h.allowed).toEqual(e.allowed);
        // row-wise normalization over visible keys; fully masked rows are all zero
        h.attention.forEach((row, q) => {
          const s = row.reduce((a, b) => a + b, 0);
          if (h.fullyMaskedRows.includes(q)) expect(s).toBe(0);
          else expect(s).toBeCloseTo(1, 12);
          row.forEach((v, k) => {
            if (!h.allowed[q]![k]) expect(v).toBe(0);
          });
        });
      });
      expect(maxAbsDiff(flat(r.concat), flat(c.expected.concat))).toBeLessThan(TOL);
      expect(maxAbsDiff(flat(r.projected), flat(c.expected.projected))).toBeLessThan(TOL);
    });
  }

  it('causal + padding produces a fully masked first row', () => {
    const c = cases.find((x) => x.name.includes('fully masked'))!;
    const r = runToyAttention({ X: c.X, WQ: c.WQ, WK: c.WK, WV: c.WV, WO: c.WO }, c.config);
    expect(r.heads[0]!.fullyMaskedRows).toEqual([0]);
  });

  it('rejects inconsistent shapes', () => {
    const c = cases[0]!;
    expect(() => runToyAttention({ X: c.X, WQ: c.WQ, WK: c.WK, WV: c.WV, WO: [[1]] }, c.config)).toThrow(
      /W_O/,
    );
    expect(() =>
      runToyAttention({ X: c.X, WQ: c.WQ, WK: c.WK, WV: c.WV, WO: c.WO }, { ...c.config, temperature: 0 }),
    ).toThrow(/temperature/);
  });
});
