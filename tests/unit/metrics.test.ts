import { describe, expect, it } from 'vitest';
import {
  attentionRollout,
  computeHeadStats,
  jensenShannon,
  normalizedRowEntropy,
  pearsonVisible,
  rowEntropy,
  rowMeanDistance,
  similarityToHead,
} from '@/core/metrics/headMetrics';
import { assertShape, headMatrix, headOffset, validateCausalMatrix } from '@/core/attention/tensor';
import { replacementMatrix } from '@/core/model-adapters/interventions';
import type { AttentionTensor } from '@/types/contracts';

const uniform = (T: number) => replacementMatrix('replace-attention-uniform', T)!;
const prev = (T: number) => replacementMatrix('replace-attention-previous-token', T)!;
const first = (T: number) => replacementMatrix('replace-attention-first-token', T)!;

function tensorFrom(heads: Float64Array[][], T: number): AttentionTensor {
  const layers = heads.length;
  const H = heads[0]!.length;
  const data = new Float32Array(layers * H * T * T);
  heads.forEach((row, l) => row.forEach((m, h) => data.set(m, (l * H + h) * T * T)));
  return {
    shape: { batch: 1, layers, heads: H, queryLength: T, keyLength: T },
    layout: ['layer', 'head', 'query', 'key'],
    kind: 'post-softmax',
    mask: 'causal',
    data,
  };
}

describe('entropy', () => {
  it('equals log(n) for uniform rows and 0 for one-hot rows', () => {
    const T = 5;
    for (let i = 0; i < T; i++) expect(rowEntropy(uniform(T), T, i)).toBeCloseTo(Math.log(i + 1), 9);
    for (let i = 0; i < T; i++) expect(rowEntropy(prev(T), T, i)).toBeCloseTo(0, 9);
    expect(normalizedRowEntropy(uniform(T), T, 3)).toBeCloseTo(1, 9);
    expect(Number.isNaN(normalizedRowEntropy(uniform(T), T, 0))).toBe(true);
  });

  it('ignores masked (future) entries', () => {
    const T = 3;
    const m = new Float64Array([1, 0, 0, 0.5, 0.5, 0, 0.2, 0.3, 0.5]);
    const dirty = m.slice();
    dirty[1] = 0.9; // a non-zero masked value must not be counted
    expect(rowEntropy(dirty, T, 0)).toBe(rowEntropy(m, T, 0));
  });
});

describe('distance and pattern scores', () => {
  it('measures attention distance', () => {
    const T = 6;
    expect(rowMeanDistance(prev(T), T, 4)).toBe(1);
    expect(rowMeanDistance(first(T), T, 4)).toBe(4);
    expect(rowMeanDistance(uniform(T), T, 4)).toBeCloseTo(2, 12);
  });

  it('identifies previous-token and first-token heads', () => {
    const T = 8;
    const p = computeHeadStats(prev(T), T);
    expect(p.previousToken).toBe(1);
    expect(p.meanDistance).toBe(1);
    const f = computeHeadStats(first(T), T);
    expect(f.firstToken).toBe(1);
    expect(f.entropy).toBeCloseTo(0, 9);
  });
});

describe('head similarity', () => {
  it('JSD is 0 for identical and 1 for disjoint patterns', () => {
    const T = 6;
    expect(jensenShannon(prev(T), prev(T), T)).toBe(0);
    // previous-token vs self-only patterns are disjoint on every row i >= 1
    const self = new Float64Array(T * T);
    for (let i = 0; i < T; i++) self[i * T + i] = 1;
    expect(jensenShannon(prev(T), self, T)).toBeCloseTo(1, 12);
  });

  it('Pearson is 1 for identical patterns', () => {
    const T = 5;
    expect(pearsonVisible(prev(T), prev(T), T)).toBeCloseTo(1, 12);
  });

  it('similarityToHead returns a layers×heads grid', () => {
    const T = 4;
    const t = tensorFrom(
      [
        [prev(T), uniform(T)],
        [first(T), prev(T)],
      ],
      T,
    );
    const s = similarityToHead(t, 0, 0, 'js');
    expect(s).toHaveLength(2);
    expect(s[0]![0]).toBe(1);
    expect(s[1]![1]).toBe(1);
  });
});

describe('tensor layout', () => {
  it('uses [layer, head, query, key] row-major order', () => {
    const shape = { batch: 1 as const, layers: 2, heads: 3, queryLength: 4, keyLength: 4 };
    expect(headOffset(shape, 1, 2)).toBe((1 * 3 + 2) * 16);
    const data = new Float32Array(2 * 3 * 16).map((_, i) => i);
    const t: AttentionTensor = {
      shape,
      layout: ['layer', 'head', 'query', 'key'],
      kind: 'post-softmax',
      mask: 'causal',
      data,
    };
    expect(headMatrix(t, 1, 0)[0]).toBe(48);
    expect(() => assertShape(shape, 95)).toThrow(/needs 96/);
    expect(() => headOffset(shape, 2, 0)).toThrow(RangeError);
  });

  it('validates causal row-stochastic matrices', () => {
    const T = 4;
    expect(validateCausalMatrix(uniform(T), T).passed).toBe(true);
    const bad = uniform(T).slice();
    bad[1] = 0.1; // future key
    const v = validateCausalMatrix(bad, T);
    expect(v.passed).toBe(false);
    expect(v.maskViolations).toBe(1);
  });
});

describe('attention rollout', () => {
  it('produces row-stochastic matrices and reduces to the identity with residualWeight = 1', () => {
    const T = 5;
    const t = tensorFrom(
      [
        [uniform(T), prev(T)],
        [prev(T), first(T)],
      ],
      T,
    );
    const R = attentionRollout(t, 0.5);
    for (let i = 0; i < T; i++) {
      let s = 0;
      for (let j = 0; j < T; j++) s += R[i * T + j]!;
      expect(s).toBeCloseTo(1, 12);
      for (let j = i + 1; j < T; j++) expect(R[i * T + j]).toBe(0);
    }
    const I = attentionRollout(t, 1);
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) expect(I[i * T + j]).toBe(i === j ? 1 : 0);
  });
});
