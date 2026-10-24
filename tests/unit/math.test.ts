import { describe, expect, it } from 'vitest';
import { erf, geluErf } from '@/core/math/erf';
import { decodeFloat16, float16Bits } from '@/core/math/float16';
import { logSoftmax, softmaxRow, topK } from '@/core/attention/softmax';
import { matmul, transpose } from '@/core/math/matrix';
import { mulberry32 } from '@/core/math/random';

describe('erf / GELU', () => {
  it('matches reference values to double precision', () => {
    // Reference values from mpmath (50 digits), rounded to double.
    const cases: [number, number][] = [
      [0, 0],
      [0.1, 0.1124629160182849],
      [0.5, 0.5204998778130465],
      [1, 0.8427007929497149],
      [2, 0.9953222650189527],
      [3, 0.9999779095030014],
      [-1.5, -0.9661051464753108],
    ];
    for (const [x, y] of cases) expect(Math.abs(erf(x) - y)).toBeLessThan(2e-16 + 1e-15 * Math.abs(y));
    expect(erf(7)).toBe(1);
    expect(erf(-7)).toBe(-1);
  });

  it('computes exact GELU', () => {
    expect(geluErf(0)).toBe(0);
    expect(geluErf(1)).toBeCloseTo(0.8413447460685429, 15);
    // -3 * Phi(-3), Phi(-3) = 0.0013498980316300945
    expect(geluErf(-3)).toBeCloseTo(-0.00404969409489028, 15);
  });
});

describe('float16 decoding', () => {
  it('decodes special and boundary bit patterns', () => {
    expect(float16Bits(0x3c00)).toBe(1);
    expect(float16Bits(0xc000)).toBe(-2);
    expect(float16Bits(0x7bff)).toBe(65504);
    expect(float16Bits(0x0001)).toBe(2 ** -24);
    expect(float16Bits(0x7c00)).toBe(Infinity);
    expect(Number.isNaN(float16Bits(0x7e00))).toBe(true);
    expect(Object.is(float16Bits(0x8000), -0)).toBe(true);
    expect(Array.from(decodeFloat16(new Uint16Array([0x3555, 0xbc00])))).toEqual([
      Math.fround(0.333251953125),
      -1,
    ]);
  });
});

describe('stable softmax', () => {
  it('is invariant to large offsets and sums to one', () => {
    const { probs } = softmaxRow([1000, 1001, 1002]);
    const e = [Math.exp(-2), Math.exp(-1), 1];
    const s = e[0]! + e[1]! + e[2]!;
    expect(probs[0]).toBeCloseTo(e[0]! / s, 15);
    expect(probs[2]).toBeCloseTo(e[2]! / s, 15);
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 15);
    const small = softmaxRow([-1000, -1001]).probs;
    expect(small[0]! + small[1]!).toBeCloseTo(1, 15);
  });

  it('assigns exactly zero to masked keys and never renormalizes them in', () => {
    const { probs, fullyMasked } = softmaxRow([5, 100, 5], [true, false, true]);
    expect(fullyMasked).toBe(false);
    expect(probs[1]).toBe(0);
    expect(probs[0]).toBeCloseTo(0.5, 15);
  });

  it('treats -Infinity scores as masked', () => {
    const { probs } = softmaxRow([0, Number.NEGATIVE_INFINITY]);
    expect(Array.from(probs)).toEqual([1, 0]);
  });

  it('reports fully masked rows instead of producing NaN', () => {
    const { probs, fullyMasked } = softmaxRow([1, 2], [false, false]);
    expect(fullyMasked).toBe(true);
    expect(Array.from(probs)).toEqual([0, 0]);
  });

  it('log-softmax matches log(softmax)', () => {
    const logits = [2, -1, 0.5, 30];
    const ls = logSoftmax(logits);
    const p = softmaxRow(logits).probs;
    for (let i = 0; i < logits.length; i++) expect(ls[i]).toBeCloseTo(Math.log(p[i]!), 12);
  });

  it('topK returns descending indices', () => {
    expect(topK([0.1, 0.7, 0.05, 0.15], 3)).toEqual([1, 3, 0]);
  });
});

describe('matrix helpers and PRNG', () => {
  it('multiplies with shape checking', () => {
    expect(matmul([[1, 2]], [[3], [4]])).toEqual([[11]]);
    expect(() => matmul([[1, 2]], [[1, 2]])).toThrow(/shape mismatch/);
    expect(transpose([[1, 2, 3]])).toEqual([[1], [2], [3]]);
  });

  it('is deterministic for a seed', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });
});
