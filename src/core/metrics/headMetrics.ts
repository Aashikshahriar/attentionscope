import type { AttentionTensor } from '@/types/contracts';
import { headMatrix, isCausallyVisible } from '@/core/attention/tensor';

export const ENTROPY_EPSILON = 1e-12;

/** Number of keys visible from query i under the causal mask. */
export function validKeyCount(query: number, keyLength: number): number {
  return Math.min(query + 1, keyLength);
}

/**
 * Shannon entropy (nats) of one attention row, H_i = -sum_j a_ij log(a_ij + eps),
 * summed over causally visible keys only.
 */
export function rowEntropy(m: ArrayLike<number>, T: number, i: number): number {
  let h = 0;
  for (let j = 0; j < T; j++) {
    if (!isCausallyVisible(i, j)) continue;
    const a = m[i * T + j]!;
    if (a > 0) h -= a * Math.log(a + ENTROPY_EPSILON);
  }
  return Math.max(0, h);
}

/** Entropy divided by its maximum, log(n_valid). Rows with one visible key are undefined (NaN). */
export function normalizedRowEntropy(m: ArrayLike<number>, T: number, i: number): number {
  const n = validKeyCount(i, T);
  if (n < 2) return Number.NaN;
  return rowEntropy(m, T, i) / Math.log(n);
}

/** Expected distance back to the attended key, sum_j a_ij (i - j). */
export function rowMeanDistance(m: ArrayLike<number>, T: number, i: number): number {
  let d = 0;
  for (let j = 0; j <= i && j < T; j++) d += m[i * T + j]! * (i - j);
  return d;
}

export interface HeadStats {
  layer: number;
  head: number;
  /** mean row entropy (nats) over rows with at least two visible keys */
  entropy: number;
  /** mean normalized entropy in [0, 1] over rows with at least two visible keys */
  normalizedEntropy: number;
  /** mean attention distance (tokens) over rows with at least two visible keys */
  meanDistance: number;
  /** mean a_{i,i-1} over i >= 1 */
  previousToken: number;
  /** mean a_{i,0} over i >= 1 (attention to the first position, often a "sink") */
  firstToken: number;
  /** mean a_{i,i} over i >= 1 */
  selfAttention: number;
  /** mean of the row maximum over i >= 1 */
  maxWeight: number;
}

export function computeHeadStats(m: ArrayLike<number>, T: number, layer = 0, head = 0): HeadStats {
  let ent = 0;
  let nent = 0;
  let dist = 0;
  let prev = 0;
  let first = 0;
  let self = 0;
  let maxw = 0;
  let rows = 0;
  for (let i = 1; i < T; i++) {
    rows++;
    ent += rowEntropy(m, T, i);
    nent += normalizedRowEntropy(m, T, i);
    dist += rowMeanDistance(m, T, i);
    prev += m[i * T + i - 1]!;
    first += m[i * T]!;
    self += m[i * T + i]!;
    let mx = 0;
    for (let j = 0; j <= i; j++) mx = Math.max(mx, m[i * T + j]!);
    maxw += mx;
  }
  const div = (v: number) => (rows > 0 ? v / rows : Number.NaN);
  return {
    layer,
    head,
    entropy: div(ent),
    normalizedEntropy: div(nent),
    meanDistance: div(dist),
    previousToken: div(prev),
    firstToken: div(first),
    selfAttention: div(self),
    maxWeight: div(maxw),
  };
}

export function computeAllHeadStats(t: AttentionTensor): HeadStats[][] {
  const T = t.shape.queryLength;
  return Array.from({ length: t.shape.layers }, (_, l) =>
    Array.from({ length: t.shape.heads }, (_, h) => computeHeadStats(headMatrix(t, l, h), T, l, h)),
  );
}

/**
 * Jensen–Shannon divergence (bits, in [0, 1]) between two attention patterns,
 * averaged over query rows with at least two visible keys.
 */
export function jensenShannon(a: ArrayLike<number>, b: ArrayLike<number>, T: number): number {
  let total = 0;
  let rows = 0;
  for (let i = 1; i < T; i++) {
    let js = 0;
    for (let j = 0; j <= i; j++) {
      const p = a[i * T + j]!;
      const q = b[i * T + j]!;
      const mix = 0.5 * (p + q);
      if (p > 0) js += 0.5 * p * Math.log2(p / mix);
      if (q > 0) js += 0.5 * q * Math.log2(q / mix);
    }
    total += Math.max(0, js);
    rows++;
  }
  return rows ? total / rows : 0;
}

/** Pearson correlation over causally visible entries of rows i >= 1. NaN if either is constant. */
export function pearsonVisible(a: ArrayLike<number>, b: ArrayLike<number>, T: number): number {
  let n = 0;
  let sa = 0;
  let sb = 0;
  for (let i = 1; i < T; i++)
    for (let j = 0; j <= i; j++) {
      sa += a[i * T + j]!;
      sb += b[i * T + j]!;
      n++;
    }
  if (n < 2) return Number.NaN;
  const ma = sa / n;
  const mb = sb / n;
  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 1; i < T; i++)
    for (let j = 0; j <= i; j++) {
      const da = a[i * T + j]! - ma;
      const db = b[i * T + j]! - mb;
      cov += da * db;
      va += da * da;
      vb += db * db;
    }
  if (va === 0 || vb === 0) return Number.NaN;
  return cov / Math.sqrt(va * vb);
}

export type SimilarityMetric = 'js' | 'pearson';

/** Similarity of every head to a reference head: 1 - JSD, or Pearson r. */
export function similarityToHead(
  t: AttentionTensor,
  layer: number,
  head: number,
  metric: SimilarityMetric,
): number[][] {
  const T = t.shape.queryLength;
  const ref = headMatrix(t, layer, head);
  return Array.from({ length: t.shape.layers }, (_, l) =>
    Array.from({ length: t.shape.heads }, (_, h) => {
      const m = headMatrix(t, l, h);
      return metric === 'js' ? 1 - jensenShannon(ref, m, T) : pearsonVisible(ref, m, T);
    }),
  );
}

/**
 * Attention rollout (Abnar & Zuidema, 2020).
 *
 * Assumptions, stated explicitly: heads are averaged with equal weight; the
 * residual connection is modelled as mixing `residualWeight` of the identity
 * into each layer's averaged attention (rows renormalized); MLPs, LayerNorm and
 * value/output projections are ignored. The result is a heuristic flow
 * estimate, not a measurement of causal influence.
 *
 * Returns R (row-major T×T) where R[i, j] estimates how much input position j
 * contributes to position i after the last layer.
 */
export function attentionRollout(t: AttentionTensor, residualWeight = 0.5, upToLayer?: number): Float64Array {
  const T = t.shape.queryLength;
  const L = upToLayer === undefined ? t.shape.layers : upToLayer + 1;
  let R = new Float64Array(T * T);
  for (let i = 0; i < T; i++) R[i * T + i] = 1;
  const layerMix = new Float64Array(T * T);
  for (let l = 0; l < L; l++) {
    layerMix.fill(0);
    for (let h = 0; h < t.shape.heads; h++) {
      const m = headMatrix(t, l, h);
      for (let x = 0; x < T * T; x++) layerMix[x] = layerMix[x]! + m[x]! / t.shape.heads;
    }
    for (let i = 0; i < T; i++) {
      let s = 0;
      for (let j = 0; j < T; j++) {
        const v = (1 - residualWeight) * layerMix[i * T + j]! + (i === j ? residualWeight : 0);
        layerMix[i * T + j] = v;
        s += v;
      }
      for (let j = 0; j < T; j++) layerMix[i * T + j] = layerMix[i * T + j]! / s;
    }
    const next = new Float64Array(T * T);
    for (let i = 0; i < T; i++)
      for (let k = 0; k < T; k++) {
        const a = layerMix[i * T + k]!;
        if (a === 0) continue;
        for (let j = 0; j < T; j++) next[i * T + j] = next[i * T + j]! + a * R[k * T + j]!;
      }
    R = next;
  }
  return R;
}
