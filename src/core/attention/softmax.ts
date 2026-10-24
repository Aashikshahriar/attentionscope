/**
 * Numerically stable softmax over one row.
 *
 * `allowed[j] === false` means key j is masked: it is treated as having a
 * score of -infinity and receives exactly zero probability. If every key in a
 * row is masked the softmax is undefined; we return all zeros and report it
 * via the `fullyMasked` flag instead of producing NaN.
 */
export function softmaxRow(
  scores: ArrayLike<number>,
  allowed?: ArrayLike<boolean>,
  out: Float64Array = new Float64Array(scores.length),
): { probs: Float64Array; fullyMasked: boolean; max: number; sumExp: number } {
  const n = scores.length;
  let max = Number.NEGATIVE_INFINITY;
  for (let j = 0; j < n; j++) {
    if (allowed && !allowed[j]) continue;
    const s = scores[j]!;
    if (s > max) max = s;
  }
  if (max === Number.NEGATIVE_INFINITY) {
    out.fill(0);
    return { probs: out, fullyMasked: true, max, sumExp: 0 };
  }
  let sum = 0;
  for (let j = 0; j < n; j++) {
    if ((allowed && !allowed[j]) || scores[j] === Number.NEGATIVE_INFINITY) {
      out[j] = 0;
      continue;
    }
    const e = Math.exp(scores[j]! - max);
    out[j] = e;
    sum += e;
  }
  for (let j = 0; j < n; j++) out[j] = out[j]! / sum;
  return { probs: out, fullyMasked: false, max, sumExp: sum };
}

/** log-softmax over a full vector (for next-token distributions). */
export function logSoftmax(logits: ArrayLike<number>): Float64Array {
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < logits.length; i++) if (logits[i]! > max) max = logits[i]!;
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i]! - max);
  const lse = max + Math.log(sum);
  const out = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) out[i] = logits[i]! - lse;
  return out;
}

/** Indices of the k largest values, descending. */
export function topK(values: ArrayLike<number>, k: number): number[] {
  const idx: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (idx.length < k) {
      idx.push(i);
      idx.sort((a, b) => values[b]! - values[a]!);
    } else if (v > values[idx[k - 1]!]!) {
      idx[k - 1] = i;
      idx.sort((a, b) => values[b]! - values[a]!);
    }
  }
  return idx;
}
