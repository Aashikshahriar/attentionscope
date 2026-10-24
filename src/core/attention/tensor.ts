import type { AttentionShape, AttentionTensor, InterventionValidation } from '@/types/contracts';

export function assertShape(shape: AttentionShape, length: number): void {
  const { batch, layers, heads, queryLength, keyLength } = shape;
  const expected = batch * layers * heads * queryLength * keyLength;
  if (![batch, layers, heads, queryLength, keyLength].every((d) => Number.isInteger(d) && d > 0)) {
    throw new Error(`invalid attention shape ${JSON.stringify(shape)}`);
  }
  if (batch !== 1) throw new Error('only batch size 1 is supported');
  if (expected !== length) {
    throw new Error(
      `attention data has ${length} values but shape ${JSON.stringify(shape)} needs ${expected}`,
    );
  }
}

export function headOffset(shape: AttentionShape, layer: number, head: number): number {
  if (layer < 0 || layer >= shape.layers) throw new RangeError(`layer ${layer} out of range`);
  if (head < 0 || head >= shape.heads) throw new RangeError(`head ${head} out of range`);
  return (layer * shape.heads + head) * shape.queryLength * shape.keyLength;
}

/** Zero-copy view of one head's [query, key] matrix (row-major). */
export function headMatrix(t: AttentionTensor, layer: number, head: number): Float32Array {
  const size = t.shape.queryLength * t.shape.keyLength;
  const o = headOffset(t.shape, layer, head);
  return t.data.subarray(o, o + size);
}

export function headScores(t: AttentionTensor, layer: number, head: number): Float32Array | null {
  if (!t.scores) return null;
  const size = t.shape.queryLength * t.shape.keyLength;
  const o = headOffset(t.shape, layer, head);
  return t.scores.subarray(o, o + size);
}

export function attentionRow(t: AttentionTensor, layer: number, head: number, query: number): Float32Array {
  const m = headMatrix(t, layer, head);
  const K = t.shape.keyLength;
  return m.subarray(query * K, (query + 1) * K);
}

/** Causal-mask semantics: key j is visible from query i iff j <= i (equal-length sequences). */
export function isCausallyVisible(query: number, key: number): boolean {
  return key <= query;
}

/** Checks that a [T, T] matrix is a valid causal attention pattern. */
export function validateCausalMatrix(m: ArrayLike<number>, T: number, tol = 1e-6): InterventionValidation {
  let maxRowSumError = 0;
  let maskViolations = 0;
  let negativeEntries = 0;
  for (let i = 0; i < T; i++) {
    let s = 0;
    for (let j = 0; j < T; j++) {
      const v = m[i * T + j]!;
      if (v < 0) negativeEntries++;
      if (!isCausallyVisible(i, j) && v !== 0) maskViolations++;
      s += v;
    }
    maxRowSumError = Math.max(maxRowSumError, Math.abs(s - 1));
  }
  return {
    shape: [T, T],
    maxRowSumError,
    maskViolations,
    negativeEntries,
    passed: maxRowSumError <= tol && maskViolations === 0 && negativeEntries === 0,
  };
}

export function validateTensor(t: AttentionTensor, tol = 1e-4): InterventionValidation {
  assertShape(t.shape, t.data.length);
  const T = t.shape.queryLength;
  let worst: InterventionValidation = {
    shape: [T, t.shape.keyLength],
    maxRowSumError: 0,
    maskViolations: 0,
    negativeEntries: 0,
    passed: true,
  };
  for (let l = 0; l < t.shape.layers; l++) {
    for (let h = 0; h < t.shape.heads; h++) {
      const v = validateCausalMatrix(headMatrix(t, l, h), T, tol);
      worst = {
        shape: worst.shape,
        maxRowSumError: Math.max(worst.maxRowSumError, v.maxRowSumError),
        maskViolations: worst.maskViolations + v.maskViolations,
        negativeEntries: worst.negativeEntries + v.negativeEntries,
        passed: worst.passed && v.passed,
      };
    }
  }
  return worst;
}
