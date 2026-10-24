import { softmaxRow } from './softmax';
import { hconcat, matmul, shapeOf, transpose, type Matrix } from '@/core/math/matrix';

/**
 * A small, fully transparent multi-head self-attention module for teaching.
 * It is NOT a pretrained model: inputs and weights are toy values the user edits.
 *
 *   Q_h = X W_Q^h,  K_h = X W_K^h,  V_h = X W_V^h
 *   A_h = softmax(Q_h K_h^T * s + M),  s = (useScaling ? 1/sqrt(d_k) : 1) / temperature
 *   O_h = A_h V_h,  Y = [O_1 … O_H] W_O
 */
export interface ToyConfig {
  temperature: number;
  useScaling: boolean;
  causal: boolean;
  /** padding[j] = true marks key j as padding (masked for every query). */
  padding: boolean[];
}

export interface ToyWeights {
  X: Matrix;
  WQ: Matrix[];
  WK: Matrix[];
  WV: Matrix[];
  WO: Matrix;
}

export interface ToyHeadResult {
  Q: Matrix;
  K: Matrix;
  V: Matrix;
  rawScores: Matrix;
  scaledScores: Matrix;
  /** scaled scores with -Infinity at masked positions */
  maskedScores: Matrix;
  allowed: boolean[][];
  attention: Matrix;
  output: Matrix;
  scale: number;
  fullyMaskedRows: number[];
  /** per-row softmax internals for the worked example */
  rowMax: number[];
  rowSumExp: number[];
}

export interface ToyResult {
  heads: ToyHeadResult[];
  concat: Matrix;
  projected: Matrix;
  dims: { T: number; dModel: number; dK: number; dV: number; heads: number };
}

export function validateToyWeights(w: ToyWeights): void {
  const [T, d] = shapeOf(w.X);
  const H = w.WQ.length;
  if (T < 1 || d < 1) throw new Error('X must be non-empty');
  if (H < 1 || w.WK.length !== H || w.WV.length !== H)
    throw new Error('W_Q, W_K, W_V need one matrix per head');
  const [, dk] = shapeOf(w.WQ[0]!);
  const [, dv] = shapeOf(w.WV[0]!);
  for (let h = 0; h < H; h++) {
    const q = shapeOf(w.WQ[h]!);
    const k = shapeOf(w.WK[h]!);
    const v = shapeOf(w.WV[h]!);
    if (q[0] !== d || k[0] !== d || v[0] !== d)
      throw new Error(`head ${h}: projection rows must equal d_model=${d}`);
    if (q[1] !== dk || k[1] !== dk)
      throw new Error(`head ${h}: W_Q and W_K must both have d_k=${dk} columns`);
    if (v[1] !== dv) throw new Error(`head ${h}: W_V must have d_v=${dv} columns`);
  }
  const o = shapeOf(w.WO);
  if (o[0] !== H * dv || o[1] !== d) throw new Error(`W_O must be (${H * dv}×${d}), got (${o[0]}×${o[1]})`);
}

export function buildAllowedMask(T: number, causal: boolean, padding: boolean[]): boolean[][] {
  return Array.from({ length: T }, (_, i) =>
    Array.from({ length: T }, (_, j) => (!causal || j <= i) && !padding[j]),
  );
}

export function runToyAttention(w: ToyWeights, cfg: ToyConfig): ToyResult {
  validateToyWeights(w);
  if (!(cfg.temperature > 0)) throw new Error('temperature must be positive');
  const [T, dModel] = shapeOf(w.X);
  if (cfg.padding.length !== T) throw new Error('padding mask length must equal sequence length');
  const allowed = buildAllowedMask(T, cfg.causal, cfg.padding);
  const heads: ToyHeadResult[] = w.WQ.map((WQ, h) => {
    const Q = matmul(w.X, WQ);
    const K = matmul(w.X, w.WK[h]!);
    const V = matmul(w.X, w.WV[h]!);
    const dk = Q[0]!.length;
    const scale = (cfg.useScaling ? 1 / Math.sqrt(dk) : 1) / cfg.temperature;
    const rawScores = matmul(Q, transpose(K));
    const scaledScores = rawScores.map((r) => r.map((v) => v * scale));
    const maskedScores = scaledScores.map((r, i) =>
      r.map((v, j) => (allowed[i]![j] ? v : Number.NEGATIVE_INFINITY)),
    );
    const fullyMaskedRows: number[] = [];
    const rowMax: number[] = [];
    const rowSumExp: number[] = [];
    const attention = scaledScores.map((r, i) => {
      const res = softmaxRow(r, allowed[i]);
      if (res.fullyMasked) fullyMaskedRows.push(i);
      rowMax.push(res.max);
      rowSumExp.push(res.sumExp);
      return Array.from(res.probs);
    });
    const output = matmul(attention, V);
    return {
      Q,
      K,
      V,
      rawScores,
      scaledScores,
      maskedScores,
      allowed,
      attention,
      output,
      scale,
      fullyMaskedRows,
      rowMax,
      rowSumExp,
    };
  });
  const concat = hconcat(heads.map((h) => h.output));
  const projected = matmul(concat, w.WO);
  return {
    heads,
    concat,
    projected,
    dims: { T, dModel, dK: heads[0]!.Q[0]!.length, dV: heads[0]!.V[0]!.length, heads: heads.length },
  };
}
