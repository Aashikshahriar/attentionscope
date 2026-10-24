import type { InterventionSpec, InterventionValidation, ModelManifest } from '@/types/contracts';
import { decodeFloat16, float16Table } from '@/core/math/float16';
import { geluErf } from '@/core/math/erf';
import { softmaxRow } from '@/core/attention/softmax';
import { validateCausalMatrix } from '@/core/attention/tensor';
import { replacementMatrix } from './interventions';

/**
 * GPT-NeoX forward pass (Pythia) written directly from the architecture:
 *
 *   x1 = LN1(h);  [q|k|v]_head = x1 W_qkv^T + b   (per-head interleaved layout)
 *   q, k <- rotary(q, k) on the first `rotaryDims` dims (NeoX "rotate_half")
 *   A_h = softmax(q k^T / sqrt(d_head) + causal mask);  O_h = A_h v
 *   attn = [O_1 … O_H] W_O^T + b
 *   h <- h + attn + MLP(LN2(h))                 (parallel residual)
 *   logits = LN_f(h) W_U^T
 *
 * Weights are float16 values held exactly in float32; activations are float64.
 * Verified against exact-float64 transformers in tests/scientific.
 */

export interface NeoxLayerWeights {
  ln1w: Float32Array;
  ln1b: Float32Array;
  ln2w: Float32Array;
  ln2b: Float32Array;
  qkvW: Float32Array;
  qkvB: Float32Array;
  denseW: Float32Array;
  denseB: Float32Array;
  fc1W: Float32Array;
  fc1B: Float32Array;
  fc2W: Float32Array;
  fc2B: Float32Array;
}

export interface NeoxDims {
  layers: number;
  heads: number;
  hidden: number;
  headDim: number;
  rotaryDims: number;
  rotaryBase: number;
  ff: number;
  vocab: number;
  eps: number;
}

export interface NeoxModel {
  dims: NeoxDims;
  /** float16 bits, decoded per looked-up row */
  embedIn: Uint16Array;
  layers: NeoxLayerWeights[];
  lnfW: Float32Array;
  lnfB: Float32Array;
  embedOut: Float32Array;
}

export function loadNeoxModel(manifest: ModelManifest, buffer: ArrayBuffer): NeoxModel {
  const a = manifest.architecture;
  if (a.family !== 'gpt-neox') throw new Error(`expected gpt-neox, got ${a.family}`);
  if (!a.parallelResidual) throw new Error('only parallel-residual GPT-NeoX is implemented');
  if (a.numKeyValueHeads !== a.numHeads) throw new Error('grouped-query attention is not implemented');
  if (buffer.byteLength !== manifest.weights.byteLength) {
    throw new Error(`weights are ${buffer.byteLength} bytes; manifest says ${manifest.weights.byteLength}`);
  }
  const dims: NeoxDims = {
    layers: a.numLayers,
    heads: a.numHeads,
    hidden: a.hiddenSize,
    headDim: a.headDim,
    rotaryDims: a.rotaryDims ?? a.headDim,
    rotaryBase: a.rotaryBase ?? 10000,
    ff: a.intermediateSize ?? 4 * a.hiddenSize,
    vocab: a.vocabSize ?? 0,
    eps: a.layerNormEps ?? 1e-5,
  };
  if (dims.headDim * dims.heads !== dims.hidden) throw new Error('headDim × heads must equal hidden size');
  const byName = new Map(manifest.weights.tensors.map((t) => [t.name, t]));
  const raw = (name: string, shape: number[]): Uint16Array => {
    const t = byName.get(name);
    if (!t) throw new Error(`missing tensor ${name}`);
    if (t.shape.length !== shape.length || t.shape.some((d, i) => d !== shape[i])) {
      throw new Error(`${name}: shape ${JSON.stringify(t.shape)} != expected ${JSON.stringify(shape)}`);
    }
    if (t.byteOffset % 2 !== 0) throw new Error(`${name}: misaligned offset`);
    return new Uint16Array(buffer, t.byteOffset, t.elementCount);
  };
  const f = (name: string, shape: number[]) => decodeFloat16(raw(name, shape));
  const { hidden: D, ff, vocab } = dims;
  const layers: NeoxLayerWeights[] = [];
  for (let i = 0; i < dims.layers; i++) {
    const p = `gpt_neox.layers.${i}.`;
    layers.push({
      ln1w: f(p + 'input_layernorm.weight', [D]),
      ln1b: f(p + 'input_layernorm.bias', [D]),
      ln2w: f(p + 'post_attention_layernorm.weight', [D]),
      ln2b: f(p + 'post_attention_layernorm.bias', [D]),
      qkvW: f(p + 'attention.query_key_value.weight', [3 * D, D]),
      qkvB: f(p + 'attention.query_key_value.bias', [3 * D]),
      denseW: f(p + 'attention.dense.weight', [D, D]),
      denseB: f(p + 'attention.dense.bias', [D]),
      fc1W: f(p + 'mlp.dense_h_to_4h.weight', [ff, D]),
      fc1B: f(p + 'mlp.dense_h_to_4h.bias', [ff]),
      fc2W: f(p + 'mlp.dense_4h_to_h.weight', [D, ff]),
      fc2B: f(p + 'mlp.dense_4h_to_h.bias', [D]),
    });
  }
  return {
    dims,
    embedIn: raw('gpt_neox.embed_in.weight', [vocab, D]),
    layers,
    lnfW: f('gpt_neox.final_layer_norm.weight', [D]),
    lnfB: f('gpt_neox.final_layer_norm.bias', [D]),
    embedOut: f('embed_out.weight', [vocab, D]),
  };
}

function layerNorm(x: Float64Array, T: number, D: number, w: Float32Array, b: Float32Array, eps: number) {
  const out = new Float64Array(T * D);
  for (let t = 0; t < T; t++) {
    const o = t * D;
    let mu = 0;
    for (let d = 0; d < D; d++) mu += x[o + d]!;
    mu /= D;
    let v = 0;
    for (let d = 0; d < D; d++) {
      const c = x[o + d]! - mu;
      v += c * c;
    }
    const inv = 1 / Math.sqrt(v / D + eps);
    for (let d = 0; d < D; d++) out[o + d] = (x[o + d]! - mu) * inv * w[d]! + b[d]!;
  }
  return out;
}

/** y = x W^T + b with W stored [out, in] (PyTorch Linear layout). */
function linear(
  x: Float64Array,
  T: number,
  inDim: number,
  W: Float32Array,
  b: Float32Array | null,
  outDim: number,
) {
  const y = new Float64Array(T * outDim);
  for (let t = 0; t < T; t++) {
    const xo = t * inDim;
    const yo = t * outDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim;
      let s = b ? b[o]! : 0;
      for (let i = 0; i < inDim; i++) s += x[xo + i]! * W[wo + i]!;
      y[yo + o] = s;
    }
  }
  return y;
}

function unembed(model: NeoxModel, xf: Float64Array, offset: number): Float64Array {
  const { vocab, hidden: D } = model.dims;
  const W = model.embedOut;
  const logits = new Float64Array(vocab);
  for (let v = 0; v < vocab; v++) {
    const wo = v * D;
    let s = 0;
    for (let d = 0; d < D; d++) s += xf[offset + d]! * W[wo + d]!;
    logits[v] = s;
  }
  return logits;
}

export interface ForwardOptions {
  intervention?: InterventionSpec | null;
  collectScores?: boolean;
  logitLens?: boolean;
  /** compute next-token logits at every position (otherwise only the last) */
  allPositions?: boolean;
  /** awaited between layers so a worker can process cancellation messages */
  yieldControl?: () => Promise<void>;
}

export interface ForwardOutput {
  T: number;
  /** [layer, head, query, key] post-softmax probabilities actually used in the forward pass */
  attention: Float32Array;
  /** [layer, head, query, key] scaled scores q·k/sqrt(d) with -Infinity at masked positions */
  scores: Float32Array | null;
  /** next-token logits at the last position */
  logits: Float64Array;
  /** logits for every position when `allPositions` is set */
  positionLogits: Float64Array[] | null;
  /** logit lens at the last position: LN_f(residual after layer l) W_U^T */
  lensLogits: Float64Array[] | null;
  /** L2 norm of the residual stream, index 0 = embeddings, l+1 = after layer l */
  residualNorms: number[][];
  validation: InterventionValidation | null;
}

export async function neoxForward(
  model: NeoxModel,
  ids: number[],
  opts: ForwardOptions = {},
): Promise<ForwardOutput> {
  const {
    layers: L,
    heads: H,
    hidden: D,
    headDim: hd,
    rotaryDims: rot,
    rotaryBase,
    ff,
    vocab,
    eps,
  } = model.dims;
  const T = ids.length;
  if (T < 1) throw new Error('empty input');
  for (const id of ids)
    if (!(id >= 0 && id < vocab)) throw new RangeError(`token id ${id} outside vocabulary`);
  const spec = opts.intervention ?? null;
  if (spec && (spec.layer < 0 || spec.layer >= L || spec.head < 0 || spec.head >= H)) {
    throw new RangeError(`intervention target ${spec.layer}.${spec.head} does not exist`);
  }
  const replacement = spec ? replacementMatrix(spec.kind, T) : null;
  const validation = replacement ? validateCausalMatrix(replacement, T, 1e-12) : null;
  if (validation && !validation.passed)
    throw new Error('replacement attention is not a valid causal distribution');

  const table = float16Table();
  const h = new Float64Array(T * D);
  for (let t = 0; t < T; t++) {
    const src = ids[t]! * D;
    for (let d = 0; d < D; d++) h[t * D + d] = table[model.embedIn[src + d]!]!;
  }

  // rotary tables: emb = [freqs, freqs]
  const half = rot / 2;
  const cos = new Float64Array(T * rot);
  const sin = new Float64Array(T * rot);
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < half; i++) {
      const invFreq = 1 / rotaryBase ** ((2 * i) / rot);
      const ang = t * invFreq;
      cos[t * rot + i] = cos[t * rot + i + half] = Math.cos(ang);
      sin[t * rot + i] = sin[t * rot + i + half] = Math.sin(ang);
    }
  }

  const norm = (x: Float64Array, t: number) => {
    let s = 0;
    for (let d = 0; d < D; d++) s += x[t * D + d]! ** 2;
    return Math.sqrt(s);
  };
  const residualNorms: number[][] = [Array.from({ length: T }, (_, t) => norm(h, t))];
  const attention = new Float32Array(L * H * T * T);
  const scores = opts.collectScores ? new Float32Array(L * H * T * T) : null;
  const lensLogits: Float64Array[] | null = opts.logitLens ? [] : null;
  const scale = 1 / Math.sqrt(hd);
  const q = new Float64Array(T * hd);
  const k = new Float64Array(T * hd);
  const v = new Float64Array(T * hd);
  const rowScores = new Float64Array(T);
  const rowProbs = new Float64Array(T);
  const allowed: boolean[] = new Array<boolean>(T);
  const probs = new Float64Array(T * T);

  for (let l = 0; l < L; l++) {
    const W = model.layers[l]!;
    const x1 = layerNorm(h, T, D, W.ln1w, W.ln1b, eps);
    const qkv = linear(x1, T, D, W.qkvW, W.qkvB, 3 * D);
    const merged = new Float64Array(T * D);
    for (let head = 0; head < H; head++) {
      const base = head * 3 * hd;
      for (let t = 0; t < T; t++) {
        const o = t * 3 * D + base;
        for (let d = 0; d < hd; d++) {
          q[t * hd + d] = qkv[o + d]!;
          k[t * hd + d] = qkv[o + hd + d]!;
          v[t * hd + d] = qkv[o + 2 * hd + d]!;
        }
        // rotate_half on the first `rot` dims: [x1, x2] -> [-x2, x1]
        for (const arr of [q, k]) {
          const ro = t * hd;
          const tmp = new Float64Array(rot);
          for (let d = 0; d < rot; d++) {
            const rotated = d < half ? -arr[ro + d + half]! : arr[ro + d - half]!;
            tmp[d] = arr[ro + d]! * cos[t * rot + d]! + rotated * sin[t * rot + d]!;
          }
          for (let d = 0; d < rot; d++) arr[ro + d] = tmp[d]!;
        }
      }
      const isTarget = spec !== null && spec.layer === l && spec.head === head;
      const aBase = (l * H + head) * T * T;
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          allowed[j] = j <= i;
          if (j > i) {
            rowScores[j] = Number.NEGATIVE_INFINITY;
            continue;
          }
          let s = 0;
          for (let d = 0; d < hd; d++) s += q[i * hd + d]! * k[j * hd + d]!;
          rowScores[j] = s * scale;
        }
        softmaxRow(rowScores, allowed, rowProbs);
        for (let j = 0; j < T; j++) {
          probs[i * T + j] = isTarget && replacement ? replacement[i * T + j]! : rowProbs[j]!;
          if (scores) scores[aBase + i * T + j] = rowScores[j]!;
        }
      }
      for (let x = 0; x < T * T; x++) attention[aBase + x] = probs[x]!;
      if (isTarget && spec.kind === 'zero-head-output') continue; // O_h = 0
      for (let i = 0; i < T; i++) {
        for (let d = 0; d < hd; d++) {
          let s = 0;
          for (let j = 0; j <= i; j++) s += probs[i * T + j]! * v[j * hd + d]!;
          merged[i * D + head * hd + d] = s;
        }
      }
    }
    const attnOut = linear(merged, T, D, W.denseW, W.denseB, D);
    const x2 = layerNorm(h, T, D, W.ln2w, W.ln2b, eps);
    const mid = linear(x2, T, D, W.fc1W, W.fc1B, ff);
    for (let x = 0; x < mid.length; x++) mid[x] = geluErf(mid[x]!);
    const mlpOut = linear(mid, T, ff, W.fc2W, W.fc2B, D);
    for (let x = 0; x < h.length; x++) h[x] = h[x]! + attnOut[x]! + mlpOut[x]!;
    residualNorms.push(Array.from({ length: T }, (_, t) => norm(h, t)));
    if (lensLogits) {
      const last = h.subarray((T - 1) * D, T * D);
      const lf = layerNorm(new Float64Array(last), 1, D, model.lnfW, model.lnfB, eps);
      lensLogits.push(unembed(model, lf, 0));
    }
    if (opts.yieldControl) await opts.yieldControl();
  }
  const hf = layerNorm(h, T, D, model.lnfW, model.lnfB, eps);
  const positionLogits = opts.allPositions
    ? Array.from({ length: T }, (_, t) => unembed(model, hf, t * D))
    : null;
  const logits = positionLogits ? positionLogits[T - 1]! : unembed(model, hf, (T - 1) * D);
  return { T, attention, scores, logits, positionLogits, lensLogits, residualNorms, validation };
}
