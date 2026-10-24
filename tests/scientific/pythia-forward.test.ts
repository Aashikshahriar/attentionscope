import { beforeAll, describe, expect, it } from 'vitest';
import { loadNeoxModel, neoxForward, type NeoxModel } from '@/core/model-adapters/gptNeox';
import { sha256Hex } from '@/core/hash';
import type { InterventionKind, ModelManifest } from '@/types/contracts';
import { maxAbsDiff, readArrayBuffer, readJson } from '../helpers';

interface Intervention {
  spec: { kind: InterventionKind; layer: number; head: number };
  logitsAtProbe: number[];
  attentionLayer3Head0Row?: number[];
  patchedAttention?: number[][];
}
interface Case {
  text: string;
  prependBos: boolean;
  ids: number[];
  probeIds: number[];
  logitsAtProbe: number[];
  top10: number[];
  attention: number[][][][];
  scoresLayer0Head0: (number | null)[][];
  logitLens: { layer: number; top5: number[]; logitsAtProbe: number[] }[];
  interventions: Intervention[];
}

const manifest = readJson<ModelManifest>('public/models/pythia-14m/manifest.json');
const fixture = readJson<{ cases: Case[] }>('tests/fixtures/pythia-reference.json');
let model: NeoxModel;
let weights: ArrayBuffer;

// Reference: transformers GPT-NeoX in exact float64. Our implementation keeps
// weights in float32 (exact copies of float16) and activations in float64, so
// the only differences are floating-point summation order.
const LOGIT_TOL = 1e-8;
// Attention is returned as Float32Array (storage rounding <= 6e-8 for values <= 1).
const ATTN_TOL = 1e-7;

beforeAll(() => {
  weights = readArrayBuffer('public/models/pythia-14m/weights.bin');
  model = loadNeoxModel(manifest, weights);
});

const pick = (logits: Float64Array, ids: number[]) => ids.map((i) => logits[i]!);
const order = (logits: Float64Array, k: number) =>
  Array.from(logits.keys())
    .sort((a, b) => logits[b]! - logits[a]!)
    .slice(0, k);

describe('Pythia-14m browser implementation vs transformers (float64)', () => {
  it('weights match the manifest checksum', async () => {
    expect(await sha256Hex(weights)).toBe(manifest.weights.sha256);
  });

  for (const c of fixture.cases) {
    describe(JSON.stringify(c.text), () => {
      it('reproduces logits, attention probabilities and pre-softmax scores', async () => {
        const out = await neoxForward(model, c.ids, { collectScores: true, logitLens: true });
        expect(maxAbsDiff(pick(out.logits, c.probeIds), c.logitsAtProbe)).toBeLessThan(LOGIT_TOL);
        expect(order(out.logits, 10)).toEqual(c.top10);
        expect(maxAbsDiff(out.attention, c.attention.flat(3))).toBeLessThan(ATTN_TOL);
        const T = c.ids.length;
        // scores are also stored as float32: compare relative to magnitude
        const ref = c.scoresLayer0Head0.flat();
        out.scores!.subarray(0, T * T).forEach((v, x) => {
          const r = ref[x];
          if (r === null || r === undefined) expect(v).toBe(Number.NEGATIVE_INFINITY);
          else expect(Math.abs(v - r)).toBeLessThan(1e-7 * Math.max(1, Math.abs(r)));
        });
        c.logitLens.forEach((lens, l) => {
          expect(maxAbsDiff(pick(out.lensLogits![l]!, c.probeIds), lens.logitsAtProbe)).toBeLessThan(
            LOGIT_TOL,
          );
          expect(order(out.lensLogits![l]!, 5)).toEqual(lens.top5);
        });
      });

      for (const iv of c.interventions) {
        it(`intervention ${iv.spec.kind} @ ${iv.spec.layer}.${iv.spec.head} matches PyTorch hooks`, async () => {
          const base = await neoxForward(model, c.ids);
          const out = await neoxForward(model, c.ids, { intervention: iv.spec });
          expect(maxAbsDiff(pick(out.logits, c.probeIds), iv.logitsAtProbe)).toBeLessThan(LOGIT_TOL);
          const T = c.ids.length;
          const size = T * T;
          const H = model.dims.heads;
          // Layers before (and, for ablations, at) the intervention are untouched.
          const untouched = iv.spec.kind === 'zero-head-output' ? iv.spec.layer + 1 : iv.spec.layer;
          for (let x = 0; x < untouched * H * size; x++) expect(out.attention[x]).toBe(base.attention[x]);
          // Downstream attention changes (the intervention propagates).
          const after = (iv.spec.layer + 1) * H * size;
          expect(maxAbsDiff(out.attention.subarray(after), base.attention.subarray(after))).toBeGreaterThan(
            0,
          );
          if (iv.patchedAttention) {
            const o = (iv.spec.layer * H + iv.spec.head) * size;
            expect(maxAbsDiff(out.attention.subarray(o, o + size), iv.patchedAttention.flat())).toBeLessThan(
              ATTN_TOL,
            );
            expect(out.validation?.passed).toBe(true);
          }
          if (iv.attentionLayer3Head0Row) {
            const o = (3 * H + 0) * size + (T - 1) * T;
            expect(maxAbsDiff(out.attention.subarray(o, o + T), iv.attentionLayer3Head0Row)).toBeLessThan(
              ATTN_TOL,
            );
          }
        });
      }
    });
  }

  it('zeroing a head changes only that head’s contribution at its layer', async () => {
    const c = fixture.cases[0]!;
    const spec = { kind: 'zero-head-output' as const, layer: 5, head: 3 };
    const a = await neoxForward(model, c.ids);
    const b = await neoxForward(model, c.ids, { intervention: spec });
    // Last layer: every attention pattern is identical (patterns are computed before O_h),
    // yet the logits differ because head 3's output was removed.
    expect(maxAbsDiff(a.attention, b.attention)).toBe(0);
    expect(maxAbsDiff(a.logits, b.logits)).toBeGreaterThan(1e-6);
  });

  it('is deterministic', async () => {
    const ids = fixture.cases[1]!.ids;
    const a = await neoxForward(model, ids);
    const b = await neoxForward(model, ids);
    expect(maxAbsDiff(a.logits, b.logits)).toBe(0);
  });

  it('rejects out-of-range inputs and intervention targets', async () => {
    await expect(neoxForward(model, [999999])).rejects.toThrow(/outside vocabulary/);
    await expect(
      neoxForward(model, [0, 1], { intervention: { kind: 'zero-head-output', layer: 6, head: 0 } }),
    ).rejects.toThrow(/does not exist/);
  });
});
