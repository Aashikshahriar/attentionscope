import { describe, expect, it } from 'vitest';
import { sha256Hex } from '@/core/hash';
import { validateTensor, headMatrix } from '@/core/attention/tensor';
import type { AttentionTensor, TokenRecord } from '@/types/contracts';
import { ByteLevelBpeTokenizer, type TokenizerJson } from '@/core/tokenizer/bpe';
import { readArrayBuffer, readJson } from '../helpers';

interface ManifestEntry {
  id: string;
  file: string;
  sha256: string;
  attentionFile: string;
  attentionSha256: string;
  tokens: number;
}
interface DemoJson {
  precomputed: boolean;
  input: { text: string; tokens: TokenRecord[] };
  attention: { shape: AttentionTensor['shape']; sha256: string; dtype: string; layout: string[] };
  ablation: { heads: unknown[][] };
  suggestedView: { layer: number; head: number; queryIndex: number; keyIndex: number; weight: number };
  model: { architecture: { numLayers: number; numHeads: number } };
}

const manifest = readJson<{ demos: ManifestEntry[] }>('public/demos/manifest.json');
const gpt2 = new ByteLevelBpeTokenizer(readJson<TokenizerJson>('public/tokenizers/gpt2/tokenizer.json'), {
  name: 'gpt2',
  repo: '',
  revision: '',
});

describe('bundled precomputed demo artifacts', () => {
  for (const entry of manifest.demos) {
    it(`${entry.id}: checksums, shape, normalization, causality and tokenization`, async () => {
      const raw = readArrayBuffer(`public/demos/${entry.file}`);
      expect(await sha256Hex(raw)).toBe(entry.sha256);
      const demo = JSON.parse(new TextDecoder().decode(raw)) as DemoJson;
      expect(demo.precomputed).toBe(true);
      expect(demo.attention.layout).toEqual(['layer', 'head', 'query', 'key']);
      expect(demo.attention.dtype).toBe('float32');
      const bin = readArrayBuffer(`public/demos/${entry.attentionFile}`);
      expect(await sha256Hex(bin)).toBe(demo.attention.sha256);
      const tensor: AttentionTensor = {
        shape: demo.attention.shape,
        layout: ['layer', 'head', 'query', 'key'],
        kind: 'post-softmax',
        mask: 'causal',
        data: new Float32Array(bin),
      };
      const v = validateTensor(tensor, 1e-5);
      expect(v.passed).toBe(true);
      const { layers, heads, queryLength } = tensor.shape;
      expect(layers).toBe(demo.model.architecture.numLayers);
      expect(heads).toBe(demo.model.architecture.numHeads);
      expect(demo.input.tokens).toHaveLength(queryLength);
      expect(demo.ablation.heads).toHaveLength(layers);
      expect(demo.ablation.heads[0]).toHaveLength(heads);

      // Token ids in the artifact equal our in-browser GPT-2 tokenizer's output (after the BOS).
      expect(demo.input.tokens.slice(1).map((t) => t.id)).toEqual(gpt2.encodeIds(demo.input.text));
      expect(demo.input.tokens[0]).toMatchObject({ special: true, synthetic: true });

      // The suggested view really is the argmax for its stated criterion.
      const sv = demo.suggestedView;
      let best = -1;
      for (let l = 0; l < layers; l++)
        for (let h = 0; h < heads; h++) {
          best = Math.max(best, headMatrix(tensor, l, h)[sv.queryIndex * queryLength + sv.keyIndex]!);
        }
      expect(headMatrix(tensor, sv.layer, sv.head)[sv.queryIndex * queryLength + sv.keyIndex]).toBe(best);
      expect(best).toBeCloseTo(sv.weight, 6);
    });
  }
});
