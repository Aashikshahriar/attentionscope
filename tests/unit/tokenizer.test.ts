import { describe, expect, it } from 'vitest';
import { ByteLevelBpeTokenizer, type TokenizerJson } from '@/core/tokenizer/bpe';
import { readJson } from '../helpers';

interface Case {
  text: string;
  gpt2: number[];
  neox: number[];
  gpt2Decoded: string;
  neoxDecoded: string;
}

const fixture = readJson<{ cases: Case[] }>('tests/fixtures/tokenizer-cases.json');
const info = { name: 'test', repo: 'test', revision: 'test' };
const gpt2 = new ByteLevelBpeTokenizer(
  readJson<TokenizerJson>('public/tokenizers/gpt2/tokenizer.json'),
  info,
);
const neox = new ByteLevelBpeTokenizer(
  readJson<TokenizerJson>('public/models/pythia-14m/tokenizer.json'),
  info,
);

describe('byte-level BPE matches the Rust tokenizers implementation', () => {
  for (const c of fixture.cases) {
    it(`GPT-2: ${JSON.stringify(c.text).slice(0, 48)}`, () => {
      expect(gpt2.encodeIds(c.text)).toEqual(c.gpt2);
      expect(gpt2.decode(c.gpt2)).toBe(c.gpt2Decoded);
    });
    it(`GPT-NeoX: ${JSON.stringify(c.text).slice(0, 48)}`, () => {
      expect(neox.encodeIds(c.text)).toEqual(c.neox);
      expect(neox.decode(c.neox)).toBe(c.neoxDecoded);
    });
  }
});

describe('token offsets', () => {
  it('byte spans tile the UTF-8 input exactly', () => {
    for (const c of fixture.cases) {
      for (const tok of [gpt2, neox]) {
        const enc = tok.encode(c.text);
        const bytes = new TextEncoder().encode(enc.normalizedText);
        let cursor = 0;
        for (const t of enc.tokens) {
          expect(t.byteStart).toBe(cursor);
          cursor = t.byteEnd;
          const hex = Array.from(bytes.slice(t.byteStart, t.byteEnd), (b) =>
            b.toString(16).padStart(2, '0'),
          ).join('');
          expect(hex).toBe(t.bytesHex);
        }
        expect(cursor).toBe(bytes.length);
      }
    }
  });

  it('flags tokens that split a multi-byte character', () => {
    const enc = gpt2.encode('আমি');
    expect(enc.tokens.some((t) => t.partialCharacter)).toBe(true);
    for (const t of enc.tokens) {
      expect(t.u16Start).toBeLessThanOrEqual(t.u16End);
      expect(t.u16End).toBeLessThanOrEqual(enc.text.length);
    }
  });

  it('marks a prepended BOS as synthetic with an empty span', () => {
    const enc = neox.encode('Hi there', { prependBos: true, bos: { id: 0, text: '<|endoftext|>' } });
    expect(enc.tokens[0]).toMatchObject({ id: 0, special: true, synthetic: true, byteStart: 0, byteEnd: 0 });
    expect(enc.tokens[1]!.byteStart).toBe(0);
  });

  it('reports NFC normalization changes for GPT-NeoX', () => {
    const enc = neox.encode('Café');
    expect(enc.normalizationChanged).toBe(true);
    expect(enc.normalizedText).toBe('Café');
    expect(gpt2.encode('Café').normalizationChanged).toBe(false);
  });
});
