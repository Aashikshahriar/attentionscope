import type { TokenRecord, TokenizedInput, TokenizerInfo } from '@/types/contracts';

/**
 * Byte-level BPE tokenizer that reads a Hugging Face `tokenizer.json`
 * (GPT-2 / GPT-NeoX family) and reproduces the Rust `tokenizers` pipeline:
 *
 *   1. split out added tokens that match the raw text (`normalized: false`)
 *   2. normalize the remaining text (NFC when the tokenizer declares it)
 *   3. split out added tokens that match normalized text (`normalized: true`)
 *   4. ByteLevel pre-tokenization with the GPT-2 regex
 *   5. map UTF-8 bytes to the byte-level alphabet and apply BPE merges by rank
 *
 * Token ids are verified against the Rust implementation in
 * tests/unit/tokenizer.test.ts. Offsets are computed from the exact byte stream.
 */

export interface AddedTokenJson {
  id: number;
  content: string;
  special: boolean;
  normalized?: boolean;
  lstrip?: boolean;
  rstrip?: boolean;
  single_word?: boolean;
}

export interface TokenizerJson {
  added_tokens: AddedTokenJson[];
  normalizer: { type: string } | null;
  pre_tokenizer: { type: string; add_prefix_space?: boolean; use_regex?: boolean } | null;
  model: { type?: string; vocab: Record<string, number>; merges: (string | [string, string])[] };
}

const GPT2_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function bytesToUnicode(): string[] {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const table = new Array<string>(256);
  bs.forEach((b, i) => (table[b] = String.fromCharCode(cs[i]!)));
  return table;
}

const BYTE_ENCODER = bytesToUnicode();
const BYTE_DECODER = new Map<string, number>(BYTE_ENCODER.map((ch, b) => [ch, b]));
const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: false });

interface Segment {
  text: string;
  addedId?: number;
  special?: boolean;
}

/** Leftmost-longest split of `text` on the given literal tokens. */
function splitOnAdded(text: string, tokens: AddedTokenJson[]): Segment[] {
  if (!tokens.length || !text) return text ? [{ text }] : [];
  const out: Segment[] = [];
  let buf = '';
  let i = 0;
  while (i < text.length) {
    let best: AddedTokenJson | null = null;
    for (const t of tokens) {
      if (t.content.length > (best?.content.length ?? 0) && text.startsWith(t.content, i)) best = t;
    }
    if (best) {
      if (buf) out.push({ text: buf });
      buf = '';
      out.push({ text: best.content, addedId: best.id, special: best.special });
      i += best.content.length;
    } else {
      buf += text[i];
      i++;
    }
  }
  if (buf) out.push({ text: buf });
  return out;
}

export class ByteLevelBpeTokenizer {
  readonly vocab: Map<string, number>;
  readonly idToToken: string[];
  private readonly ranks: Map<string, number>;
  private readonly rawAdded: AddedTokenJson[];
  private readonly normalizedAdded: AddedTokenJson[];
  private readonly specialIds: Set<number>;
  private readonly addedIds: Set<number>;
  private readonly normalizer: 'none' | 'NFC';
  private readonly cache = new Map<string, number[]>();

  constructor(
    json: TokenizerJson,
    readonly info: Omit<TokenizerInfo, 'prependBos'>,
  ) {
    if (json.model.type && json.model.type !== 'BPE')
      throw new Error(`unsupported model type ${json.model.type}`);
    if (json.pre_tokenizer?.type !== 'ByteLevel')
      throw new Error('only ByteLevel pre-tokenizers are supported');
    if (json.pre_tokenizer.add_prefix_space) throw new Error('add_prefix_space=true is not supported');
    if (json.normalizer && json.normalizer.type !== 'NFC') {
      throw new Error(`unsupported normalizer ${json.normalizer.type}`);
    }
    this.normalizer = json.normalizer ? 'NFC' : 'none';
    this.vocab = new Map(Object.entries(json.model.vocab));
    for (const t of json.added_tokens) this.vocab.set(t.content, t.id);
    this.idToToken = [];
    for (const [tok, id] of this.vocab) this.idToToken[id] = tok;
    this.ranks = new Map();
    json.model.merges.forEach((m, i) => this.ranks.set(Array.isArray(m) ? `${m[0]} ${m[1]}` : m, i));
    for (const t of json.added_tokens) {
      if (t.lstrip || t.rstrip || t.single_word)
        throw new Error(`added token options unsupported: ${t.content}`);
    }
    this.rawAdded = json.added_tokens.filter(
      (t) => t.normalized === false || (t.special && t.normalized !== true),
    );
    this.normalizedAdded = json.added_tokens.filter((t) => !this.rawAdded.includes(t));
    this.specialIds = new Set(json.added_tokens.filter((t) => t.special).map((t) => t.id));
    this.addedIds = new Set(json.added_tokens.map((t) => t.id));
  }

  get vocabSize(): number {
    return this.idToToken.length;
  }

  isSpecial(id: number): boolean {
    return this.specialIds.has(id);
  }

  piece(id: number): string {
    const p = this.idToToken[id];
    if (p === undefined) throw new RangeError(`token id ${id} not in vocabulary`);
    return p;
  }

  /** Raw bytes a token stands for. Added tokens map to the UTF-8 of their content. */
  tokenBytes(id: number): Uint8Array {
    const p = this.piece(id);
    if (this.addedIds.has(id)) return encoder.encode(p);
    const out = new Uint8Array(p.length);
    for (let i = 0; i < p.length; i++) {
      const b = BYTE_DECODER.get(p[i]!);
      if (b === undefined) return encoder.encode(p);
      out[i] = b;
    }
    return out;
  }

  decode(ids: number[]): string {
    const parts = ids.map((id) => this.tokenBytes(id));
    const total = parts.reduce((n, p) => n + p.length, 0);
    const buf = new Uint8Array(total);
    let o = 0;
    for (const p of parts) {
      buf.set(p, o);
      o += p.length;
    }
    return decoder.decode(buf);
  }

  private bpe(word: string): number[] {
    const hit = this.cache.get(word);
    if (hit) return hit;
    let symbols = Array.from(word);
    while (symbols.length > 1) {
      let bestRank = Number.POSITIVE_INFINITY;
      let bestIdx = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = this.ranks.get(`${symbols[i]} ${symbols[i + 1]}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) break;
      const a = symbols[bestIdx]!;
      const b = symbols[bestIdx + 1]!;
      const merged: string[] = [];
      for (let i = 0; i < symbols.length;) {
        if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
          merged.push(a + b);
          i += 2;
        } else {
          merged.push(symbols[i]!);
          i++;
        }
      }
      symbols = merged;
    }
    const ids = symbols.map((s) => {
      const id = this.vocab.get(s);
      if (id === undefined) throw new Error(`BPE produced out-of-vocabulary symbol ${JSON.stringify(s)}`);
      return id;
    });
    if (this.cache.size > 20000) this.cache.clear();
    this.cache.set(word, ids);
    return ids;
  }

  private encodeOrdinary(text: string): number[] {
    const ids: number[] = [];
    for (const match of text.matchAll(GPT2_PATTERN)) {
      const bytes = encoder.encode(match[0]);
      let word = '';
      for (const b of bytes) word += BYTE_ENCODER[b];
      ids.push(...this.bpe(word));
    }
    return ids;
  }

  /** Token ids exactly as `tokenizer(text, add_special_tokens=False)` would return them. */
  encodeIds(text: string): number[] {
    return this.segment(text).ids;
  }

  private segment(text: string): { ids: number[]; normalizedText: string } {
    const ids: number[] = [];
    let normalizedText = '';
    for (const raw of splitOnAdded(text, this.rawAdded)) {
      if (raw.addedId !== undefined) {
        ids.push(raw.addedId);
        normalizedText += raw.text;
        continue;
      }
      const norm = this.normalizer === 'NFC' ? raw.text.normalize('NFC') : raw.text;
      normalizedText += norm;
      for (const seg of splitOnAdded(norm, this.normalizedAdded)) {
        if (seg.addedId !== undefined) ids.push(seg.addedId);
        else ids.push(...this.encodeOrdinary(seg.text));
      }
    }
    return { ids, normalizedText };
  }

  encode(
    text: string,
    opts: { prependBos?: boolean; bos?: { id: number; text: string } } = {},
  ): TokenizedInput {
    const { ids, normalizedText } = this.segment(text);
    const allIds = opts.prependBos && opts.bos ? [opts.bos.id, ...ids] : ids;
    return {
      text,
      normalization: this.normalizer,
      normalizedText,
      normalizationChanged: normalizedText !== text,
      tokens: this.records(normalizedText, allIds, opts.prependBos && opts.bos ? 1 : 0),
    };
  }

  /** Token records with byte and UTF-16 spans into `text` (which must be the normalized text). */
  records(text: string, ids: number[], syntheticPrefix = 0): TokenRecord[] {
    const encoded = encoder.encode(text);
    // byte index -> UTF-16 index at character boundaries
    const byteToU16 = new Map<number, number>();
    let b = 0;
    let u = 0;
    for (const ch of text) {
      byteToU16.set(b, u);
      b += encoder.encode(ch).length;
      u += ch.length;
    }
    byteToU16.set(b, u);
    const floorU16 = (x: number) => {
      while (!byteToU16.has(x)) x--;
      return byteToU16.get(x)!;
    };
    const ceilU16 = (x: number) => {
      while (!byteToU16.has(x)) x++;
      return byteToU16.get(x)!;
    };
    let cursor = 0;
    const records = ids.map((id, index): TokenRecord => {
      const bytes = this.tokenBytes(id);
      const synthetic = index < syntheticPrefix;
      const start = cursor;
      const end = synthetic ? cursor : cursor + bytes.length;
      if (!synthetic) {
        for (let k = 0; k < bytes.length; k++) {
          if (encoded[start + k] !== bytes[k]) throw new Error('token bytes do not reproduce the input');
        }
        cursor = end;
      }
      return {
        index,
        id,
        piece: this.piece(id),
        text: decoder.decode(bytes),
        bytesHex: Array.from(bytes, (x) => x.toString(16).padStart(2, '0')).join(''),
        byteStart: start,
        byteEnd: end,
        u16Start: floorU16(start),
        u16End: ceilU16(end),
        special: this.specialIds.has(id),
        synthetic,
        partialCharacter: !byteToU16.has(start) || !byteToU16.has(end),
      };
    });
    if (cursor !== encoded.length) throw new Error('tokens do not cover the full input');
    return records;
  }
}
