/// <reference lib="webworker" />
import { exposeWorker, withTransfer, type HandlerContext } from './rpc';
import { MODEL_CACHE_NAME, type LoadProgress, type ModelApi } from './modelApi';
import {
  loadNeoxModel,
  neoxForward,
  type ForwardOutput,
  type NeoxModel,
} from '@/core/model-adapters/gptNeox';
import { interventionDefinition, interventionLocation } from '@/core/model-adapters/interventions';
import { ByteLevelBpeTokenizer, type TokenizerJson } from '@/core/tokenizer/bpe';
import { logSoftmax, topK } from '@/core/attention/softmax';
import { ChecksumError, sha256Hex } from '@/core/hash';
import type { ComparisonResult, ModelManifest, Prediction } from '@/types/contracts';

let manifest: ModelManifest | null = null;
let model: NeoxModel | null = null;
let tokenizer: ByteLevelBpeTokenizer | null = null;
const baselineCache = new Map<string, ForwardOutput>();

function requireModel() {
  if (!manifest || !model || !tokenizer) throw new Error('Model is not loaded');
  return { manifest, model, tokenizer };
}

async function fetchBytes(
  url: string,
  expectedBytes: number | null,
  cache: boolean,
  onProgress: (loaded: number, total: number, fromCache: boolean) => void,
): Promise<{ buffer: ArrayBuffer; fromCache: boolean }> {
  let res: Response | undefined;
  let store: Cache | null = null;
  if (cache && 'caches' in self) {
    store = await caches.open(MODEL_CACHE_NAME);
    res = await store.match(url);
  }
  const fromCache = !!res;
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
    if (store) await store.put(url, res.clone());
  }
  // Content-Length may describe a compressed body; the manifest size is exact.
  const total = expectedBytes ?? (Number(res.headers.get('content-length')) || 0);
  if (!res.body) {
    const buffer = await res.arrayBuffer();
    onProgress(buffer.byteLength, total, fromCache);
    return { buffer, fromCache };
  }
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total, fromCache);
  }
  const joined = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    joined.set(c, o);
    o += c.byteLength;
  }
  return { buffer: joined.buffer, fromCache };
}

function predictions(logits: Float64Array, k: number, tok: ByteLevelBpeTokenizer): Prediction[] {
  const lp = logSoftmax(logits);
  return topK(logits, k).map((id) => ({
    id,
    text: safeDecode(tok, id),
    prob: Math.exp(lp[id]!),
    logit: logits[id]!,
  }));
}

function safeDecode(tok: ByteLevelBpeTokenizer, id: number): string {
  // The model's embedding matrix is padded beyond the tokenizer vocabulary.
  return id < tok.vocabSize && tok.idToToken[id] !== undefined ? tok.decode([id]) : `<unused ${id}>`;
}

async function baseline(ids: number[], checkpoint: () => Promise<void>) {
  const key = ids.join(',');
  const hit = baselineCache.get(key);
  if (hit) return hit;
  const { model } = requireModel();
  const out = await neoxForward(model, ids, { yieldControl: checkpoint });
  if (baselineCache.size > 8) baselineCache.clear();
  baselineCache.set(key, out);
  return out;
}

function checkLength(n: number) {
  const max = requireModel().manifest.runtime.maxSequenceLength;
  if (n > max) throw new Error(`Input is ${n} tokens; the in-browser limit for this model is ${max}.`);
  if (n < 1) throw new Error('Input is empty.');
}

exposeWorker<ModelApi>({
  async load({ manifestUrl, cache }, ctx: HandlerContext<LoadProgress>) {
    ctx.progress({ phase: 'manifest', loaded: 0, total: 1 });
    const mres = await fetch(manifestUrl, { cache: 'no-cache' });
    if (!mres.ok) throw new Error(`HTTP ${mres.status} fetching model manifest`);
    const m = (await mres.json()) as ModelManifest;
    const base = new URL('.', manifestUrl);
    const t0 = performance.now();
    const w = await fetchBytes(
      new URL(m.weights.file, base).href,
      m.weights.byteLength,
      cache,
      (loaded, total, fromCache) => ctx.progress({ phase: 'weights', loaded, total, fromCache }),
    );
    ctx.progress({ phase: 'tokenizer', loaded: 0, total: 1 });
    const tk = await fetchBytes(new URL(m.tokenizer.file, base).href, null, cache, () => undefined);
    const t1 = performance.now();
    ctx.progress({ phase: 'verifying', loaded: 0, total: 1 });
    const wsha = await sha256Hex(w.buffer);
    if (wsha !== m.weights.sha256) {
      if (cache && 'caches' in self) await caches.delete(MODEL_CACHE_NAME);
      throw new ChecksumError(m.weights.file, m.weights.sha256, wsha);
    }
    const tsha = await sha256Hex(tk.buffer);
    if (tsha !== m.tokenizer.sha256) throw new ChecksumError(m.tokenizer.file, m.tokenizer.sha256, tsha);
    const t2 = performance.now();
    ctx.progress({ phase: 'decoding', loaded: 0, total: 1 });
    model = loadNeoxModel(m, w.buffer);
    tokenizer = new ByteLevelBpeTokenizer(JSON.parse(new TextDecoder().decode(tk.buffer)) as TokenizerJson, {
      name: m.tokenizer.name,
      repo: m.source.repo,
      revision: m.source.revision,
    });
    manifest = m;
    baselineCache.clear();
    const t3 = performance.now();
    return {
      manifest: m,
      fromCache: w.fromCache,
      timings: { downloadMs: t1 - t0, verifyMs: t2 - t1, decodeMs: t3 - t2 },
      verified: { weightsSha256: wsha, tokenizerSha256: tsha },
    };
  },

  tokenize({ text, prependBos }) {
    const { manifest, tokenizer } = requireModel();
    return tokenizer.encode(text, { prependBos, bos: manifest.tokenizer.bosToken });
  },

  async run({ text, prependBos }, ctx) {
    const { manifest, model, tokenizer } = requireModel();
    const t0 = performance.now();
    const input = tokenizer.encode(text, { prependBos, bos: manifest.tokenizer.bosToken });
    const ids = input.tokens.map((t) => t.id);
    checkLength(ids.length);
    let layer = 0;
    const out = await neoxForward(model, ids, {
      collectScores: true,
      logitLens: true,
      allPositions: true,
      yieldControl: async () => {
        ctx.progress({ layer: ++layer });
        await ctx.checkpoint();
      },
    });
    baselineCache.set(ids.join(','), out);
    const result = {
      input,
      attention: out.attention,
      scores: out.scores!,
      predictions: {
        position: ids.length - 1,
        top: predictions(out.logits, 10, tokenizer),
        perPosition: out.positionLogits!.map((l) => ({
          top: predictions(l, 3, tokenizer).map(({ id, text, prob }) => ({ id, text, prob })),
        })),
      },
      logitLens: out.lensLogits!.map((l, i) => ({ layer: i, top: predictions(l, 5, tokenizer) })),
      residualNorms: out.residualNorms,
      durationMs: performance.now() - t0,
    };
    return withTransfer(result, [out.attention.buffer, out.scores!.buffer]);
  },

  async compare({ tokenIds, spec, targetId }, ctx) {
    const { manifest, model, tokenizer } = requireModel();
    checkLength(tokenIds.length);
    const t0 = performance.now();
    const base = await baseline(tokenIds, ctx.checkpoint);
    const out = await neoxForward(model, tokenIds, { intervention: spec, yieldControl: ctx.checkpoint });
    const lb = logSoftmax(base.logits);
    const li = logSoftmax(out.logits);
    const target = targetId ?? topK(base.logits, 1)[0]!;
    let kl = 0;
    for (let i = 0; i < lb.length; i++) kl += Math.exp(lb[i]!) * (lb[i]! - li[i]!);
    const def = interventionDefinition(spec.kind);
    const result: ComparisonResult = {
      spec,
      location: interventionLocation(spec),
      description: def.description,
      model: { id: manifest.id, revision: manifest.source.revision },
      inputTokenIds: tokenIds,
      position: tokenIds.length - 1,
      target: {
        id: target,
        text: safeDecode(tokenizer, target),
        baselineProb: Math.exp(lb[target]!),
        intervenedProb: Math.exp(li[target]!),
        baselineLogit: base.logits[target]!,
        intervenedLogit: out.logits[target]!,
        deltaLogProb: li[target]! - lb[target]!,
      },
      baselineTop: predictions(base.logits, 10, tokenizer),
      intervenedTop: predictions(out.logits, 10, tokenizer),
      klDivergence: Math.max(0, kl),
      validation: out.validation,
      durationMs: performance.now() - t0,
    };
    return result;
  },

  async sweep({ tokenIds, kind, targetId }, ctx) {
    const { model, tokenizer } = requireModel();
    checkLength(tokenIds.length);
    const t0 = performance.now();
    const base = await baseline(tokenIds, ctx.checkpoint);
    const lb = logSoftmax(base.logits);
    const target = targetId ?? topK(base.logits, 1)[0]!;
    const { layers: L, heads: H } = model.dims;
    const deltaLogProb: number[][] = [];
    const intervenedProb: number[][] = [];
    const klDivergence: number[][] = [];
    for (let l = 0; l < L; l++) {
      const d: number[] = [];
      const p: number[] = [];
      const k: number[] = [];
      for (let h = 0; h < H; h++) {
        const out = await neoxForward(model, tokenIds, { intervention: { kind, layer: l, head: h } });
        const li = logSoftmax(out.logits);
        let kl = 0;
        for (let i = 0; i < lb.length; i++) kl += Math.exp(lb[i]!) * (lb[i]! - li[i]!);
        d.push(li[target]! - lb[target]!);
        p.push(Math.exp(li[target]!));
        k.push(Math.max(0, kl));
        ctx.progress({ done: l * H + h + 1, total: L * H });
        await ctx.checkpoint();
      }
      deltaLogProb.push(d);
      intervenedProb.push(p);
      klDivergence.push(k);
    }
    return {
      kind,
      inputTokenIds: tokenIds,
      target: { id: target, text: safeDecode(tokenizer, target), baselineProb: Math.exp(lb[target]!) },
      deltaLogProb,
      intervenedProb,
      klDivergence,
      durationMs: performance.now() - t0,
    };
  },

  resolveToken({ text }) {
    const { tokenizer } = requireModel();
    const ids = tokenizer.encodeIds(text);
    return { ids, pieces: ids.map((i) => tokenizer.piece(i)) };
  },

  async clearCache() {
    if (!('caches' in self)) return false;
    return caches.delete(MODEL_CACHE_NAME);
  },
});
