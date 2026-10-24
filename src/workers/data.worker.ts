/// <reference lib="webworker" />
import { exposeWorker, withTransfer } from './rpc';
import type { DataApi } from './dataApi';
import type { AttentionTensor } from '@/types/contracts';
import type { DemoArtifact } from '@/types/artifacts';
import { verifySha256 } from '@/core/hash';
import { assertShape, validateTensor } from '@/core/attention/tensor';
import { attentionRollout, computeAllHeadStats, similarityToHead } from '@/core/metrics/headMetrics';
import { ByteLevelBpeTokenizer, type TokenizerJson } from '@/core/tokenizer/bpe';

const tensors = new Map<string, AttentionTensor>();
const tokenizers = new Map<string, ByteLevelBpeTokenizer>();

async function fetchBuffer(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  return res.arrayBuffer();
}

function tensor(key: string): AttentionTensor {
  const t = tensors.get(key);
  if (!t) throw new Error(`no tensor registered for ${key}`);
  return t;
}

exposeWorker<DataApi>({
  async loadDemo({ jsonUrl, binUrl, jsonSha256 }) {
    const t0 = performance.now();
    const json = await fetchBuffer(jsonUrl);
    const t1 = performance.now();
    await verifySha256(jsonUrl.split('/').pop() ?? 'demo.json', json, jsonSha256);
    const artifact = JSON.parse(new TextDecoder().decode(json)) as DemoArtifact;
    if (artifact.precomputed !== true) throw new Error('artifact is not labelled as precomputed');
    if (artifact.attention.dtype !== 'float32' || artifact.attention.byteOrder !== 'little-endian') {
      throw new Error(`unsupported attention encoding ${artifact.attention.dtype}`);
    }
    const t2 = performance.now();
    const bin = await fetchBuffer(binUrl);
    const t3 = performance.now();
    await verifySha256(artifact.attention.file, bin, artifact.attention.sha256);
    const data = new Float32Array(bin);
    assertShape(artifact.attention.shape, data.length);
    if (artifact.input.tokens.length !== artifact.attention.shape.queryLength) {
      throw new Error('token count does not match the attention tensor');
    }
    const t: AttentionTensor = {
      shape: artifact.attention.shape,
      layout: ['layer', 'head', 'query', 'key'],
      kind: 'post-softmax',
      mask: 'causal',
      data,
    };
    const v = validateTensor(t, 1e-4);
    if (!v.passed) throw new Error('attention tensor failed normalization/causality validation');
    const t4 = performance.now();
    return withTransfer(
      {
        artifact,
        attention: data,
        bytes: json.byteLength + bin.byteLength,
        fetchMs: t1 - t0 + (t3 - t2),
        verifyMs: t2 - t1 + (t4 - t3),
      },
      [data.buffer],
    );
  },

  registerTensor({ key, shape, data }) {
    assertShape(shape, data.length);
    tensors.set(key, {
      shape,
      layout: ['layer', 'head', 'query', 'key'],
      kind: 'post-softmax',
      mask: 'causal',
      data,
    });
    if (tensors.size > 6) tensors.delete(tensors.keys().next().value!);
    return true;
  },

  headStats({ key }) {
    return computeAllHeadStats(tensor(key));
  },

  similarity({ key, layer, head, metric }) {
    return similarityToHead(tensor(key), layer, head, metric);
  },

  rollout({ key, residualWeight }) {
    const r = attentionRollout(tensor(key), residualWeight);
    return withTransfer(r, [r.buffer]);
  },

  async loadTokenizer({ id, url, sha256, name, repo, revision }) {
    const t0 = performance.now();
    if (!tokenizers.has(id)) {
      const buf = await fetchBuffer(url);
      await verifySha256(`${id}/tokenizer.json`, buf, sha256);
      const json = JSON.parse(new TextDecoder().decode(buf)) as TokenizerJson;
      tokenizers.set(id, new ByteLevelBpeTokenizer(json, { name, repo, revision }));
    }
    return { vocabSize: tokenizers.get(id)!.vocabSize, ms: performance.now() - t0 };
  },

  tokenize({ id, text, prependBos, bos }) {
    const tok = tokenizers.get(id);
    if (!tok) throw new Error(`tokenizer ${id} is not loaded`);
    const t0 = performance.now();
    const out = tok.encode(text, { prependBos, bos });
    return { ...out, ms: performance.now() - t0 };
  },
});
