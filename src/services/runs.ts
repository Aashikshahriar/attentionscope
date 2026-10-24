import type { AttentionRun, TokenizedInput } from '@/types/contracts';
import type { DemoManifest, DemoManifestEntry, TokenizerManifestEntry } from '@/types/artifacts';
import type { LiveRunResult, LoadedModel } from '@/workers/modelApi';
import { assetUrl } from './paths';
import { dataWorker } from './workers';
import { APP_INFO, nowIso, randomId } from './provenance';

let demoManifest: Promise<DemoManifest> | null = null;
let tokenizerManifest: Promise<TokenizerManifestEntry[]> | null = null;

async function getJson<T>(rel: string): Promise<T> {
  const res = await fetch(assetUrl(rel));
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${rel}`);
  return (await res.json()) as T;
}

export function fetchDemoManifest(): Promise<DemoManifest> {
  demoManifest ??= getJson<DemoManifest>('demos/manifest.json').catch((e: unknown) => {
    demoManifest = null;
    throw e;
  });
  return demoManifest;
}

export function fetchTokenizerManifest(): Promise<TokenizerManifestEntry[]> {
  tokenizerManifest ??= getJson<{ tokenizers: TokenizerManifestEntry[] }>('tokenizers/manifest.json')
    .then((m) => m.tokenizers)
    .catch((e: unknown) => {
      tokenizerManifest = null;
      throw e;
    });
  return tokenizerManifest;
}

export interface LoadedDemo {
  run: AttentionRun;
  bytes: number;
  fetchMs: number;
  verifyMs: number;
}

/** Fetch, checksum-verify and validate a precomputed demo in the data worker. */
export async function loadDemoRun(entry: DemoManifestEntry, signal?: AbortSignal): Promise<LoadedDemo> {
  const {
    artifact: a,
    attention,
    bytes,
    fetchMs,
    verifyMs,
  } = await dataWorker().call(
    'loadDemo',
    {
      jsonUrl: assetUrl(`demos/${entry.file}`),
      binUrl: assetUrl(`demos/${entry.attentionFile}`),
      jsonSha256: entry.sha256,
    },
    { signal },
  );
  const input: TokenizedInput = {
    text: a.input.text,
    tokens: a.input.tokens,
    normalization: 'none',
    normalizedText: a.input.text,
    normalizationChanged: false,
  };
  const run: AttentionRun = {
    id: `demo-${a.id}`,
    title: a.title,
    source: {
      kind: 'precomputed',
      demoId: a.id,
      artifactFile: `demos/${entry.file}`,
      artifactSha256: entry.sha256,
      attentionSha256: a.attention.sha256,
    },
    model: {
      id: a.model.id,
      displayName: a.model.displayName,
      repo: a.model.repo,
      revision: a.model.revision,
      url: a.model.url,
      license: a.model.license,
      citation: a.model.citation,
      architecture: a.model.architecture,
    },
    tokenizer: a.tokenizer,
    input,
    attention: {
      shape: a.attention.shape,
      layout: ['layer', 'head', 'query', 'key'],
      kind: 'post-softmax',
      mask: 'causal',
      data: attention,
    },
    predictions: a.predictions,
    logitLens: a.logitLens,
    residualNorms: a.residualNorms,
    ablation: a.ablation,
    suggestedView: a.suggestedView,
    summary: a.summary,
    literature: a.literature,
    provenance: {
      app: APP_INFO,
      precomputed: true,
      generator: `${a.generator.script} — ${a.generator.framework}, ${a.generator.attnImplementation} attention, ${a.generator.dtype}, ${a.generator.device}, ${a.generator.mode}`,
      computePrecision: a.generator.dtype,
      storedPrecision: a.attention.dtype,
      deterministic: a.generator.deterministicAlgorithms,
      seed: a.generator.seed,
      computedAt: a.generator.generatedAt,
      environment: a.generator.environment,
    },
  };
  return { run, bytes, fetchMs, verifyMs };
}

export function liveRunToAttentionRun(
  r: LiveRunResult,
  loaded: LoadedModel,
  prependBos: boolean,
): AttentionRun {
  const m = loaded.manifest;
  const T = r.input.tokens.length;
  return {
    id: randomId('live'),
    title: 'Live run',
    source: { kind: 'live', modelId: m.id, weightsSha256: loaded.verified.weightsSha256 },
    model: {
      id: m.id,
      displayName: m.displayName,
      repo: m.source.repo,
      revision: m.source.revision,
      url: m.source.url,
      license: m.source.license,
      citation: m.source.citation,
      architecture: m.architecture,
    },
    tokenizer: {
      name: m.tokenizer.name,
      repo: m.source.repo,
      revision: m.source.revision,
      prependBos,
      bosToken: m.tokenizer.bosToken,
    },
    input: r.input,
    attention: {
      shape: {
        batch: 1,
        layers: m.architecture.numLayers,
        heads: m.architecture.numHeads,
        queryLength: T,
        keyLength: T,
      },
      layout: ['layer', 'head', 'query', 'key'],
      kind: 'post-softmax',
      mask: 'causal',
      data: r.attention,
      scores: r.scores,
    },
    predictions: r.predictions,
    logitLens: {
      position: T - 1,
      method:
        'embed_out(final_layer_norm(residual stream after layer l)); the final layer matches the model output',
      layers: r.logitLens,
    },
    residualNorms: {
      method:
        'L2 norm of the residual stream: index 0 = token embedding (rotary models have no position embedding), index l+1 = after layer l',
      values: r.residualNorms,
    },
    provenance: {
      app: APP_INFO,
      precomputed: false,
      generator: `AttentionScope GPT-NeoX implementation (in-browser, Web Worker) on ${m.source.repo}@${m.source.revision.slice(0, 7)}`,
      computePrecision: 'float64 activations, float16 weights (exact)',
      storedPrecision: 'float32',
      deterministic: true,
      seed: null,
      computedAt: nowIso(),
    },
  };
}
