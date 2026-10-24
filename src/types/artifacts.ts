import type {
  AttentionShape,
  LogitLensLayer,
  ModelInfo,
  PrecomputedAblation,
  PredictionSet,
  TokenRecord,
  TokenizerInfo,
} from './contracts';

export interface DemoManifestEntry {
  id: string;
  title: string;
  category: string;
  summary: string;
  model: string;
  modelDisplayName: string;
  tokens: number;
  file: string;
  sha256: string;
  attentionFile: string;
  attentionSha256: string;
  bytes: number;
}

export interface DemoManifest {
  schemaVersion: 1;
  generatedAt: string;
  generator: string;
  notice: string;
  demos: DemoManifestEntry[];
}

/** JSON half of a precomputed demo (the attention tensor lives in a separate binary file). */
export interface DemoArtifact {
  schemaVersion: 1;
  id: string;
  title: string;
  category: string;
  summary: string;
  literature: string;
  suggestedView: {
    layer: number;
    head: number;
    queryIndex: number;
    keyIndex: number;
    weight: number;
    criterion: string;
  };
  precomputed: true;
  model: Omit<ModelInfo, 'source'>;
  tokenizer: Omit<TokenizerInfo, 'prependBos'> & { prependBos: boolean };
  generator: {
    script: string;
    framework: string;
    attnImplementation: string;
    dtype: string;
    device: string;
    mode: string;
    deterministicAlgorithms: boolean;
    seed: number;
    generatedAt: string;
    environment: Record<string, string>;
  };
  input: { text: string; tokens: TokenRecord[] };
  attention: {
    file: string;
    dtype: 'float32';
    byteOrder: 'little-endian';
    layout: ['layer', 'head', 'query', 'key'];
    shape: AttentionShape;
    kind: 'post-softmax';
    mask: 'causal';
    normalization: string;
    sha256: string;
    checks: { maxRowSumError: number; maxUpperTriangle: number };
  };
  predictions: Required<Pick<PredictionSet, 'position' | 'top' | 'target' | 'perPosition'>>;
  logitLens: { position: number; method: string; layers: LogitLensLayer[] };
  residualNorms: { method: string; values: number[][] };
  ablation: PrecomputedAblation;
}

export interface TokenizerManifestEntry {
  id: string;
  name: string;
  usedBy: string;
  repo: string;
  revision: string;
  license: string;
  path: string;
  sha256: string;
  bos: { id: number; text: string };
}
