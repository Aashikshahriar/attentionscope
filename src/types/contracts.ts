/**
 * Typed contracts shared by artifacts, workers, stores and exports.
 * Tensor shapes are always carried explicitly; nothing infers them from data length.
 */

export type AttentionVariant = 'multi-head' | 'grouped-query' | 'multi-query';

export interface ModelArchitecture {
  family: 'gpt2' | 'gpt-neox';
  numLayers: number;
  numHeads: number;
  /** Number of distinct key/value heads. Equal to numHeads for standard multi-head attention. */
  numKeyValueHeads: number;
  attentionType: AttentionVariant;
  hiddenSize: number;
  headDim: number;
  intermediateSize?: number;
  vocabSize?: number;
  maxPositions?: number;
  positional: string;
  parallelResidual?: boolean;
  layerNormEps?: number;
  activation?: string;
  attentionScale: string;
  tiedEmbeddings?: boolean;
  rotaryDims?: number;
  rotaryBase?: number;
  qkvLayout?: string;
}

export interface ModelSource {
  repo: string;
  revision: string;
  url: string;
  license: string;
  citation: string;
}

export interface ModelInfo extends ModelSource {
  id: string;
  displayName: string;
  architecture: ModelArchitecture;
}

export interface TensorEntry {
  name: string;
  shape: number[];
  byteOffset: number;
  elementCount: number;
}

export interface ModelManifest {
  schemaVersion: 1;
  id: string;
  displayName: string;
  kind: 'browser-inference';
  description: string;
  source: ModelSource;
  architecture: ModelArchitecture;
  tokenizer: {
    file: string;
    sha256: string;
    type: 'byte-level-bpe';
    name: string;
    bosToken: { id: number; text: string };
  };
  weights: {
    file: string;
    dtype: 'float16';
    byteOrder: 'little-endian';
    byteLength: number;
    sha256: string;
    tensors: TensorEntry[];
  };
  runtime: { maxSequenceLength: number; computePrecision: string; deterministic: boolean };
  capabilities: Record<string, boolean>;
  validation?: Record<string, unknown>;
  exportedAt: string;
  exportEnvironment: Record<string, string>;
}

export interface TokenRecord {
  index: number;
  id: number;
  /** Vocabulary entry as stored by the tokenizer (byte-level alphabet, e.g. "Ġthe"). */
  piece: string;
  /** UTF-8 decoding of this token's bytes (U+FFFD where a character is split). */
  text: string;
  bytesHex: string;
  /** Byte span in the UTF-8 encoding of the (normalized) input. */
  byteStart: number;
  byteEnd: number;
  /** UTF-16 span in the (normalized) input, widened to whole characters. */
  u16Start: number;
  u16End: number;
  special: boolean;
  /** True for tokens not present in the text (e.g. a prepended BOS). */
  synthetic: boolean;
  /** True when the token boundary falls inside a multi-byte character. */
  partialCharacter: boolean;
}

export interface TokenizerInfo {
  name: string;
  repo: string;
  revision: string;
  prependBos: boolean;
  bosToken?: { id: number; text: string };
}

export interface TokenizedInput {
  text: string;
  tokens: TokenRecord[];
  /** Normalization applied before tokenization; offsets refer to the normalized text. */
  normalization?: 'none' | 'NFC';
  normalizedText?: string;
  normalizationChanged?: boolean;
}

export interface AttentionShape {
  batch: 1;
  layers: number;
  heads: number;
  queryLength: number;
  keyLength: number;
}

export interface AttentionTensor {
  shape: AttentionShape;
  layout: ['layer', 'head', 'query', 'key'];
  kind: 'post-softmax';
  mask: 'causal';
  data: Float32Array;
  /** Optional pre-softmax scores (scaled QK^T + mask, -Infinity where masked). */
  scores?: Float32Array;
}

export interface Prediction {
  id: number;
  text: string;
  prob: number;
  logit: number;
}

export interface PredictionSet {
  position: number;
  top: Prediction[];
  target?: Prediction & { rank: number };
  perPosition?: { top: Omit<Prediction, 'logit'>[] }[];
}

export interface LogitLensLayer {
  layer: number;
  top: Prediction[];
  targetProb?: number;
  targetRank?: number;
}

export interface AblationCell {
  targetProb: number;
  targetLogit: number;
  top1: { id: number; text: string; prob: number };
  klFromBaseline: number;
}

export interface PrecomputedAblation {
  precomputed: true;
  method: 'zero-head-output';
  location: string;
  measuredAt: { position: number; target: { id: number; text: string } };
  baseline: { targetProb: number; targetLogit: number };
  caveat: string;
  heads: AblationCell[][];
}

export type RunSource =
  | {
      kind: 'precomputed';
      demoId: string;
      artifactFile: string;
      artifactSha256: string;
      attentionSha256: string;
    }
  | { kind: 'live'; modelId: string; weightsSha256: string };

export interface Provenance {
  app: { name: 'AttentionScope'; version: string; commit: string };
  precomputed: boolean;
  generator: string;
  computePrecision: string;
  storedPrecision: string;
  deterministic: boolean;
  seed: number | null;
  computedAt: string;
  environment?: Record<string, string>;
}

export interface AttentionRun {
  id: string;
  title: string;
  source: RunSource;
  model: ModelInfo;
  tokenizer: TokenizerInfo;
  input: TokenizedInput;
  attention: AttentionTensor;
  predictions?: PredictionSet;
  logitLens?: { position: number; method: string; layers: LogitLensLayer[] };
  residualNorms?: { method: string; values: number[][] };
  ablation?: PrecomputedAblation;
  intervention?: InterventionSpec;
  suggestedView?: { layer: number; head: number; queryIndex: number; keyIndex: number; criterion: string };
  summary?: string;
  literature?: string;
  provenance: Provenance;
}

export type InterventionKind =
  | 'zero-head-output'
  | 'replace-attention-uniform'
  | 'replace-attention-previous-token'
  | 'replace-attention-first-token';

export interface InterventionSpec {
  kind: InterventionKind;
  layer: number;
  head: number;
}

export interface InterventionValidation {
  shape: [number, number];
  maxRowSumError: number;
  maskViolations: number;
  negativeEntries: number;
  passed: boolean;
}

export interface ComparisonResult {
  spec: InterventionSpec;
  location: string;
  description: string;
  model: { id: string; revision: string };
  inputTokenIds: number[];
  position: number;
  target: {
    id: number;
    text: string;
    baselineProb: number;
    intervenedProb: number;
    baselineLogit: number;
    intervenedLogit: number;
    deltaLogProb: number;
  };
  baselineTop: Prediction[];
  intervenedTop: Prediction[];
  klDivergence: number;
  validation: InterventionValidation | null;
  durationMs: number;
}

export interface ExperimentSession {
  schemaVersion: 1;
  id: string;
  name: string;
  createdAt: string;
  app: Provenance['app'];
  source: RunSource;
  model: { id: string; repo: string; revision: string };
  tokenizer: TokenizerInfo;
  input: { text: string; tokenIds: number[] };
  view: { layer: number; head: number; queryIndex: number };
  intervention?: ComparisonResult;
  sweep?: { spec: Omit<InterventionSpec, 'layer' | 'head'>; targetId: number; deltaLogProb: number[][] };
  notes: string;
}
