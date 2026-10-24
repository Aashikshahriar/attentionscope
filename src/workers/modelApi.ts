import type {
  ComparisonResult,
  InterventionKind,
  InterventionSpec,
  LogitLensLayer,
  ModelManifest,
  PredictionSet,
  TokenizedInput,
} from '@/types/contracts';

export interface LoadProgress {
  phase: 'manifest' | 'weights' | 'tokenizer' | 'verifying' | 'decoding';
  loaded: number;
  total: number;
  fromCache?: boolean;
}

export interface LoadedModel {
  manifest: ModelManifest;
  fromCache: boolean;
  timings: { downloadMs: number; verifyMs: number; decodeMs: number };
  verified: { weightsSha256: string; tokenizerSha256: string };
}

export interface LiveRunResult {
  input: TokenizedInput;
  attention: Float32Array;
  scores: Float32Array;
  predictions: PredictionSet;
  logitLens: LogitLensLayer[];
  residualNorms: number[][];
  durationMs: number;
}

export interface SweepResult {
  kind: InterventionKind;
  inputTokenIds: number[];
  target: { id: number; text: string; baselineProb: number };
  deltaLogProb: number[][];
  intervenedProb: number[][];
  klDivergence: number[][];
  durationMs: number;
}

export type ModelApi = {
  load: { params: { manifestUrl: string; cache: boolean }; result: LoadedModel; progress: LoadProgress };
  tokenize: { params: { text: string; prependBos: boolean }; result: TokenizedInput };
  run: { params: { text: string; prependBos: boolean }; result: LiveRunResult; progress: { layer: number } };
  compare: {
    params: { tokenIds: number[]; spec: InterventionSpec; targetId: number | null };
    result: ComparisonResult;
  };
  sweep: {
    params: { tokenIds: number[]; kind: InterventionKind; targetId: number | null };
    result: SweepResult;
    progress: { done: number; total: number };
  };
  resolveToken: { params: { text: string }; result: { ids: number[]; pieces: string[] } };
  clearCache: { params: null; result: boolean };
};

export const MODEL_CACHE_NAME = 'attentionscope-models-v1';
