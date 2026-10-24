import type { AttentionShape, TokenizedInput } from '@/types/contracts';
import type { DemoArtifact } from '@/types/artifacts';
import type { HeadStats, SimilarityMetric } from '@/core/metrics/headMetrics';

export type DataApi = {
  loadDemo: {
    params: { jsonUrl: string; binUrl: string; jsonSha256: string };
    result: {
      artifact: DemoArtifact;
      attention: Float32Array;
      bytes: number;
      verifyMs: number;
      fetchMs: number;
    };
  };
  registerTensor: { params: { key: string; shape: AttentionShape; data: Float32Array }; result: true };
  headStats: { params: { key: string }; result: HeadStats[][] };
  similarity: {
    params: { key: string; layer: number; head: number; metric: SimilarityMetric };
    result: number[][];
  };
  rollout: { params: { key: string; residualWeight: number }; result: Float64Array };
  loadTokenizer: {
    params: { id: string; url: string; sha256: string; name: string; repo: string; revision: string };
    result: { vocabSize: number; ms: number };
  };
  tokenize: {
    params: { id: string; text: string; prependBos: boolean; bos?: { id: number; text: string } };
    result: TokenizedInput & { ms: number };
  };
};
