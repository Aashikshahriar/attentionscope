import type { InterventionKind, InterventionSpec } from '@/types/contracts';

export interface InterventionDefinition {
  kind: InterventionKind;
  label: string;
  family: 'head-output-ablation' | 'attention-matrix-replacement';
  description: string;
}

export const INTERVENTIONS: InterventionDefinition[] = [
  {
    kind: 'zero-head-output',
    label: 'Zero head output',
    family: 'head-output-ablation',
    description:
      'Sets the head output O_h = A_h V_h to zero before concatenation and the output projection W_O. The head’s attention pattern is still computed; its contribution to the residual stream is removed (the W_O bias is still added).',
  },
  {
    kind: 'replace-attention-uniform',
    label: 'Replace attention: uniform',
    family: 'attention-matrix-replacement',
    description:
      'Replaces the head’s post-softmax attention with a uniform distribution over the causally visible keys (row i: 1/(i+1) for j ≤ i), then computes O_h with the original values V_h.',
  },
  {
    kind: 'replace-attention-previous-token',
    label: 'Replace attention: previous token',
    family: 'attention-matrix-replacement',
    description:
      'Replaces the head’s attention with a one-hot pattern on the previous token (row 0 attends to itself), then computes O_h with the original values V_h.',
  },
  {
    kind: 'replace-attention-first-token',
    label: 'Replace attention: first token',
    family: 'attention-matrix-replacement',
    description:
      'Replaces the head’s attention with a one-hot pattern on position 0 for every query, then computes O_h with the original values V_h.',
  },
];

export function interventionDefinition(kind: InterventionKind): InterventionDefinition {
  const d = INTERVENTIONS.find((i) => i.kind === kind);
  if (!d) throw new Error(`unknown intervention ${kind}`);
  return d;
}

export function interventionLocation(spec: InterventionSpec): string {
  const d = interventionDefinition(spec.kind);
  return d.family === 'head-output-ablation'
    ? `layer ${spec.layer}, head ${spec.head}: head output before W_O (attention.dense input slice [${spec.head}·d_head, ${spec.head + 1}·d_head))`
    : `layer ${spec.layer}, head ${spec.head}: post-softmax attention probabilities (before multiplication by V)`;
}

/** Row-stochastic, causally valid replacement pattern (row-major T×T), or null for ablations. */
export function replacementMatrix(kind: InterventionKind, T: number): Float64Array | null {
  if (kind === 'zero-head-output') return null;
  const m = new Float64Array(T * T);
  for (let i = 0; i < T; i++) {
    if (kind === 'replace-attention-uniform') {
      for (let j = 0; j <= i; j++) m[i * T + j] = 1 / (i + 1);
    } else if (kind === 'replace-attention-previous-token') {
      m[i * T + Math.max(0, i - 1)] = 1;
    } else {
      m[i * T] = 1;
    }
  }
  return m;
}
