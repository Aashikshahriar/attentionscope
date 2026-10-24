import type { AttentionRun, ComparisonResult } from '@/types/contracts';
import { headMatrix, headScores } from '@/core/attention/tensor';
import { APP_INFO, nowIso } from './provenance';
import { colorFor } from '@/features/heatmap/colormaps';
import type { Colormap } from '@/stores/settings';

export function provenanceRecord(run: AttentionRun, extra: Record<string, unknown> = {}) {
  return {
    exportedAt: nowIso(),
    app: APP_INFO,
    source: run.source,
    precomputed: run.provenance.precomputed,
    model: {
      id: run.model.id,
      name: run.model.displayName,
      repo: run.model.repo,
      revision: run.model.revision,
      license: run.model.license,
      architecture: run.model.architecture,
    },
    tokenizer: run.tokenizer,
    input: {
      text: run.input.text,
      tokenIds: run.input.tokens.map((t) => t.id),
      normalization: run.input.normalization ?? 'none',
    },
    computation: {
      generator: run.provenance.generator,
      computePrecision: run.provenance.computePrecision,
      storedPrecision: run.provenance.storedPrecision,
      deterministic: run.provenance.deterministic,
      seed: run.provenance.seed,
      computedAt: run.provenance.computedAt,
      environment: run.provenance.environment ?? null,
    },
    attention: {
      shape: run.attention.shape,
      layout: run.attention.layout,
      kind: run.attention.kind,
      mask: run.attention.mask,
    },
    ...extra,
  };
}

function csvCell(v: string): string {
  return /[",\n\r]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
}

const visible = (s: string) => s.replace(/ /g, '·').replace(/\n/g, '\\n').replace(/\t/g, '\\t');

export function headCsv(run: AttentionRun, layer: number, head: number): string {
  const T = run.attention.shape.queryLength;
  const m = headMatrix(run.attention, layer, head);
  const toks = run.input.tokens;
  const p = provenanceRecord(run, { layer, head });
  const lines = [
    `# AttentionScope export — post-softmax attention, layer ${layer}, head ${head}`,
    `# rows = query positions, columns = key positions; each row sums to 1; entries with key > query are 0 (causal mask)`,
    `# model: ${p.model.repo}@${p.model.revision}`,
    `# source: ${run.source.kind}${run.source.kind === 'precomputed' ? ` (${run.source.artifactFile}, attention sha256 ${run.source.attentionSha256})` : ''}`,
    `# tokenizer: ${run.tokenizer.name} (prepend BOS: ${run.tokenizer.prependBos})`,
    `# token ids: ${toks.map((t) => t.id).join(' ')}`,
    `# generator: ${run.provenance.generator}`,
    `# app: AttentionScope ${APP_INFO.version} (${APP_INFO.commit}); exported ${p.exportedAt}`,
    ['query_index', 'query_token', ...toks.map((t, j) => `k${j}:${visible(t.text)}`)].map(csvCell).join(','),
  ];
  for (let i = 0; i < T; i++) {
    const row = [String(i), visible(toks[i]!.text)];
    for (let j = 0; j < T; j++) row.push(String(m[i * T + j]));
    lines.push(row.map(csvCell).join(','));
  }
  return lines.join('\n') + '\n';
}

const toRows = (m: ArrayLike<number>, T: number) =>
  Array.from({ length: T }, (_, i) => Array.from({ length: T }, (_, j) => m[i * T + j]!));

export function headJson(run: AttentionRun, layer: number, head: number): string {
  const T = run.attention.shape.queryLength;
  const scores = headScores(run.attention, layer, head);
  return JSON.stringify(
    {
      provenance: provenanceRecord(run, { layer, head }),
      tokens: run.input.tokens,
      layer,
      head,
      attention: toRows(headMatrix(run.attention, layer, head), T),
      ...(scores
        ? { preSoftmaxScores: toRows(scores, T).map((r) => r.map((v) => (Number.isFinite(v) ? v : null))) }
        : {}),
      predictions: run.predictions ?? null,
    },
    null,
    2,
  );
}

export function runJson(run: AttentionRun, comparison?: ComparisonResult | null): string {
  const { layers, heads, queryLength: T } = run.attention.shape;
  const attention = Array.from({ length: layers }, (_, l) =>
    Array.from({ length: heads }, (_, h) => toRows(headMatrix(run.attention, l, h), T)),
  );
  return JSON.stringify({
    provenance: provenanceRecord(run, comparison ? { intervention: comparison } : {}),
    tokens: run.input.tokens,
    attention,
    predictions: run.predictions ?? null,
    logitLens: run.logitLens ?? null,
    residualNorms: run.residualNorms ?? null,
    precomputedAblation: run.ablation ?? null,
  });
}

export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadText(filename: string, text: string, type = 'application/json'): void {
  downloadBlob(filename, new Blob([text], { type: `${type};charset=utf-8` }));
}

export function exportBaseName(run: AttentionRun, layer: number, head: number): string {
  const src = run.source.kind === 'precomputed' ? run.source.demoId : 'live';
  return `attentionscope_${run.model.id}_${src}_L${layer}H${head}`;
}

/** Renders the head's heatmap with labels and a provenance footer into a PNG. */
export async function headPng(
  run: AttentionRun,
  layer: number,
  head: number,
  colormap: Colormap,
): Promise<Blob> {
  const T = run.attention.shape.queryLength;
  const m = headMatrix(run.attention, layer, head);
  const cell = Math.max(14, Math.min(36, Math.floor(720 / T)));
  const labelW = 140;
  const top = 150;
  const footer = 120;
  const W = labelW + T * cell + 40;
  const H = top + T * cell + footer;
  const scale = 2;
  const canvas = document.createElement('canvas');
  canvas.width = W * scale;
  canvas.height = H * scale;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D is not available');
  ctx.scale(scale, scale);
  ctx.fillStyle = '#080B12';
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#F4F7FB';
  ctx.font = '600 15px system-ui, sans-serif';
  ctx.fillText(`${run.model.displayName} — layer ${layer}, head ${head} (post-softmax attention)`, 16, 26);
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillStyle = '#A1AEC2';
  const label = (s: string) => visible(s).slice(0, 18);
  for (let j = 0; j < T; j++) {
    ctx.save();
    ctx.translate(labelW + j * cell + cell / 2 + 4, top - 8);
    ctx.rotate(-Math.PI / 3);
    ctx.fillText(label(run.input.tokens[j]!.text), 0, 0);
    ctx.restore();
  }
  for (let i = 0; i < T; i++) {
    ctx.textAlign = 'right';
    ctx.fillText(label(run.input.tokens[i]!.text), labelW - 8, top + i * cell + cell / 2 + 4);
    ctx.textAlign = 'left';
    for (let j = 0; j < T; j++) {
      if (j > i) {
        ctx.fillStyle = '#111827';
      } else {
        ctx.fillStyle = colorFor(colormap, m[i * T + j]!);
      }
      ctx.fillRect(labelW + j * cell, top + i * cell, cell - 1, cell - 1);
    }
  }
  ctx.fillStyle = '#A1AEC2';
  ctx.font = '10.5px ui-monospace, monospace';
  const lines = [
    `rows: query tokens · columns: key tokens · color: probability 0 → 1 (${colormap}) · dark cells: causally masked`,
    `source: ${run.provenance.precomputed ? 'PRECOMPUTED artifact' : 'live in-browser forward pass'} · ${run.model.repo}@${run.model.revision.slice(0, 10)}`,
    `tokenizer: ${run.tokenizer.name}, BOS prepended: ${run.tokenizer.prependBos} · token ids: ${run.input.tokens
      .map((t) => t.id)
      .join(' ')
      .slice(0, 90)}`,
    `generator: ${run.provenance.generator.slice(0, 120)}`,
    `AttentionScope ${APP_INFO.version} (${APP_INFO.commit}) · exported ${nowIso()}`,
  ];
  lines.forEach((l, i) => ctx.fillText(l, 16, top + T * cell + 28 + i * 17));
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG encoding failed'))), 'image/png'),
  );
}
