import type { AttentionRun, ComparisonResult, ExperimentSession } from '@/types/contracts';
import type { SweepResult } from '@/workers/modelApi';
import { APP_INFO, nowIso, randomId } from './provenance';

export const MAX_SESSION_BYTES = 2_000_000;

export function makeSession(
  run: AttentionRun,
  view: { layer: number; head: number; queryIndex: number },
  opts: { name: string; notes?: string; comparison?: ComparisonResult | null; sweep?: SweepResult | null },
): ExperimentSession {
  return {
    schemaVersion: 1,
    id: randomId('exp'),
    name: opts.name.trim() || 'Untitled experiment',
    createdAt: nowIso(),
    app: APP_INFO,
    source: run.source,
    model: { id: run.model.id, repo: run.model.repo, revision: run.model.revision },
    tokenizer: run.tokenizer,
    input: { text: run.input.text, tokenIds: run.input.tokens.map((t) => t.id) },
    view,
    ...(opts.comparison ? { intervention: opts.comparison } : {}),
    ...(opts.sweep
      ? {
          sweep: {
            spec: { kind: opts.sweep.kind },
            targetId: opts.sweep.target.id,
            deltaLogProb: opts.sweep.deltaLogProb,
          },
        }
      : {}),
    notes: opts.notes ?? '',
  };
}

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);
const isInt = (v: unknown): v is number => Number.isInteger(v) && (v as number) >= 0;

/** Validates untrusted JSON (e.g. an imported file) as an ExperimentSession. */
export function parseSession(value: unknown): ExperimentSession {
  const fail = (why: string): never => {
    throw new Error(`Invalid experiment file: ${why}`);
  };
  if (!isObj(value)) fail('expected a JSON object');
  const v = value as Record<string, unknown>;
  if (v.schemaVersion !== 1) fail(`unsupported schemaVersion ${String(v.schemaVersion)}`);
  if (typeof v.id !== 'string' || !v.id) fail('missing id');
  if (typeof v.name !== 'string') fail('missing name');
  if (typeof v.createdAt !== 'string' || Number.isNaN(Date.parse(v.createdAt))) fail('invalid createdAt');
  const src = v.source;
  if (!isObj(src)) fail('missing source');
  const s = src as Record<string, unknown>;
  if (s.kind === 'precomputed') {
    if (typeof s.demoId !== 'string' || !/^[a-z0-9-]{1,64}$/.test(s.demoId)) fail('invalid demoId');
  } else if (s.kind === 'live') {
    if (typeof s.modelId !== 'string') fail('invalid modelId');
  } else fail('source.kind must be "precomputed" or "live"');
  const input = v.input;
  if (!isObj(input)) fail('missing input');
  const inp = input as Record<string, unknown>;
  if (typeof inp.text !== 'string' || inp.text.length > 20000) fail('input.text must be a string');
  if (!Array.isArray(inp.tokenIds) || !inp.tokenIds.every(isInt)) fail('input.tokenIds must be integers');
  const view = v.view;
  if (!isObj(view)) fail('missing view');
  const vw = view as Record<string, unknown>;
  if (!isInt(vw.layer) || !isInt(vw.head) || !isInt(vw.queryIndex))
    fail('view must contain layer, head, queryIndex');
  if (!isObj(v.model) || typeof (v.model as Record<string, unknown>).revision !== 'string')
    fail('missing model revision');
  if (!isObj(v.tokenizer) || typeof (v.tokenizer as Record<string, unknown>).prependBos !== 'boolean') {
    fail('missing tokenizer.prependBos');
  }
  if (typeof v.notes !== 'string') fail('notes must be a string');
  return value as ExperimentSession;
}

export async function readSessionFile(file: File): Promise<ExperimentSession> {
  if (file.size > MAX_SESSION_BYTES)
    throw new Error(`File is ${(file.size / 1e6).toFixed(1)} MB; the limit is 2 MB`);
  if (!/\.json$/i.test(file.name) && file.type && file.type !== 'application/json') {
    throw new Error('Expected a .json experiment file');
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(await file.text());
  } catch {
    throw new Error('Invalid experiment file: not valid JSON');
  }
  return parseSession(parsed);
}

/** Hash-route share link. Live-run text is included only when the user opts in. */
export function shareHash(
  run: AttentionRun,
  view: { layer: number; head: number; queryIndex: number },
  tab: string,
  includeText: boolean,
): string {
  const p = new URLSearchParams();
  if (run.source.kind === 'precomputed') {
    p.set('demo', run.source.demoId);
  } else {
    p.set('src', 'live');
    if (includeText) p.set('text', run.input.text);
    if (!run.tokenizer.prependBos) p.set('bos', '0');
  }
  p.set('l', String(view.layer));
  p.set('h', String(view.head));
  p.set('q', String(view.queryIndex));
  p.set('tab', tab);
  return `#/lab?${p.toString()}`;
}
