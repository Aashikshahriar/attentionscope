import { create } from 'zustand';
import type {
  AttentionRun,
  ComparisonResult,
  ExperimentSession,
  InterventionKind,
  InterventionSpec,
} from '@/types/contracts';
import type { DemoManifestEntry } from '@/types/artifacts';
import type { LoadProgress, LoadedModel, SweepResult } from '@/workers/modelApi';
import { fetchDemoManifest, liveRunToAttentionRun, loadDemoRun } from '@/services/runs';
import { modelWorker } from '@/services/workers';
import { assetUrl } from '@/services/paths';
import { isAbortError } from '@/workers/rpc';
import { useSettings } from './settings';
import { log } from './log';

export type LabTab = 'heatmap' | 'arcs' | 'tokens' | 'compare' | 'interventions' | 'qkv';
export const LAB_TABS: LabTab[] = ['heatmap', 'arcs', 'tokens', 'compare', 'interventions', 'qkv'];

export type ModelStatus =
  | { state: 'unloaded' }
  | { state: 'loading'; progress: LoadProgress | null }
  | { state: 'ready'; loaded: LoadedModel }
  | { state: 'error'; message: string };

export interface ViewState {
  layer: number;
  head: number;
  queryIndex: number;
  keyIndex: number | null;
}

export interface WorkspaceState extends ViewState {
  source: 'demo' | 'live';
  demos: DemoManifestEntry[];
  demosError: string | null;
  demoId: string | null;
  run: AttentionRun | null;
  runStatus: 'idle' | 'loading' | 'ready' | 'error';
  runError: string | null;
  tab: LabTab;
  valueMode: 'probabilities' | 'scores';
  liveText: string;
  prependBos: boolean;
  model: ModelStatus;
  liveLayer: number | null;
  comparison: ComparisonResult | null;
  comparisonStatus: 'idle' | 'running' | 'error';
  comparisonError: string | null;
  sweep: SweepResult | null;
  sweepProgress: { done: number; total: number } | null;
  sweepError: string | null;

  init: () => Promise<void>;
  setSource: (s: 'demo' | 'live') => void;
  selectDemo: (id: string, view?: Partial<ViewState>) => Promise<void>;
  setView: (v: Partial<ViewState>) => void;
  setTab: (t: LabTab) => void;
  setValueMode: (m: 'probabilities' | 'scores') => void;
  setLiveText: (t: string) => void;
  setPrependBos: (b: boolean) => void;
  loadModel: () => Promise<void>;
  cancelModelLoad: () => void;
  runLive: (view?: Partial<ViewState>) => Promise<void>;
  cancelLive: () => void;
  runComparison: (spec: InterventionSpec, targetId: number | null) => Promise<void>;
  runSweep: (kind: InterventionKind, targetId: number | null) => Promise<void>;
  cancelSweep: () => void;
  restoreSession: (s: ExperimentSession) => Promise<void>;
}

export const DEFAULT_LIVE_TEXT = 'When Mary and John went to the store, John gave a drink to';

let demoAbort: AbortController | null = null;
let modelAbort: AbortController | null = null;
let liveAbort: AbortController | null = null;
let sweepAbort: AbortController | null = null;

function clampView(run: AttentionRun, v: ViewState): ViewState {
  const { layers, heads, queryLength } = run.attention.shape;
  const c = (x: number, n: number) => Math.min(Math.max(0, Math.trunc(x) || 0), n - 1);
  return {
    layer: c(v.layer, layers),
    head: c(v.head, heads),
    queryIndex: c(v.queryIndex, queryLength),
    keyIndex: v.keyIndex === null ? null : c(v.keyIndex, queryLength),
  };
}

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export const useWorkspace = create<WorkspaceState>((set, get) => ({
  source: 'demo',
  demos: [],
  demosError: null,
  demoId: null,
  run: null,
  runStatus: 'idle',
  runError: null,
  layer: 0,
  head: 0,
  queryIndex: 0,
  keyIndex: null,
  tab: 'heatmap',
  valueMode: 'probabilities',
  liveText: DEFAULT_LIVE_TEXT,
  prependBos: true,
  model: { state: 'unloaded' },
  liveLayer: null,
  comparison: null,
  comparisonStatus: 'idle',
  comparisonError: null,
  sweep: null,
  sweepProgress: null,
  sweepError: null,

  async init() {
    if (get().demos.length) return;
    try {
      const m = await fetchDemoManifest();
      set({ demos: m.demos, demosError: null });
    } catch (e) {
      set({ demosError: message(e) });
      log({ level: 'error', scope: 'demo', message: `Could not load demo manifest: ${message(e)}` });
    }
  },

  setSource(source) {
    set({ source });
  },

  async selectDemo(id, view) {
    await get().init();
    const entry = get().demos.find((d) => d.id === id);
    if (!entry) {
      set({ runStatus: 'error', runError: `Unknown demo "${id}"` });
      return;
    }
    demoAbort?.abort();
    const ctrl = new AbortController();
    demoAbort = ctrl;
    set({ source: 'demo', demoId: id, runStatus: 'loading', runError: null, valueMode: 'probabilities' });
    try {
      const { run, bytes, fetchMs, verifyMs } = await loadDemoRun(entry, ctrl.signal);
      if (ctrl.signal.aborted) return;
      const sv = run.suggestedView;
      const next = clampView(run, {
        layer: view?.layer ?? sv?.layer ?? 0,
        head: view?.head ?? sv?.head ?? 0,
        queryIndex: view?.queryIndex ?? sv?.queryIndex ?? run.attention.shape.queryLength - 1,
        keyIndex: view?.keyIndex ?? null,
      });
      set({
        run,
        runStatus: 'ready',
        ...next,
        comparison: null,
        sweep: null,
        sweepError: null,
        comparisonError: null,
      });
      log({
        level: 'info',
        scope: 'demo',
        message: `Loaded precomputed demo "${entry.title}" (${(bytes / 1024).toFixed(0)} KB, ${run.attention.shape.layers}×${run.attention.shape.heads} heads, T=${run.attention.shape.queryLength}); SHA-256 verified in ${verifyMs.toFixed(0)} ms`,
        durationMs: fetchMs,
      });
    } catch (e) {
      if (isAbortError(e)) return;
      set({ runStatus: 'error', runError: message(e) });
      log({ level: 'error', scope: 'demo', message: `Failed to load demo "${id}": ${message(e)}` });
    }
  },

  setView(v) {
    const { run } = get();
    const merged = { ...pickView(get()), ...v };
    set(run ? clampView(run, merged) : merged);
  },

  setTab(tab) {
    set({ tab });
  },

  setValueMode(valueMode) {
    set({ valueMode });
  },

  setLiveText(liveText) {
    set({ liveText });
  },

  setPrependBos(prependBos) {
    set({ prependBos });
  },

  async loadModel() {
    if (get().model.state === 'loading' || get().model.state === 'ready') return;
    modelAbort?.abort();
    const ctrl = new AbortController();
    modelAbort = ctrl;
    set({ model: { state: 'loading', progress: null } });
    const cache = useSettings.getState().cacheModel;
    try {
      const loaded = await modelWorker().call(
        'load',
        { manifestUrl: assetUrl('models/pythia-14m/manifest.json'), cache },
        {
          signal: ctrl.signal,
          onProgress: (progress) => set({ model: { state: 'loading', progress } }),
        },
      );
      if (ctrl.signal.aborted) return;
      set({ model: { state: 'ready', loaded } });
      log({
        level: 'info',
        scope: 'model',
        message: `Loaded ${loaded.manifest.displayName} (${(loaded.manifest.weights.byteLength / 1e6).toFixed(1)} MB${loaded.fromCache ? ', from browser cache' : ''}); weights and tokenizer SHA-256 verified (${loaded.timings.verifyMs.toFixed(0)} ms), decoded in ${loaded.timings.decodeMs.toFixed(0)} ms`,
        durationMs: loaded.timings.downloadMs,
      });
    } catch (e) {
      if (isAbortError(e) || ctrl.signal.aborted) return;
      set({ model: { state: 'error', message: message(e) } });
      log({ level: 'error', scope: 'model', message: `Model load failed: ${message(e)}` });
    }
  },

  cancelModelLoad() {
    modelAbort?.abort();
    // A download in progress cannot be interrupted cooperatively; restart the worker.
    modelWorker().terminate();
    set({ model: { state: 'unloaded' } });
    log({ level: 'warn', scope: 'model', message: 'Model load cancelled' });
  },

  async runLive(view) {
    const { model, liveText, prependBos } = get();
    if (model.state !== 'ready') return;
    liveAbort?.abort();
    const ctrl = new AbortController();
    liveAbort = ctrl;
    set({ source: 'live', runStatus: 'loading', runError: null, liveLayer: 0 });
    try {
      const r = await modelWorker().call(
        'run',
        { text: liveText, prependBos },
        { signal: ctrl.signal, onProgress: (p) => set({ liveLayer: p.layer }) },
      );
      const run = liveRunToAttentionRun(r, model.loaded, prependBos);
      const T = run.attention.shape.queryLength;
      const prev = pickView(get());
      set({
        run,
        runStatus: 'ready',
        liveLayer: null,
        comparison: null,
        sweep: null,
        sweepError: null,
        comparisonError: null,
        ...clampView(run, { ...prev, queryIndex: T - 1, keyIndex: null, ...view }),
      });
      log({
        level: 'info',
        scope: 'run',
        message: `Forward pass on ${run.model.displayName}: T=${T}, ${run.attention.shape.layers}×${run.attention.shape.heads} heads, logits at ${T} positions + logit lens`,
        durationMs: r.durationMs,
      });
    } catch (e) {
      set({ liveLayer: null });
      if (isAbortError(e)) {
        set({ runStatus: get().run ? 'ready' : 'idle' });
        log({ level: 'warn', scope: 'run', message: 'Forward pass cancelled' });
        return;
      }
      set({ runStatus: 'error', runError: message(e) });
      log({ level: 'error', scope: 'run', message: `Forward pass failed: ${message(e)}` });
    }
  },

  cancelLive() {
    liveAbort?.abort();
  },

  async runComparison(spec, targetId) {
    const { run, model } = get();
    if (!run || run.source.kind !== 'live' || model.state !== 'ready') return;
    set({ comparisonStatus: 'running', comparisonError: null });
    try {
      const result = await modelWorker().call('compare', {
        tokenIds: run.input.tokens.map((t) => t.id),
        spec,
        targetId,
      });
      set({ comparison: result, comparisonStatus: 'idle' });
      log({
        level: 'info',
        scope: 'intervention',
        message: `${spec.kind} at L${spec.layer}H${spec.head}: P(${JSON.stringify(result.target.text)}) ${result.target.baselineProb.toFixed(4)} → ${result.target.intervenedProb.toFixed(4)}, KL ${result.klDivergence.toFixed(4)} nats`,
        durationMs: result.durationMs,
      });
    } catch (e) {
      set({ comparisonStatus: 'error', comparisonError: message(e) });
      log({ level: 'error', scope: 'intervention', message: `Intervention failed: ${message(e)}` });
    }
  },

  async runSweep(kind, targetId) {
    const { run, model } = get();
    if (!run || run.source.kind !== 'live' || model.state !== 'ready') return;
    sweepAbort?.abort();
    const ctrl = new AbortController();
    sweepAbort = ctrl;
    set({
      sweepProgress: { done: 0, total: run.attention.shape.layers * run.attention.shape.heads },
      sweepError: null,
    });
    try {
      const result = await modelWorker().call(
        'sweep',
        { tokenIds: run.input.tokens.map((t) => t.id), kind, targetId },
        { signal: ctrl.signal, onProgress: (p) => set({ sweepProgress: p }) },
      );
      set({ sweep: result, sweepProgress: null });
      log({
        level: 'info',
        scope: 'intervention',
        message: `Sweep ${kind} over ${result.deltaLogProb.length * (result.deltaLogProb[0]?.length ?? 0)} heads (target ${JSON.stringify(result.target.text)})`,
        durationMs: result.durationMs,
      });
    } catch (e) {
      set({ sweepProgress: null });
      if (isAbortError(e)) {
        log({ level: 'warn', scope: 'intervention', message: 'Sweep cancelled' });
        return;
      }
      set({ sweepError: message(e) });
      log({ level: 'error', scope: 'intervention', message: `Sweep failed: ${message(e)}` });
    }
  },

  cancelSweep() {
    sweepAbort?.abort();
  },

  async restoreSession(s) {
    const view = { layer: s.view.layer, head: s.view.head, queryIndex: s.view.queryIndex, keyIndex: null };
    if (s.source.kind === 'precomputed') {
      await get().selectDemo(s.source.demoId, view);
    } else {
      set({ source: 'live', liveText: s.input.text, prependBos: s.tokenizer.prependBos });
      if (get().model.state === 'ready') await get().runLive(view);
    }
    log({ level: 'info', scope: 'session', message: `Restored experiment "${s.name}"` });
  },
}));

function pickView(s: ViewState): ViewState {
  return { layer: s.layer, head: s.head, queryIndex: s.queryIndex, keyIndex: s.keyIndex };
}
