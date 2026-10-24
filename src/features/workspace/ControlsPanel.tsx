import { useEffect, useState } from 'react';
import { Cpu, Database, Play, Square, Sparkles, ShieldCheck, Trash2 } from 'lucide-react';
import { useWorkspace } from '@/stores/workspace';
import { useSettings } from '@/stores/settings';
import { Badge, Button, Callout, Field, Panel, ProgressBar, cn } from '@/components/ui/primitives';
import { Segmented, Select, Stepper, Switch, Tip } from '@/components/ui/controls';
import { modelWorker, releaseModelWorker } from '@/services/workers';
import { log } from '@/stores/log';

const MODEL_MB = 28.1;

function ModelCard() {
  const { model, loadModel, cancelModelLoad } = useWorkspace();
  const { cacheModel, set } = useSettings();
  if (model.state === 'ready') {
    const m = model.loaded.manifest;
    return (
      <div className="space-y-2 rounded-lg border border-teal/30 bg-teal-soft p-3 text-[12.5px]">
        <div className="flex items-center gap-2 font-medium text-fg">
          <ShieldCheck aria-hidden className="size-4 text-teal" /> {m.displayName} ready
        </div>
        <p className="text-muted">
          {m.architecture.numLayers} layers · {m.architecture.numHeads} heads · d_model{' '}
          {m.architecture.hiddenSize} · max {m.runtime.maxSequenceLength} tokens. Weights SHA-256 verified (
          {model.loaded.verified.weightsSha256.slice(0, 10)}…)
          {model.loaded.fromCache ? ', loaded from browser storage' : ''}.
        </p>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            releaseModelWorker();
            useWorkspace.setState({ model: { state: 'unloaded' } });
            log({ level: 'info', scope: 'model', message: 'Model released from memory' });
          }}
        >
          Release from memory
        </Button>
      </div>
    );
  }
  if (model.state === 'loading') {
    const p = model.progress;
    const frac = p && p.phase === 'weights' && p.total ? p.loaded / p.total : null;
    const phase = {
      manifest: 'Fetching manifest',
      weights: p?.fromCache ? 'Reading cached weights' : 'Downloading weights',
      tokenizer: 'Fetching tokenizer',
      verifying: 'Verifying SHA-256 checksums',
      decoding: 'Decoding float16 weights',
    }[p?.phase ?? 'manifest'];
    return (
      <div
        className="space-y-2 rounded-lg border border-line bg-elevated p-3 text-[12.5px]"
        aria-live="polite"
      >
        <div className="flex items-center justify-between gap-2">
          <span className="text-fg">{phase}…</span>
          {p?.phase === 'weights' && (
            <span className="font-mono tabular text-muted">
              {(p.loaded / 1e6).toFixed(1)} / {(p.total / 1e6).toFixed(1)} MB
            </span>
          )}
        </div>
        <ProgressBar value={frac} label="Model loading progress" />
        <Button size="sm" variant="ghost" onClick={cancelModelLoad}>
          Cancel
        </Button>
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {model.state === 'error' && (
        <Callout tone="danger" title="Model could not be loaded">
          {model.message}
        </Callout>
      )}
      <p className="text-[12.5px] leading-relaxed text-muted">
        Runs EleutherAI Pythia-14m (Apache-2.0) in a Web Worker with AttentionScope&apos;s own GPT-NeoX
        implementation, exposing every attention probability. Download: {MODEL_MB} MB, verified by SHA-256.
        Your text never leaves the browser.
      </p>
      <Switch
        id="cache-model"
        label="Keep a copy in browser storage"
        description={`Stores ~${MODEL_MB} MB in Cache Storage so the next visit loads offline. Off by default.`}
        checked={cacheModel}
        onChange={(v) => set({ cacheModel: v })}
      />
      <Button variant="primary" className="w-full" onClick={() => void loadModel()}>
        <Cpu aria-hidden className="size-4" /> Load model ({MODEL_MB} MB)
      </Button>
      {cacheModel && (
        <button
          type="button"
          className="inline-flex items-center gap-1 text-[11.5px] text-muted underline hover:text-fg"
          onClick={async () => {
            const ok = await modelWorker().call('clearCache', null);
            log({
              level: 'info',
              scope: 'model',
              message: ok ? 'Cleared cached model files' : 'No cached model files',
            });
          }}
        >
          <Trash2 aria-hidden className="size-3" /> Clear cached model
        </button>
      )}
    </div>
  );
}

function LiveInput() {
  const {
    model,
    liveText,
    setLiveText,
    prependBos,
    setPrependBos,
    runLive,
    cancelLive,
    runStatus,
    liveLayer,
    source,
  } = useWorkspace();
  const ready = model.state === 'ready';
  const [count, setCount] = useState<number | null>(null);
  const max = ready ? model.loaded.manifest.runtime.maxSequenceLength : 64;
  useEffect(() => {
    if (!ready) return;
    const t = setTimeout(() => {
      modelWorker()
        .call('tokenize', { text: liveText, prependBos })
        .then((r) => setCount(r.tokens.length))
        .catch(() => setCount(null));
    }, 180);
    return () => clearTimeout(t);
  }, [liveText, prependBos, ready]);
  const over = count !== null && count > max;
  const running = runStatus === 'loading' && source === 'live';
  return (
    <div className="space-y-3">
      <Field
        label="Input text"
        htmlFor="live-text"
        hint={
          ready && count !== null ? (
            <span className={cn(over && 'text-danger')}>
              {count} tokens{over ? ` — exceeds the ${max}-token in-browser limit` : ` of ${max} max`}
            </span>
          ) : (
            'Processed locally in your browser.'
          )
        }
      >
        <textarea
          id="live-text"
          value={liveText}
          maxLength={2000}
          onChange={(e) => setLiveText(e.target.value)}
          rows={4}
          dir="auto"
          className="w-full resize-y rounded-lg border border-line bg-elevated px-3 py-2 text-[13.5px] text-fg placeholder:text-muted"
          placeholder="Type a sentence…"
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && ready && !over) void runLive();
          }}
        />
      </Field>
      <Switch
        id="bos"
        label="Prepend <|endoftext|> (BOS)"
        description="Pythia saw this separator between documents in training. Recorded in exports."
        checked={prependBos}
        onChange={setPrependBos}
      />
      {running ? (
        <div className="space-y-2" aria-live="polite">
          <ProgressBar value={liveLayer === null ? null : liveLayer / 6} label="Forward pass progress" />
          <Button className="w-full" onClick={cancelLive}>
            <Square aria-hidden className="size-3.5" /> Cancel
          </Button>
        </div>
      ) : (
        <Button
          variant="primary"
          className="w-full"
          disabled={!ready || over || !liveText.trim()}
          onClick={() => void runLive()}
        >
          <Play aria-hidden className="size-4" /> Run forward pass
        </Button>
      )}
    </div>
  );
}

export function ControlsPanel() {
  const ws = useWorkspace();
  const { source, setSource, demos, demoId, selectDemo, run, layer, head, setView, valueMode, setValueMode } =
    ws;
  const shape = run?.attention.shape;
  const demo = demos.find((d) => d.id === demoId);
  return (
    <div className="space-y-3">
      <Panel title="Source" id="ctl-source">
        <div className="space-y-3">
          <Segmented<'demo' | 'live'>
            label="Data source"
            value={source}
            onChange={setSource}
            className="w-full"
            options={[
              { value: 'demo', label: 'Precomputed demo' },
              { value: 'live', label: 'Live model' },
            ]}
          />
          {source === 'demo' ? (
            <div className="space-y-2.5">
              <Field label="Demo" htmlFor="demo-select">
                <Select
                  id="demo-select"
                  label="Demo"
                  value={demoId ?? undefined}
                  placeholder={ws.demosError ? 'Demos unavailable' : 'Loading demos…'}
                  onChange={(id) => void selectDemo(id)}
                  options={demos.map((d) => ({
                    value: d.id,
                    label: d.title,
                    description: `${d.category} · ${d.tokens} tokens`,
                  }))}
                />
              </Field>
              {demo && <p className="text-[12.5px] leading-relaxed text-muted">{demo.summary}</p>}
              <div className="flex flex-wrap gap-1.5">
                <Badge tone="warn">
                  <Database aria-hidden className="size-3" /> precomputed
                </Badge>
                <Badge>GPT-2 small · MIT</Badge>
              </div>
              <p className="text-[11.5px] leading-snug text-muted">
                Real attention tensors from offline forward passes of GPT-2 small, bundled with SHA-256
                checksums. Not computed in your browser.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <ModelCard />
              <LiveInput />
            </div>
          )}
        </div>
      </Panel>

      <Panel title="View" id="ctl-view">
        {shape ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2.5">
              <Field label={`Layer (0–${shape.layers - 1})`} htmlFor="layer">
                <Stepper
                  id="layer"
                  label="Layer"
                  value={layer}
                  min={0}
                  max={shape.layers - 1}
                  onChange={(v) => setView({ layer: v })}
                />
              </Field>
              <Field label={`Head (0–${shape.heads - 1})`} htmlFor="head">
                <Stepper
                  id="head"
                  label="Head"
                  value={head}
                  min={0}
                  max={shape.heads - 1}
                  onChange={(v) => setView({ head: v })}
                />
              </Field>
            </div>
            {run?.suggestedView && (
              <Tip content={run.suggestedView.criterion}>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() =>
                    setView({
                      layer: run.suggestedView!.layer,
                      head: run.suggestedView!.head,
                      queryIndex: run.suggestedView!.queryIndex,
                      keyIndex: run.suggestedView!.keyIndex,
                    })
                  }
                >
                  <Sparkles aria-hidden className="size-3.5" /> Suggested head {run.suggestedView.layer}.
                  {run.suggestedView.head}
                </Button>
              </Tip>
            )}
            <Field
              label="Values"
              hint={
                run && !run.attention.scores
                  ? 'Pre-softmax scores are not stored in precomputed artifacts.'
                  : 'Scores are q·k/√d before the mask and softmax.'
              }
            >
              <Segmented<'probabilities' | 'scores'>
                label="Values shown"
                value={run?.attention.scores ? valueMode : 'probabilities'}
                onChange={setValueMode}
                className="w-full"
                size="sm"
                options={[
                  { value: 'probabilities', label: 'Post-softmax' },
                  { value: 'scores', label: 'Pre-softmax', disabled: !run?.attention.scores },
                ]}
              />
            </Field>
            <div className="rounded-lg border border-line bg-elevated p-2.5 text-[11.5px] leading-relaxed text-muted">
              <strong className="font-medium text-fg">Masks.</strong> Causal: key j is visible from query i
              only if j ≤ i (masked scores are −∞, probabilities exactly 0). Padding: none — a single unpadded
              sequence, attention mask all ones. Dropout: disabled (evaluation mode).
            </div>
          </div>
        ) : (
          <p className="text-[12.5px] text-muted">
            Load a demo or run the live model to choose a layer and head.
          </p>
        )}
      </Panel>
    </div>
  );
}
