import { useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Tabs } from 'radix-ui';
import { Grid3x3, Spline, Type, LayoutGrid, FlaskConical, Calculator, AlertTriangle } from 'lucide-react';
import { LAB_TABS, useWorkspace, type LabTab } from '@/stores/workspace';
import { Callout, EmptyState, Kbd, Spinner, cn } from '@/components/ui/primitives';
import { ControlsPanel } from './ControlsPanel';
import { Inspector } from './Inspector';
import { BottomStrip } from './BottomStrip';
import { HeatmapTab } from './HeatmapTab';
import { ArcView } from '@/features/arcs/ArcView';
import { TokensTab } from '@/features/tokenizer/TokenView';
import { CompareView } from '@/features/compare/CompareView';
import { InterventionsView } from '@/features/interventions/InterventionsView';
import { QkvLab } from '@/features/qkv/QkvLab';
import { headMatrix } from '@/core/attention/tensor';

const TAB_META: Record<LabTab, { label: string; icon: typeof Grid3x3; needsRun: boolean }> = {
  heatmap: { label: 'Heatmap', icon: Grid3x3, needsRun: true },
  arcs: { label: 'Arcs', icon: Spline, needsRun: true },
  tokens: { label: 'Tokens', icon: Type, needsRun: true },
  compare: { label: 'Compare', icon: LayoutGrid, needsRun: true },
  interventions: { label: 'Interventions', icon: FlaskConical, needsRun: true },
  qkv: { label: 'Q/K/V Lab', icon: Calculator, needsRun: false },
};

function useUrlState() {
  const [params] = useSearchParams();
  // Apply the URL whenever it changes (initial load, share links, in-app links between lab views).
  const applied = useRef<string | null>(null);
  useEffect(() => {
    const qs = params.toString();
    if (applied.current === qs) return;
    applied.current = qs;
    const ws = useWorkspace.getState();
    const num = (k: string) => {
      const v = params.get(k);
      return v !== null && /^\d+$/.test(v) ? Number(v) : undefined;
    };
    const view = { layer: num('l'), head: num('h'), queryIndex: num('q') };
    const clean = Object.fromEntries(Object.entries(view).filter(([, v]) => v !== undefined));
    const tab = params.get('tab');
    if (tab && (LAB_TABS as string[]).includes(tab)) ws.setTab(tab as LabTab);
    if (params.get('src') === 'live') {
      ws.setSource('live');
      const text = params.get('text');
      if (text) ws.setLiveText(text.slice(0, 2000));
      if (params.get('bos') === '0') ws.setPrependBos(false);
      void ws.init();
      return;
    }
    const demo = params.get('demo');
    if (demo && /^[a-z0-9-]{1,64}$/.test(demo)) void ws.selectDemo(demo, clean);
    else if (!ws.run)
      void ws.init().then(() => {
        const first = useWorkspace.getState().demos[0];
        if (first && !useWorkspace.getState().run) void useWorkspace.getState().selectDemo(first.id);
      });
  }, [params]);
}

function useShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName))) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const ws = useWorkspace.getState();
      if (!ws.run) return;
      const map: Record<string, () => void> = {
        '[': () => ws.setView({ layer: ws.layer - 1 }),
        ']': () => ws.setView({ layer: ws.layer + 1 }),
        ',': () => ws.setView({ head: ws.head - 1 }),
        '.': () => ws.setView({ head: ws.head + 1 }),
      };
      const f = map[e.key];
      if (f) {
        e.preventDefault();
        f();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
}

function RunStateGate({ children }: { children: React.ReactNode }) {
  const { run, runStatus, runError, source, model } = useWorkspace();
  if (runStatus === 'error') {
    return (
      <div className="p-4">
        <Callout tone="danger" title="Could not produce attention data">
          {runError}
        </Callout>
      </div>
    );
  }
  if (!run && runStatus === 'loading') {
    return (
      <div className="flex items-center justify-center gap-3 py-24 text-sm text-muted" role="status">
        <Spinner /> {source === 'live' ? 'Running forward pass…' : 'Loading and verifying artifact…'}
      </div>
    );
  }
  if (!run) {
    return (
      <EmptyState icon={<Grid3x3 className="size-8" aria-hidden />} title="No attention data yet">
        {source === 'live'
          ? model.state === 'ready'
            ? 'Enter text and run a forward pass.'
            : 'Load the Pythia-14m model in the left panel, then run a forward pass on your own text.'
          : 'Choose a precomputed demo in the left panel.'}
      </EmptyState>
    );
  }
  return <>{children}</>;
}

export default function LabPage() {
  useUrlState();
  useShortcuts();
  const { run, tab, setTab, layer, head, queryIndex, keyIndex, setView, runStatus, source } = useWorkspace();
  const T = run?.attention.shape.queryLength ?? 0;
  const stale =
    run &&
    ((source === 'live' && run.source.kind !== 'live') ||
      (source === 'demo' && run.source.kind !== 'precomputed'));

  return (
    <div className="grid gap-3 p-3 lg:grid-cols-[19rem_minmax(0,1fr)] 2xl:grid-cols-[19.5rem_minmax(0,1fr)_22rem]">
      <aside aria-label="Controls" className="min-w-0">
        <ControlsPanel />
      </aside>
      <div className="min-w-0 space-y-3">
        <section aria-label="Visualization" className="rounded-xl border border-line bg-panel">
          <Tabs.Root value={tab} onValueChange={(v) => setTab(v as LabTab)}>
            <div className="flex items-center gap-2 overflow-x-auto border-b border-line px-2 scrollbar-thin">
              <Tabs.List aria-label="Views" className="flex shrink-0">
                {LAB_TABS.map((t) => {
                  const M = TAB_META[t];
                  const Icon = M.icon;
                  return (
                    <Tabs.Trigger
                      key={t}
                      value={t}
                      className="relative inline-flex h-11 items-center gap-1.5 px-3 text-[13px] font-medium text-muted hover:text-fg data-[state=active]:text-fg data-[state=active]:after:absolute data-[state=active]:after:inset-x-2 data-[state=active]:after:bottom-0 data-[state=active]:after:h-0.5 data-[state=active]:after:rounded-full data-[state=active]:after:bg-accent"
                    >
                      <Icon aria-hidden className="size-4" />
                      {M.label}
                    </Tabs.Trigger>
                  );
                })}
              </Tabs.List>
              {run && tab !== 'qkv' && (
                <div className="ml-auto hidden shrink-0 items-center gap-2 pr-2 text-[12px] text-muted md:flex">
                  {runStatus === 'loading' && <Spinner className="size-3" />}
                  <span className="font-mono">
                    {run.model.displayName} · L{layer} H{head} · q{queryIndex}
                  </span>
                </div>
              )}
            </div>
            {stale && tab !== 'qkv' && (
              <div className="px-4 pt-3">
                <Callout tone="warn">
                  <span className="inline-flex items-center gap-1">
                    <AlertTriangle aria-hidden className="size-3.5" />
                  </span>{' '}
                  Showing the previous {run.source.kind === 'live' ? 'live run' : 'precomputed demo'} until
                  you {source === 'live' ? 'run the live model' : 'select a demo'}.
                </Callout>
              </div>
            )}
            <div className={cn('p-4')}>
              <Tabs.Content value="heatmap" className="outline-none">
                <RunStateGate>{run && <HeatmapTab run={run} />}</RunStateGate>
              </Tabs.Content>
              <Tabs.Content value="arcs" className="outline-none">
                <RunStateGate>
                  {run && (
                    <ArcView
                      values={headMatrix(run.attention, layer, head)}
                      T={T}
                      tokens={run.input.tokens}
                      selectedQuery={queryIndex}
                      selectedKey={keyIndex}
                      onSelect={(q, k) => setView({ queryIndex: q, keyIndex: k })}
                    />
                  )}
                </RunStateGate>
              </Tabs.Content>
              <Tabs.Content value="tokens" className="outline-none">
                <RunStateGate>
                  {run && (
                    <TokensTab run={run} selected={queryIndex} onSelect={(i) => setView({ queryIndex: i })} />
                  )}
                </RunStateGate>
              </Tabs.Content>
              <Tabs.Content value="compare" className="outline-none">
                <RunStateGate>{run && <CompareView run={run} />}</RunStateGate>
              </Tabs.Content>
              <Tabs.Content value="interventions" className="outline-none">
                <RunStateGate>{run && <InterventionsView run={run} />}</RunStateGate>
              </Tabs.Content>
              <Tabs.Content value="qkv" className="outline-none">
                <QkvLab />
              </Tabs.Content>
            </div>
          </Tabs.Root>
        </section>
        <BottomStrip />
        <p className="px-1 text-[11.5px] text-muted">
          Shortcuts: <Kbd>[</Kbd> <Kbd>]</Kbd> layer · <Kbd>,</Kbd> <Kbd>.</Kbd> head · arrow keys inside the
          heatmap move the selected cell · <Kbd>Esc</Kbd> clears the key.
        </p>
      </div>
      <aside aria-label="Inspector" className="min-w-0 lg:col-span-2 2xl:col-span-1">
        {run ? (
          <Inspector run={run} />
        ) : (
          <div className="rounded-xl border border-line bg-panel p-4 text-[12.5px] text-muted">
            The inspector shows the selected token, its attention row, tensor shapes and caveats.
          </div>
        )}
      </aside>
    </div>
  );
}
