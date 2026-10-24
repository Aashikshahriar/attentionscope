import { useCallback, useEffect, useRef, useState } from 'react';
import { Tabs } from 'radix-ui';
import {
  ChevronDown,
  Download,
  FileJson,
  FileSpreadsheet,
  Image,
  Link2,
  Save,
  Trash2,
  Upload,
  RotateCcw,
} from 'lucide-react';
import { useWorkspace } from '@/stores/workspace';
import { useSettings } from '@/stores/settings';
import { useLog, log } from '@/stores/log';
import { Button, Callout, EmptyState, Badge, cn } from '@/components/ui/primitives';
import { Switch } from '@/components/ui/controls';
import {
  downloadBlob,
  downloadText,
  exportBaseName,
  headCsv,
  headJson,
  headPng,
  runJson,
} from '@/services/export';
import { makeSession, readSessionFile, shareHash } from '@/services/session';
import { deleteExperiment, listExperiments, saveExperiment } from '@/services/persistence/db';
import type { ExperimentSession } from '@/types/contracts';

function ExportPanel() {
  const { run, layer, head, queryIndex, tab, comparison, sweep } = useWorkspace();
  const colormap = useSettings((s) => s.colormap);
  const [includeText, setIncludeText] = useState(false);
  const [copied, setCopied] = useState(false);
  const [name, setName] = useState('');
  const [notes, setNotes] = useState('');
  const [saved, setSaved] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  if (!run)
    return <EmptyState title="Nothing to export yet">Load a demo or run the live model first.</EmptyState>;
  const base = exportBaseName(run, layer, head);
  const doExport = async (kind: string, f: () => Promise<void> | void) => {
    try {
      setErr(null);
      await f();
      log({
        level: 'info',
        scope: 'export',
        message: `Exported ${kind} for L${layer}H${head} with provenance`,
      });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };
  const live = run.source.kind === 'live';
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <div className="space-y-3">
        <h3 className="text-[13px] font-semibold text-fg">Export current head</h3>
        <p className="text-[12px] text-muted">
          Every export embeds the model repo and revision, tokenizer, input text and token ids, source
          (precomputed/live with checksums), precision and app version.
        </p>
        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            onClick={() =>
              doExport('PNG', async () =>
                downloadBlob(`${base}.png`, await headPng(run, layer, head, colormap)),
              )
            }
          >
            <Image aria-hidden className="size-3.5" /> PNG
          </Button>
          <Button
            size="sm"
            onClick={() =>
              doExport('CSV', () => downloadText(`${base}.csv`, headCsv(run, layer, head), 'text/csv'))
            }
          >
            <FileSpreadsheet aria-hidden className="size-3.5" /> CSV
          </Button>
          <Button
            size="sm"
            onClick={() => doExport('JSON', () => downloadText(`${base}.json`, headJson(run, layer, head)))}
          >
            <FileJson aria-hidden className="size-3.5" /> JSON (head)
          </Button>
          <Button
            size="sm"
            onClick={() =>
              doExport('full-run JSON', () =>
                downloadText(
                  `attentionscope_${run.model.id}_${run.source.kind === 'precomputed' ? run.source.demoId : 'live'}_all-heads.json`,
                  runJson(run, comparison),
                ),
              )
            }
          >
            <Download aria-hidden className="size-3.5" /> JSON (all layers & heads)
          </Button>
        </div>
        <div className="space-y-2 border-t border-line pt-3">
          <h3 className="text-[13px] font-semibold text-fg">Share configuration</h3>
          {live && (
            <Switch
              id="share-text"
              label="Include input text in the link"
              description="Off by default: links contain only the model and view unless you opt in."
              checked={includeText}
              onChange={setIncludeText}
            />
          )}
          <Button
            size="sm"
            onClick={async () => {
              const url = `${window.location.origin}${window.location.pathname}${shareHash(run, { layer, head, queryIndex }, tab, includeText)}`;
              try {
                await navigator.clipboard.writeText(url);
                setCopied(true);
                setTimeout(() => setCopied(false), 1800);
              } catch {
                window.prompt('Copy this link', url);
              }
            }}
          >
            <Link2 aria-hidden className="size-3.5" /> {copied ? 'Copied' : 'Copy link'}
          </Button>
          {live && !includeText && (
            <p className="text-[11.5px] text-muted">
              A live-run link reopens the view; the recipient loads the model and enters text.
            </p>
          )}
        </div>
      </div>
      <div className="space-y-3">
        <h3 className="text-[13px] font-semibold text-fg">Save experiment (this browser)</h3>
        <div className="grid gap-2">
          <label className="sr-only" htmlFor="exp-name">
            Experiment name
          </label>
          <input
            id="exp-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Experiment name"
            maxLength={120}
            className="h-9 rounded-lg border border-line bg-elevated px-3 text-[13px] text-fg placeholder:text-muted"
          />
          <label className="sr-only" htmlFor="exp-notes">
            Notes
          </label>
          <textarea
            id="exp-notes"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Notes (optional)"
            rows={2}
            maxLength={4000}
            className="rounded-lg border border-line bg-elevated px-3 py-2 text-[13px] text-fg placeholder:text-muted"
          />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="primary"
            onClick={async () => {
              try {
                const s = makeSession(
                  run,
                  { layer, head, queryIndex },
                  { name: name || run.title, notes, comparison, sweep },
                );
                await saveExperiment(s);
                setSaved(s.name);
                setErr(null);
                window.dispatchEvent(new Event('attentionscope:experiments'));
                log({
                  level: 'info',
                  scope: 'session',
                  message: `Saved experiment "${s.name}" to IndexedDB`,
                });
              } catch (e) {
                setErr(`Could not save: ${e instanceof Error ? e.message : String(e)}`);
              }
            }}
          >
            <Save aria-hidden className="size-3.5" /> Save
          </Button>
          {(comparison || sweep) && <Badge tone="accent">includes intervention results</Badge>}
          {saved && <span className="text-[12px] text-teal">Saved “{saved}”.</span>}
        </div>
        {err && <Callout tone="danger">{err}</Callout>}
      </div>
    </div>
  );
}

function ExperimentsPanel() {
  const restore = useWorkspace((s) => s.restoreSession);
  const [items, setItems] = useState<ExperimentSession[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const refresh = useCallback(() => {
    listExperiments()
      .then((x) => {
        setItems(x);
        setErr(null);
      })
      .catch((e: unknown) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);
  useEffect(() => {
    refresh();
    window.addEventListener('attentionscope:experiments', refresh);
    return () => window.removeEventListener('attentionscope:experiments', refresh);
  }, [refresh]);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="sr-only"
          id="import-session"
          onChange={async (e) => {
            const f = e.target.files?.[0];
            e.target.value = '';
            if (!f) return;
            try {
              const s = await readSessionFile(f);
              await saveExperiment(s);
              setErr(null);
              refresh();
              log({ level: 'info', scope: 'session', message: `Imported experiment "${s.name}"` });
            } catch (er) {
              setErr(er instanceof Error ? er.message : String(er));
              log({
                level: 'error',
                scope: 'session',
                message: `Import rejected: ${er instanceof Error ? er.message : String(er)}`,
              });
            }
          }}
        />
        <Button size="sm" onClick={() => fileRef.current?.click()}>
          <Upload aria-hidden className="size-3.5" /> Import JSON
        </Button>
        <span className="text-[12px] text-muted">Stored locally in IndexedDB; nothing is uploaded.</span>
      </div>
      {err && <Callout tone="danger">{err}</Callout>}
      {items === null ? null : items.length === 0 ? (
        <EmptyState title="No saved experiments">
          Use “Save” in the Export tab to keep a reproducible snapshot.
        </EmptyState>
      ) : (
        <ul className="divide-y divide-line rounded-lg border border-line">
          {items.map((s) => (
            <li key={s.id} className="flex flex-wrap items-center gap-3 px-3 py-2 text-[12.5px]">
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium text-fg">{s.name}</div>
                <div className="font-mono text-[11px] text-muted">
                  {new Date(s.createdAt).toLocaleString()} ·{' '}
                  {s.source.kind === 'precomputed' ? `demo ${s.source.demoId}` : `live ${s.model.id}`} · L
                  {s.view.layer}H{s.view.head} · {s.input.tokenIds.length} tokens
                  {s.intervention ? ` · ${s.intervention.spec.kind}` : ''}
                </div>
              </div>
              <div className="flex gap-1">
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => void restore(s)}
                  aria-label={`Open ${s.name}`}
                >
                  <RotateCcw aria-hidden className="size-3.5" /> Open
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  aria-label={`Download ${s.name}`}
                  onClick={() =>
                    downloadText(
                      `${s.name.replace(/[^\w-]+/g, '_').slice(0, 60)}.attentionscope.json`,
                      JSON.stringify(s, null, 2),
                    )
                  }
                >
                  <Download aria-hidden className="size-3.5" />
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  aria-label={`Delete ${s.name}`}
                  onClick={async () => {
                    await deleteExperiment(s.id);
                    refresh();
                  }}
                >
                  <Trash2 aria-hidden className="size-3.5" />
                </Button>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function LogPanel() {
  const { entries, clear } = useLog();
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-[12px] text-muted">
          Every line is a real event: fetches, checksum checks, forward passes, exports.
        </p>
        <Button size="sm" variant="ghost" onClick={clear}>
          Clear
        </Button>
      </div>
      {entries.length === 0 ? (
        <p className="py-4 text-center text-[12.5px] text-muted">No events yet.</p>
      ) : (
        <ol
          className="max-h-56 overflow-auto rounded-lg border border-line bg-elevated p-2 font-mono text-[11.5px] scrollbar-thin"
          aria-live="polite"
        >
          {[...entries].reverse().map((e) => (
            <li key={e.id} className="flex gap-2 py-0.5">
              <span className="shrink-0 text-muted">{e.time.slice(11, 19)}</span>
              <span
                className={cn(
                  'shrink-0 uppercase',
                  e.level === 'error' ? 'text-danger' : e.level === 'warn' ? 'text-warn' : 'text-teal',
                )}
              >
                {e.scope}
              </span>
              <span className="min-w-0 break-words text-fg">{e.message}</span>
              {e.durationMs !== undefined && (
                <span className="ml-auto shrink-0 text-muted">{e.durationMs.toFixed(0)} ms</span>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export function BottomStrip() {
  const [open, setOpen] = useState(true);
  const [tab, setTab] = useState('export');
  const count = useLog((s) => s.entries.length);
  return (
    <section aria-label="Experiments, log and export" className="rounded-xl border border-line bg-panel">
      <Tabs.Root
        value={tab}
        onValueChange={(v) => {
          setTab(v);
          setOpen(true);
        }}
      >
        <div className="flex items-center border-b border-line px-2">
          <Tabs.List aria-label="Workspace tools" className="flex">
            {[
              ['export', 'Export'],
              ['experiments', 'Experiments'],
              ['log', `Log${count ? ` (${count})` : ''}`],
            ].map(([v, l]) => (
              <Tabs.Trigger
                key={v}
                value={v!}
                className="h-10 px-3 text-[12.5px] font-medium text-muted hover:text-fg data-[state=active]:text-fg"
              >
                {l}
              </Tabs.Trigger>
            ))}
          </Tabs.List>
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            aria-controls="bottom-strip-body"
            className="ml-auto inline-flex size-8 items-center justify-center rounded-md text-muted hover:bg-elevated hover:text-fg"
            aria-label={open ? 'Collapse panel' : 'Expand panel'}
          >
            <ChevronDown aria-hidden className={cn('size-4 transition-transform', !open && '-rotate-90')} />
          </button>
        </div>
        <div id="bottom-strip-body" hidden={!open} className="p-4">
          <Tabs.Content value="export" className="outline-none">
            <ExportPanel />
          </Tabs.Content>
          <Tabs.Content value="experiments" className="outline-none">
            <ExperimentsPanel />
          </Tabs.Content>
          <Tabs.Content value="log" className="outline-none">
            <LogPanel />
          </Tabs.Content>
        </div>
      </Tabs.Root>
    </section>
  );
}
