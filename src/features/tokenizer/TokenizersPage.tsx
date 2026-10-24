import { useEffect, useMemo, useRef, useState } from 'react';
import type { TokenRecord, TokenizedInput } from '@/types/contracts';
import type { TokenizerManifestEntry } from '@/types/artifacts';
import { fetchTokenizerManifest } from '@/services/runs';
import { dataWorker } from '@/services/workers';
import { assetUrl } from '@/services/paths';
import { Badge, Callout, Panel, Spinner, Stat, TokenText, cn } from '@/components/ui/primitives';
import { Switch } from '@/components/ui/controls';
import { TokenizedText } from './TokenView';

const SAMPLES: { id: string; label: string; text: string }[] = [
  { id: 'en', label: 'English', text: 'Attention lets every token read information from earlier tokens.' },
  { id: 'bn', label: 'Bengali', text: 'মনোযোগ প্রতিটি টোকেনকে আগের টোকেন থেকে তথ্য পড়তে দেয়।' },
  { id: 'hi', label: 'Hindi', text: 'ध्यान हर टोकन को पहले के टोकन से जानकारी पढ़ने देता है।' },
  { id: 'ko', label: 'Korean', text: '어텐션은 각 토큰이 이전 토큰의 정보를 읽게 합니다.' },
  { id: 'ar', label: 'Arabic (RTL)', text: 'يسمح الانتباه لكل رمز بقراءة المعلومات من الرموز السابقة.' },
  { id: 'code', label: 'Code', text: 'def softmax(x):\n    e = np.exp(x - x.max())\n    return e / e.sum()' },
];

const ROW = 30;

function VirtualTokenList({ tokens }: { tokens: TokenRecord[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const [scroll, setScroll] = useState(0);
  const height = Math.min(360, tokens.length * ROW);
  const start = Math.max(0, Math.floor(scroll / ROW) - 6);
  const end = Math.min(tokens.length, Math.ceil((scroll + height) / ROW) + 6);
  return (
    <div
      ref={ref}
      onScroll={(e) => setScroll(e.currentTarget.scrollTop)}
      className="overflow-auto rounded-lg border border-line scrollbar-thin"
      style={{ height }}
      role="list"
      tabIndex={0}
      aria-label={`${tokens.length} tokens`}
    >
      <div style={{ height: tokens.length * ROW, position: 'relative' }}>
        {tokens.slice(start, end).map((t) => (
          <div
            key={t.index}
            role="listitem"
            className="absolute inset-x-0 grid grid-cols-[3rem_minmax(0,1fr)_4.5rem_6rem] items-center gap-2 border-b border-line px-2.5 text-[12px]"
            style={{ top: t.index * ROW, height: ROW }}
          >
            <span className="font-mono text-muted">{t.index}</span>
            <span className="flex min-w-0 items-center gap-1.5">
              <TokenText text={t.text} className="truncate text-fg" />
              {t.partialCharacter && <Badge tone="warn">partial</Badge>}
            </span>
            <span className="text-right font-mono tabular text-fg">{t.id}</span>
            <span className="text-right font-mono text-[11px] text-muted">{t.byteEnd - t.byteStart} B</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Column({
  entry,
  text,
  prependBos,
  ready,
}: {
  entry: TokenizerManifestEntry;
  text: string;
  prependBos: boolean;
  ready: boolean;
}) {
  const [result, setResult] = useState<(TokenizedInput & { ms: number }) | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    const t = setTimeout(() => {
      dataWorker()
        .call('tokenize', { id: entry.id, text, prependBos, bos: entry.bos })
        .then((r) => !cancelled && (setResult(r), setErr(null)))
        .catch((e: unknown) => !cancelled && setErr(e instanceof Error ? e.message : String(e)));
    }, 120);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [entry, text, prependBos, ready]);
  const bytes = new TextEncoder().encode(text).length;
  const chars = Array.from(text).length;
  const n = result?.tokens.filter((t) => !t.synthetic).length ?? 0;
  const partial = result?.tokens.filter((t) => t.partialCharacter).length ?? 0;
  return (
    <Panel
      title={entry.name}
      id={`tok-${entry.id}`}
      actions={<span className="text-[11px] text-muted">{entry.usedBy}</span>}
    >
      {err ? (
        <Callout tone="danger">{err}</Callout>
      ) : !result ? (
        <div className="flex items-center gap-2 text-sm text-muted" role="status">
          <Spinner /> Loading tokenizer…
        </div>
      ) : (
        <div className="space-y-3">
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="tokens" value={n} />
            <Stat label="chars / token" value={n ? (chars / n).toFixed(2) : '—'} />
            <Stat label="bytes / token" value={n ? (bytes / n).toFixed(2) : '—'} />
            <Stat label="split chars" value={partial} />
          </dl>
          {result.normalizationChanged && (
            <Callout tone="info">NFC normalization changed this text before tokenization.</Callout>
          )}
          <TokenizedText input={result} />
          <VirtualTokenList tokens={result.tokens} />
          <p className="font-mono text-[10.5px] text-muted">
            {entry.repo}@{entry.revision.slice(0, 10)} · {entry.license} · tokenized in {result.ms.toFixed(1)}{' '}
            ms (Web Worker)
          </p>
        </div>
      )}
    </Panel>
  );
}

export default function TokenizersPage() {
  const [entries, setEntries] = useState<TokenizerManifestEntry[]>([]);
  const [ready, setReady] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [sample, setSample] = useState('en');
  const [text, setText] = useState(SAMPLES[0]!.text);
  const [prependBos, setPrependBos] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchTokenizerManifest()
      .then(async (list) => {
        await Promise.all(
          list.map((e) =>
            dataWorker().call('loadTokenizer', {
              id: e.id,
              url: assetUrl(e.path),
              sha256: e.sha256,
              name: e.name,
              repo: e.repo,
              revision: e.revision,
            }),
          ),
        );
        if (!cancelled) {
          setEntries(list);
          setReady(true);
        }
      })
      .catch((e: unknown) => !cancelled && setErr(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, []);

  const charCount = useMemo(() => Array.from(text).length, [text]);

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-10">
      <div className="max-w-3xl space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight text-fg">Tokenizer comparison</h1>
        <p className="text-[15px] leading-relaxed text-muted">
          The two byte-level BPE tokenizers used in AttentionScope, run in your browser and verified to
          reproduce the Hugging Face (Rust) implementation token-for-token. Scripts underrepresented in
          training data need many more tokens — and every extra token is another row in the attention matrix.
        </p>
      </div>
      <Panel title="Input" id="tok-input">
        <div className="space-y-3">
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Sample texts">
            {SAMPLES.map((s) => (
              <button
                key={s.id}
                type="button"
                aria-pressed={sample === s.id}
                onClick={() => {
                  setSample(s.id);
                  setText(s.text);
                }}
                className={cn(
                  'rounded-md border px-2.5 py-1 text-[12.5px]',
                  sample === s.id
                    ? 'border-accent bg-accent-soft text-accent'
                    : 'border-line bg-elevated text-muted hover:text-fg',
                )}
              >
                {s.label}
              </button>
            ))}
          </div>
          <label htmlFor="tok-text" className="sr-only">
            Text to tokenize
          </label>
          <textarea
            id="tok-text"
            value={text}
            dir="auto"
            maxLength={20000}
            onChange={(e) => {
              setText(e.target.value);
              setSample('');
            }}
            rows={3}
            className="w-full rounded-lg border border-line bg-elevated px-3 py-2 text-[14px] text-fg"
          />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span className="text-[12px] text-muted">
              {charCount} characters · {new TextEncoder().encode(text).length} UTF-8 bytes · processed locally
            </span>
            <div className="w-72">
              <Switch
                id="tok-bos"
                label="Prepend <|endoftext|>"
                checked={prependBos}
                onChange={setPrependBos}
              />
            </div>
          </div>
        </div>
      </Panel>
      {err && (
        <Callout tone="danger" title="Tokenizers could not be loaded">
          {err}
        </Callout>
      )}
      <div className="grid gap-4 lg:grid-cols-2">
        {entries.map((e) => (
          <Column key={e.id} entry={e} text={text} prependBos={prependBos} ready={ready} />
        ))}
        {!ready && !err && (
          <div className="flex items-center gap-2 text-sm text-muted" role="status">
            <Spinner /> Downloading and verifying tokenizers (≈3.5 MB)…
          </div>
        )}
      </div>
      <Callout title="Reading the results">
        Token boundaries are shown on the original text using exact byte offsets. A token marked “partial”
        ends or starts inside a multi-byte UTF-8 character, so its own text is not a valid string — common for
        Bengali, Hindi and Korean under English-centric vocabularies. Right-to-left text is displayed with
        bidirectional isolation; token order is always logical (reading) order.
      </Callout>
    </div>
  );
}
