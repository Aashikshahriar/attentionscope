import type { AttentionRun, TokenRecord, TokenizedInput } from '@/types/contracts';
import { Badge, Callout, TokenText, cn } from '@/components/ui/primitives';

const PALETTE = ['bg-accent-soft', 'bg-teal-soft'];

/** Raw text with token boundaries overlaid, using exact UTF-16 spans. */
export function TokenizedText({
  input,
  selected,
  onSelect,
}: {
  input: TokenizedInput;
  selected?: number;
  onSelect?: (i: number) => void;
}) {
  const text = input.normalizedText ?? input.text;
  return (
    <p
      dir="auto"
      className="whitespace-pre-wrap break-words rounded-lg border border-line bg-elevated p-3 font-mono text-[13.5px] leading-7"
    >
      {input.tokens.map((t, i) => {
        const shown = t.synthetic ? t.text : text.slice(t.u16Start, t.u16End);
        const Tag = onSelect ? 'button' : 'span';
        return (
          <Tag
            key={i}
            type={onSelect ? 'button' : undefined}
            onClick={onSelect ? () => onSelect(i) : undefined}
            title={`#${i} id ${t.id}${t.partialCharacter ? ' (splits a character)' : ''}`}
            className={cn(
              'bidi-isolate rounded-[3px] px-px py-0.5',
              t.synthetic ? 'border border-dashed border-teal/60 text-teal' : PALETTE[i % 2],
              t.partialCharacter && 'underline decoration-warn decoration-dotted',
              selected === i && 'ring-2 ring-accent',
            )}
          >
            {shown.replace(/\n/g, '↵\n') || '∅'}
          </Tag>
        );
      })}
    </p>
  );
}

export function TokenTable({
  tokens,
  selected,
  onSelect,
}: {
  tokens: TokenRecord[];
  selected?: number;
  onSelect?: (i: number) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border border-line scrollbar-thin">
      <table className="w-full border-collapse text-left text-[12px]">
        <caption className="sr-only">Tokens with ids, vocabulary pieces, bytes and offsets</caption>
        <thead className="bg-elevated text-muted">
          <tr>
            {['#', 'Token', 'ID', 'Vocabulary piece', 'UTF-8 bytes', 'Byte span', 'Flags'].map((h) => (
              <th key={h} scope="col" className="whitespace-nowrap px-2.5 py-1.5 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {tokens.map((t) => (
            <tr
              key={t.index}
              className={cn(
                'border-t border-line',
                selected === t.index && 'bg-accent-soft',
                onSelect && 'cursor-pointer hover:bg-elevated',
              )}
              onClick={onSelect ? () => onSelect(t.index) : undefined}
            >
              <td className="px-2.5 py-1 font-mono text-muted">{t.index}</td>
              <td className="px-2.5 py-1 text-fg">
                <TokenText text={t.text} />
              </td>
              <td className="px-2.5 py-1 font-mono tabular text-fg">{t.id}</td>
              <td className="px-2.5 py-1 font-mono text-muted">
                <span className="bidi-isolate">{t.piece}</span>
              </td>
              <td className="px-2.5 py-1 font-mono text-[11px] text-muted">
                {t.bytesHex.match(/../g)?.join(' ') ?? ''}
              </td>
              <td className="whitespace-nowrap px-2.5 py-1 font-mono text-muted">
                {t.synthetic ? '—' : `[${t.byteStart}, ${t.byteEnd})`}
              </td>
              <td className="px-2.5 py-1">
                <div className="flex flex-wrap gap-1">
                  {t.special && <Badge tone="teal">special</Badge>}
                  {t.synthetic && <Badge tone="teal">prepended</Badge>}
                  {t.partialCharacter && <Badge tone="warn">partial char</Badge>}
                  {/^[ Ġ]/.test(t.text) && !t.special && <Badge>leading space</Badge>}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function TokensTab({
  run,
  selected,
  onSelect,
}: {
  run: AttentionRun;
  selected: number;
  onSelect: (i: number) => void;
}) {
  const partial = run.input.tokens.filter((t) => t.partialCharacter).length;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-muted">
        <span>
          {run.input.tokens.length} tokens · {run.tokenizer.name}
        </span>
        <span aria-hidden>·</span>
        <span>attention mask: all 1 (single sequence, no padding)</span>
        {run.input.normalization === 'NFC' && (
          <>
            <span aria-hidden>·</span>
            <span>
              NFC normalization
              {run.input.normalizationChanged
                ? ' changed the text; offsets refer to the normalized text'
                : ' (text unchanged)'}
            </span>
          </>
        )}
      </div>
      <TokenizedText input={run.input} selected={selected} onSelect={onSelect} />
      <TokenTable tokens={run.input.tokens} selected={selected} onSelect={onSelect} />
      <div className="grid gap-3 md:grid-cols-2">
        <Callout title="Why tokens are not words">
          Byte-level BPE splits text into frequent byte sequences learned from the training corpus. A leading
          space is part of the following token (shown as ·), common words are single tokens, and rare words,
          numbers and non-Latin scripts are split into several pieces. Attention operates on these tokens, not
          on words.
        </Callout>
        <Callout title="Offsets and special tokens">
          Byte spans are computed from the exact UTF-8 byte stream and always tile the input.
          {partial > 0 &&
            ` ${partial} token(s) here split a multi-byte character; such a token's text alone is not valid UTF-8 (shown with U+FFFD).`}
          {run.tokenizer.prependBos &&
            ` The first token ${run.tokenizer.bosToken?.text ?? ''} was prepended (not in your text); many heads use it as an attention “sink”.`}
        </Callout>
      </div>
    </div>
  );
}
