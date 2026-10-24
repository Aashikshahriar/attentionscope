import { useEffect, useState } from 'react';
import type KatexDefault from 'katex';
import { cn } from './primitives';

type Katex = typeof KatexDefault;
let katexPromise: Promise<Katex> | null = null;
let katexModule: Katex | null = null;

function loadKatex(): Promise<Katex> {
  katexPromise ??= Promise.all([import('katex'), import('katex/dist/katex.min.css')]).then(([m]) => {
    katexModule = m.default;
    return m.default;
  });
  return katexPromise;
}

/** KaTeX rendering (lazy-loaded). Input is always author-controlled TeX, never user text. */
export function Tex({
  children,
  block = false,
  className,
}: {
  children: string;
  block?: boolean;
  className?: string;
}) {
  const [k, setK] = useState<Katex | null>(katexModule);
  useEffect(() => {
    if (!k) void loadKatex().then(setK);
  }, [k]);
  const Tag = block ? 'div' : 'span';
  if (!k) {
    return (
      <Tag
        className={cn(block && 'my-2 min-h-8', 'font-mono text-[12px] text-muted', className)}
        aria-busy="true"
      >
        {block ? '' : children}
      </Tag>
    );
  }
  const html = k.renderToString(children, {
    displayMode: block,
    throwOnError: false,
    output: 'htmlAndMathml',
    strict: 'ignore',
  });
  return <Tag className={cn(block && 'my-2', className)} dangerouslySetInnerHTML={{ __html: html }} />;
}
