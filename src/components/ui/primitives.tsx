import { clsx, type ClassValue } from 'clsx';
import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from 'react';
import { AlertTriangle, Info, OctagonAlert, CheckCircle2 } from 'lucide-react';

// eslint-disable-next-line react-refresh/only-export-components -- tiny shared helper
export const cn = (...v: ClassValue[]) => clsx(v);

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger';
type Size = 'sm' | 'md' | 'lg';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const variants: Record<Variant, string> = {
  primary: 'bg-accent text-accent-fg hover:brightness-110 shadow-[0_0_0_1px_var(--accent)]',
  secondary: 'bg-elevated text-fg border border-line hover:border-line-strong',
  ghost: 'text-muted hover:text-fg hover:bg-elevated',
  danger: 'bg-danger-soft text-danger border border-danger/40 hover:bg-danger/20',
};
const sizes: Record<Size, string> = {
  sm: 'h-8 px-2.5 text-xs gap-1.5',
  md: 'h-9 px-3.5 text-sm gap-2',
  lg: 'h-11 px-5 text-[15px] gap-2',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', className, type = 'button', ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        'inline-flex items-center justify-center rounded-lg font-medium transition-[filter,background,border-color,color] duration-150 disabled:pointer-events-none disabled:opacity-45 whitespace-nowrap',
        variants[variant],
        sizes[size],
        className,
      )}
      {...rest}
    />
  );
});

export function Badge({
  tone = 'neutral',
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLSpanElement> & { tone?: 'neutral' | 'accent' | 'teal' | 'warn' | 'danger' }) {
  const tones = {
    neutral: 'bg-elevated text-muted border-line',
    accent: 'bg-accent-soft text-accent border-accent/30',
    teal: 'bg-teal-soft text-teal border-teal/30',
    warn: 'bg-warn-soft text-warn border-warn/30',
    danger: 'bg-danger-soft text-danger border-danger/30',
  };
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium uppercase tracking-wide',
        tones[tone],
        className,
      )}
      {...rest}
    >
      {children}
    </span>
  );
}

export function Panel({
  title,
  actions,
  children,
  className,
  bodyClassName,
  id,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  id?: string;
}) {
  return (
    <section
      aria-labelledby={title && id ? `${id}-title` : undefined}
      className={cn('rounded-xl border border-line bg-panel', className)}
    >
      {title && (
        <header className="flex items-center justify-between gap-2 border-b border-line px-3.5 py-2.5">
          <h2 id={id ? `${id}-title` : undefined} className="text-[13px] font-semibold tracking-wide text-fg">
            {title}
          </h2>
          {actions}
        </header>
      )}
      <div className={cn('p-3.5', bodyClassName)}>{children}</div>
    </section>
  );
}

export function Callout({
  tone = 'info',
  title,
  children,
  className,
}: {
  tone?: 'info' | 'warn' | 'danger' | 'success';
  title?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  const map = {
    info: { cls: 'border-accent/30 bg-accent-soft', icon: Info, ic: 'text-accent' },
    warn: { cls: 'border-warn/35 bg-warn-soft', icon: AlertTriangle, ic: 'text-warn' },
    danger: { cls: 'border-danger/35 bg-danger-soft', icon: OctagonAlert, ic: 'text-danger' },
    success: { cls: 'border-teal/35 bg-teal-soft', icon: CheckCircle2, ic: 'text-teal' },
  }[tone];
  const Icon = map.icon;
  return (
    <div
      role={tone === 'danger' ? 'alert' : undefined}
      className={cn(
        'flex gap-2.5 rounded-lg border px-3 py-2.5 text-[13px] leading-relaxed',
        map.cls,
        className,
      )}
    >
      <Icon aria-hidden className={cn('mt-0.5 size-4 shrink-0', map.ic)} />
      <div className="min-w-0">
        {title && <div className="mb-0.5 font-semibold text-fg">{title}</div>}
        <div className="text-muted">{children}</div>
      </div>
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded border border-line-strong bg-elevated px-1.5 py-px font-mono text-[11px] text-fg">
      {children}
    </kbd>
  );
}

export function ProgressBar({ value, label }: { value: number | null; label: string }) {
  const pct = value === null ? null : Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct ?? undefined}
      className="h-1.5 w-full overflow-hidden rounded-full bg-elevated"
    >
      <div
        className={cn(
          'h-full rounded-full bg-accent transition-[width] duration-200',
          pct === null && 'animate-pulse-soft w-1/3',
        )}
        style={pct === null ? undefined : { width: `${pct}%` }}
      />
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cn(
        'inline-block size-4 animate-spin rounded-full border-2 border-line-strong border-t-accent',
        className,
      )}
    />
  );
}

export function EmptyState({
  icon,
  title,
  children,
}: {
  icon?: ReactNode;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-14 text-center">
      {icon && <div className="text-muted">{icon}</div>}
      <p className="text-sm font-semibold text-fg">{title}</p>
      {children && <div className="max-w-md text-[13px] leading-relaxed text-muted">{children}</div>}
    </div>
  );
}

export function Field({
  label,
  htmlFor,
  hint,
  children,
}: {
  label: string;
  htmlFor?: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className="block text-[12px] font-medium text-muted">
        {label}
      </label>
      {children}
      {hint && <p className="text-[11.5px] leading-snug text-muted">{hint}</p>}
    </div>
  );
}

export function Stat({ label, value, mono = true }: { label: string; value: ReactNode; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] uppercase tracking-wide text-muted">{label}</dt>
      <dd className={cn('truncate text-[13px] text-fg', mono && 'font-mono tabular')}>{value}</dd>
    </div>
  );
}

/** Render a token's text with visible whitespace markers. */
export function TokenText({ text, className }: { text: string; className?: string }) {
  const parts: ReactNode[] = [];
  let buf = '';
  const flush = (k: number) => {
    if (buf) parts.push(<span key={`t${k}`}>{buf}</span>);
    buf = '';
  };
  Array.from(text).forEach((ch, i) => {
    const mark = ch === ' ' ? '·' : ch === '\n' ? '↵' : ch === '\t' ? '⇥' : ch === '\r' ? '␍' : null;
    if (mark) {
      flush(i);
      parts.push(
        <span key={`w${i}`} className="token-ws" aria-hidden>
          {mark}
        </span>,
      );
    } else buf += ch;
  });
  flush(-1);
  const sr = text.replace(/ /g, ' space ').replace(/\n/g, ' newline ').replace(/\t/g, ' tab ').trim();
  return (
    <span className={cn('bidi-isolate font-mono', className)} dir="auto">
      <span aria-hidden>{parts.length ? parts : <span className="token-ws">∅</span>}</span>
      <span className="sr-only">{sr || 'empty token'}</span>
    </span>
  );
}
