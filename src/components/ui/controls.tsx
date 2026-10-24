import { Select as RSelect, Switch as RSwitch, ToggleGroup, Tooltip as RTooltip } from 'radix-ui';
import { Check, ChevronDown, Minus, Plus } from 'lucide-react';
import type { ReactNode } from 'react';
import { cn } from './primitives';

export interface Option<T extends string> {
  value: T;
  label: ReactNode;
  description?: string;
  disabled?: boolean;
}

export function Select<T extends string>({
  value,
  onChange,
  options,
  label,
  id,
  className,
  placeholder,
}: {
  value: T | undefined;
  onChange: (v: T) => void;
  options: Option<T>[];
  label: string;
  id?: string;
  className?: string;
  placeholder?: string;
}) {
  return (
    <RSelect.Root value={value} onValueChange={(v) => onChange(v as T)}>
      <RSelect.Trigger
        id={id}
        aria-label={label}
        className={cn(
          'inline-flex h-9 w-full items-center justify-between gap-2 rounded-lg border border-line bg-elevated px-3 text-left text-sm text-fg hover:border-line-strong data-[placeholder]:text-muted',
          className,
        )}
      >
        <span className="min-w-0 truncate">
          <RSelect.Value placeholder={placeholder} />
        </span>
        <RSelect.Icon>
          <ChevronDown aria-hidden className="size-4 text-muted" />
        </RSelect.Icon>
      </RSelect.Trigger>
      <RSelect.Portal>
        <RSelect.Content
          position="popper"
          sideOffset={4}
          className="z-50 max-h-[min(24rem,var(--radix-select-content-available-height))] min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-lg border border-line-strong bg-panel shadow-2xl"
        >
          <RSelect.Viewport className="p-1">
            {options.map((o) => (
              <RSelect.Item
                key={o.value}
                value={o.value}
                disabled={o.disabled}
                className="relative flex cursor-default select-none flex-col rounded-md py-1.5 pl-7 pr-3 text-sm text-fg outline-none data-[disabled]:opacity-40 data-[highlighted]:bg-elevated"
              >
                <RSelect.ItemIndicator className="absolute left-2 top-2">
                  <Check aria-hidden className="size-3.5 text-accent" />
                </RSelect.ItemIndicator>
                <RSelect.ItemText>{o.label}</RSelect.ItemText>
                {o.description && <span className="text-[11.5px] text-muted">{o.description}</span>}
              </RSelect.Item>
            ))}
          </RSelect.Viewport>
        </RSelect.Content>
      </RSelect.Portal>
    </RSelect.Root>
  );
}

export function Switch({
  checked,
  onChange,
  label,
  id,
  description,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  id: string;
  description?: ReactNode;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div className="min-w-0">
        <label htmlFor={id} className="text-[13px] text-fg">
          {label}
        </label>
        {description && <p className="text-[11.5px] leading-snug text-muted">{description}</p>}
      </div>
      <RSwitch.Root
        id={id}
        checked={checked}
        disabled={disabled}
        onCheckedChange={onChange}
        className="relative mt-0.5 h-5 w-9 shrink-0 rounded-full border border-line-strong bg-elevated transition-colors data-[state=checked]:border-accent data-[state=checked]:bg-accent disabled:opacity-40"
      >
        <RSwitch.Thumb className="block size-3.5 translate-x-0.5 rounded-full bg-fg shadow transition-transform data-[state=checked]:translate-x-[18px] data-[state=checked]:bg-accent-fg" />
      </RSwitch.Root>
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
  className,
  size = 'md',
}: {
  value: T;
  onChange: (v: T) => void;
  options: Option<T>[];
  label: string;
  className?: string;
  size?: 'sm' | 'md';
}) {
  return (
    <ToggleGroup.Root
      type="single"
      value={value}
      onValueChange={(v) => v && onChange(v as T)}
      aria-label={label}
      className={cn('inline-flex rounded-lg border border-line bg-elevated p-0.5', className)}
    >
      {options.map((o) => (
        <ToggleGroup.Item
          key={o.value}
          value={o.value}
          disabled={o.disabled}
          title={o.description}
          className={cn(
            'flex-1 rounded-md px-2.5 font-medium text-muted transition-colors hover:text-fg data-[state=on]:bg-panel data-[state=on]:text-fg data-[state=on]:shadow-sm disabled:opacity-40',
            size === 'sm' ? 'h-7 text-xs' : 'h-8 text-[13px]',
          )}
        >
          {o.label}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}

export function Stepper({
  value,
  onChange,
  min,
  max,
  label,
  id,
  format = (v) => String(v),
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  label: string;
  id: string;
  format?: (v: number) => string;
}) {
  const clamp = (v: number) => Math.min(max, Math.max(min, v));
  return (
    <div className="flex h-9 items-stretch overflow-hidden rounded-lg border border-line bg-elevated">
      <button
        type="button"
        aria-label={`Decrease ${label}`}
        disabled={value <= min}
        onClick={() => onChange(clamp(value - 1))}
        className="px-2 text-muted hover:bg-panel hover:text-fg disabled:opacity-30"
      >
        <Minus aria-hidden className="size-3.5" />
      </button>
      <input
        id={id}
        type="number"
        inputMode="numeric"
        aria-label={label}
        min={min}
        max={max}
        value={value}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) onChange(clamp(Math.trunc(v)));
        }}
        className="w-full min-w-0 border-x border-line bg-transparent text-center font-mono text-sm text-fg tabular [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none"
      />
      <span className="sr-only">{format(value)}</span>
      <button
        type="button"
        aria-label={`Increase ${label}`}
        disabled={value >= max}
        onClick={() => onChange(clamp(value + 1))}
        className="px-2 text-muted hover:bg-panel hover:text-fg disabled:opacity-30"
      >
        <Plus aria-hidden className="size-3.5" />
      </button>
    </div>
  );
}

export function Tip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <RTooltip.Root delayDuration={250}>
      <RTooltip.Trigger asChild>{children}</RTooltip.Trigger>
      <RTooltip.Portal>
        <RTooltip.Content
          sideOffset={6}
          className="z-50 max-w-xs rounded-md border border-line-strong bg-panel px-2.5 py-1.5 text-xs leading-snug text-fg shadow-xl"
        >
          {content}
          <RTooltip.Arrow className="fill-[var(--line-strong)]" />
        </RTooltip.Content>
      </RTooltip.Portal>
    </RTooltip.Root>
  );
}

export const TooltipProvider = RTooltip.Provider;
