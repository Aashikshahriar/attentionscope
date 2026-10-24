import { Popover } from 'radix-ui';
import { SlidersHorizontal } from 'lucide-react';
import {
  useSettings,
  type Colormap,
  type ColorScale,
  type MotionSetting,
  type ThemeSetting,
} from '@/stores/settings';
import { Segmented, Select, Switch } from '@/components/ui/controls';
import { COLORMAPS, gradientCss } from '@/features/heatmap/colormaps';

export function SettingsMenu() {
  const s = useSettings();
  return (
    <Popover.Root>
      <Popover.Trigger
        className="inline-flex size-9 items-center justify-center rounded-lg text-muted hover:bg-elevated hover:text-fg"
        aria-label="Display and accessibility settings"
      >
        <SlidersHorizontal aria-hidden className="size-[18px]" />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={8}
          className="z-50 w-[min(92vw,20rem)] space-y-4 rounded-xl border border-line-strong bg-panel p-4 shadow-2xl"
        >
          <h2 className="text-sm font-semibold text-fg">Display &amp; accessibility</h2>
          <div className="space-y-1.5">
            <span id="set-theme" className="text-[12px] font-medium text-muted">
              Theme
            </span>
            <Segmented<ThemeSetting>
              label="Theme"
              value={s.theme}
              onChange={(theme) => s.set({ theme })}
              className="w-full"
              options={[
                { value: 'dark', label: 'Dark' },
                { value: 'light', label: 'Light' },
                { value: 'system', label: 'System' },
              ]}
            />
          </div>
          <div className="space-y-1.5">
            <span className="text-[12px] font-medium text-muted">Motion</span>
            <Segmented<MotionSetting>
              label="Motion"
              value={s.motion}
              onChange={(motion) => s.set({ motion })}
              className="w-full"
              options={[
                { value: 'system', label: 'System' },
                { value: 'reduce', label: 'Reduced' },
                { value: 'full', label: 'Full' },
              ]}
            />
          </div>
          <Switch
            id="set-contrast"
            label="High contrast"
            description="Stronger borders and text."
            checked={s.highContrast}
            onChange={(highContrast) => s.set({ highContrast })}
          />
          <div className="space-y-1.5">
            <label htmlFor="set-cmap" className="text-[12px] font-medium text-muted">
              Heatmap colormap (colorblind-safe)
            </label>
            <Select<Colormap>
              id="set-cmap"
              label="Heatmap colormap"
              value={s.colormap}
              onChange={(colormap) => s.set({ colormap })}
              options={(Object.keys(COLORMAPS) as Colormap[]).map((k) => ({
                value: k,
                label: COLORMAPS[k].label,
              }))}
            />
            <div aria-hidden className="h-2 rounded" style={{ background: gradientCss(s.colormap) }} />
          </div>
          <div className="space-y-1.5">
            <span className="text-[12px] font-medium text-muted">Color scale (display only)</span>
            <Segmented<ColorScale>
              label="Color scale"
              value={s.colorScale}
              onChange={(colorScale) => s.set({ colorScale })}
              className="w-full"
              options={[
                { value: 'linear', label: 'Linear' },
                { value: 'sqrt', label: '√ (boost small)' },
              ]}
            />
            <p className="text-[11.5px] text-muted">
              Changes colors only. Numbers shown are always the exact values.
            </p>
          </div>
          <Popover.Arrow className="fill-[var(--line-strong)]" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
