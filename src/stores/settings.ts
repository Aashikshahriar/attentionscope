import { create } from 'zustand';
import { createJSONStorage, persist, type StateStorage } from 'zustand/middleware';

export type ThemeSetting = 'dark' | 'light' | 'system';
export type MotionSetting = 'system' | 'reduce' | 'full';
export type Colormap = 'viridis' | 'magma' | 'cividis' | 'greys';
export type ColorScale = 'linear' | 'sqrt';

export interface SettingsState {
  theme: ThemeSetting;
  motion: MotionSetting;
  highContrast: boolean;
  colormap: Colormap;
  colorScale: ColorScale;
  /** user consent to keep model weights in Cache Storage */
  cacheModel: boolean;
  set: (patch: Partial<Omit<SettingsState, 'set'>>) => void;
}

/** localStorage can throw (private mode, blocked storage); fall back to memory. */
const memory = new Map<string, string>();
const safeStorage: StateStorage = {
  getItem: (k) => {
    try {
      return window.localStorage.getItem(k);
    } catch {
      return memory.get(k) ?? null;
    }
  },
  setItem: (k, v) => {
    try {
      window.localStorage.setItem(k, v);
    } catch {
      memory.set(k, v);
    }
  },
  removeItem: (k) => {
    try {
      window.localStorage.removeItem(k);
    } catch {
      memory.delete(k);
    }
  },
};

export const useSettings = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'dark',
      motion: 'system',
      highContrast: false,
      colormap: 'viridis',
      colorScale: 'linear',
      cacheModel: false,
      set: (patch) => set(patch),
    }),
    {
      name: 'attentionscope-settings',
      version: 1,
      storage: createJSONStorage(() => safeStorage),
      partialize: ({ set: _set, ...rest }) => rest,
    },
  ),
);

export function resolvedTheme(theme: ThemeSetting): 'dark' | 'light' {
  if (theme !== 'system') return theme;
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export function prefersReducedMotion(motion: MotionSetting): boolean {
  if (motion !== 'system') return motion === 'reduce';
  return !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}
