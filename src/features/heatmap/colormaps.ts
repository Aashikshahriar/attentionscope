import {
  interpolateCividis,
  interpolateGreys,
  interpolateMagma,
  interpolateViridis,
} from 'd3-scale-chromatic';
import type { Colormap, ColorScale } from '@/stores/settings';

/** Perceptually uniform, colorblind-safe sequential maps (low = dark, high = bright). */
export const COLORMAPS: Record<Colormap, { label: string; fn: (t: number) => string }> = {
  viridis: { label: 'Viridis', fn: interpolateViridis },
  cividis: { label: 'Cividis', fn: interpolateCividis },
  magma: { label: 'Magma', fn: interpolateMagma },
  greys: { label: 'Greyscale', fn: (t) => interpolateGreys(1 - t) },
};

const LUT_SIZE = 256;
const luts = new Map<Colormap, string[]>();
const lums = new Map<Colormap, number[]>();

function parse(c: string): [number, number, number] {
  const m = c.match(/\d+(\.\d+)?/g);
  if (c.startsWith('#')) {
    return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
  }
  return [Number(m?.[0] ?? 0), Number(m?.[1] ?? 0), Number(m?.[2] ?? 0)];
}

function lut(map: Colormap): string[] {
  let l = luts.get(map);
  if (!l) {
    l = Array.from({ length: LUT_SIZE }, (_, i) => COLORMAPS[map].fn(i / (LUT_SIZE - 1)));
    luts.set(map, l);
    lums.set(
      map,
      l.map((c) => {
        const [r, g, b] = parse(c).map((x) => {
          const s = x / 255;
          return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
        }) as [number, number, number];
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
      }),
    );
  }
  return l;
}

/** Display transform only; the underlying values are never changed. */
export function displayValue(v: number, scale: ColorScale = 'linear'): number {
  const c = Math.min(1, Math.max(0, v));
  return scale === 'sqrt' ? Math.sqrt(c) : c;
}

export function colorFor(map: Colormap, v: number, scale: ColorScale = 'linear'): string {
  const l = lut(map);
  return l[Math.round(displayValue(v, scale) * (LUT_SIZE - 1))]!;
}

/** Readable text color on top of a cell of value v. */
export function textOn(map: Colormap, v: number, scale: ColorScale = 'linear'): string {
  lut(map);
  const lum = lums.get(map)![Math.round(displayValue(v, scale) * (LUT_SIZE - 1))]!;
  return lum > 0.35 ? '#0b0f17' : '#f4f7fb';
}

export function gradientCss(map: Colormap): string {
  const stops = Array.from({ length: 11 }, (_, i) => `${COLORMAPS[map].fn(i / 10)} ${i * 10}%`);
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}

/** Diverging map for signed effects (e.g. Δ log-probability). Blue = negative, orange = positive. */
export function divergingColor(v: number, maxAbs: number): string {
  if (!Number.isFinite(v) || maxAbs <= 0) return 'rgb(128,128,128)';
  const t = Math.max(-1, Math.min(1, v / maxAbs));
  // endpoints chosen for colorblind safety (blue/orange)
  const neg = [33, 102, 172];
  const mid = [235, 235, 235];
  const pos = [214, 96, 77];
  const [a, b, f] = t < 0 ? [mid, neg, -t] : [mid, pos, t];
  const c = a.map((x, i) => Math.round(x + (b[i]! - x) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** Black or white text, whichever has the higher WCAG contrast on an rgb()/hex background. */
export function readableOn(bg: string): string {
  const [r, g, b] = parse(bg).map((x) => {
    const c = x / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
  const L = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return (L + 0.05) / 0.05 >= 1.05 / (L + 0.05) ? '#000000' : '#ffffff';
}
