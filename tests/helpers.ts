import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

export const ROOT = resolve(__dirname, '..');

export function readJson<T>(rel: string): T {
  return JSON.parse(readFileSync(resolve(ROOT, rel), 'utf8')) as T;
}

/** Copy a file into a standalone ArrayBuffer (avoids Node Buffer pooling). */
export function readArrayBuffer(rel: string): ArrayBuffer {
  const buf = readFileSync(resolve(ROOT, rel));
  const out = new ArrayBuffer(buf.byteLength);
  new Uint8Array(out).set(buf);
  return out;
}

export function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number | null>): number {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} vs ${b.length}`);
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i];
    if (y === null || y === undefined) {
      if (x !== Number.NEGATIVE_INFINITY) return Number.POSITIVE_INFINITY;
      continue;
    }
    m = Math.max(m, Math.abs(x - y));
  }
  return m;
}

export const flat = (m: number[][]): number[] => m.flat();
