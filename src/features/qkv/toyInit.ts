import type { ToyWeights } from '@/core/attention/toyAttention';
import { gaussian, mulberry32 } from '@/core/math/random';
import type { Matrix } from '@/core/math/matrix';

function randomMatrix(rand: () => number, r: number, c: number, scale: number): Matrix {
  return Array.from({ length: r }, () =>
    Array.from({ length: c }, () => Math.round(gaussian(rand) * scale * 100) / 100),
  );
}

/** Seeded toy weights (rounded to 2 decimals so they are easy to read and edit). */
export function initToyWeights(seed: number, T: number, d: number, dk: number, heads: number): ToyWeights {
  const rand = mulberry32(seed);
  const s = 1 / Math.sqrt(d);
  return {
    X: randomMatrix(rand, T, d, 1),
    WQ: Array.from({ length: heads }, () => randomMatrix(rand, d, dk, s * 1.6)),
    WK: Array.from({ length: heads }, () => randomMatrix(rand, d, dk, s * 1.6)),
    WV: Array.from({ length: heads }, () => randomMatrix(rand, d, dk, s)),
    WO: randomMatrix(rand, heads * dk, d, 1 / Math.sqrt(heads * dk)),
  };
}
