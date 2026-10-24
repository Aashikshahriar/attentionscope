const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI);

/**
 * Error function to (near) double precision.
 *
 * Uses the everywhere-convergent series
 *   erf(x) = 2/sqrt(pi) * exp(-x^2) * sum_n 2^n x^(2n+1) / (1*3*...*(2n+1)),
 * whose terms are all positive (no cancellation). For |x| >= 6, erf(x) equals
 * +-1 in double precision.
 */
export function erf(x: number): number {
  if (Number.isNaN(x)) return Number.NaN;
  const ax = Math.abs(x);
  if (ax >= 6) return x > 0 ? 1 : -1;
  const x2 = ax * ax;
  let term = ax;
  let sum = ax;
  for (let n = 1; n < 400; n++) {
    term *= (2 * x2) / (2 * n + 1);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  const r = TWO_OVER_SQRT_PI * Math.exp(-x2) * sum;
  return x < 0 ? -r : r;
}

/** Exact (erf-based) GELU, as used by GPT-NeoX `hidden_act: "gelu"`. */
export function geluErf(x: number): number {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

/** Tanh-approximate GELU ("gelu_new"), as used by GPT-2. */
export function geluTanh(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}
