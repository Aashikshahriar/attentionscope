/**
 * IEEE-754 binary16 decoding. Every float16 value is exactly representable in
 * float32, so decoding through a 65 536-entry lookup table is lossless.
 */
let lut: Float32Array | null = null;

export function float16Bits(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

export function float16Table(): Float32Array {
  if (!lut) {
    lut = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) lut[i] = float16Bits(i);
  }
  return lut;
}

/** Decode little-endian float16 values into a new Float32Array. */
export function decodeFloat16(src: Uint16Array): Float32Array {
  const table = float16Table();
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = table[src[i]!]!;
  return out;
}
