/** SHA-256 of a buffer as lowercase hex (Web Crypto; available in secure contexts and workers). */
export async function sha256Hex(data: ArrayBuffer | Uint8Array): Promise<string> {
  const view = data instanceof Uint8Array ? data : new Uint8Array(data);
  // Copy into a fresh ArrayBuffer so SharedArrayBuffer-backed views are accepted.
  const digest = await crypto.subtle.digest('SHA-256', view.slice().buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

export class ChecksumError extends Error {
  constructor(
    readonly resource: string,
    readonly expected: string,
    readonly actual: string,
  ) {
    super(
      `Checksum mismatch for ${resource}: expected ${expected.slice(0, 12)}…, got ${actual.slice(0, 12)}…`,
    );
    this.name = 'ChecksumError';
  }
}

export async function verifySha256(resource: string, data: ArrayBuffer | Uint8Array, expected: string) {
  const actual = await sha256Hex(data);
  if (actual !== expected) throw new ChecksumError(resource, expected, actual);
  return actual;
}
