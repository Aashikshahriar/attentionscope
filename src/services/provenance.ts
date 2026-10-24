import type { Provenance } from '@/types/contracts';

export const APP_INFO: Provenance['app'] = {
  name: 'AttentionScope',
  version: __APP_VERSION__,
  commit: __BUILD_COMMIT__,
};

export function nowIso(): string {
  return new Date().toISOString();
}

export function randomId(prefix: string): string {
  const bytes = crypto.getRandomValues(new Uint8Array(6));
  return `${prefix}-${Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')}`;
}
