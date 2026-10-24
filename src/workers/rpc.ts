/**
 * Minimal typed request/response protocol between the UI and Web Workers.
 *
 * Long computations call `ctx.checkpoint()` between units of work. That yields
 * to the worker's event loop, so a `cancel` message can be received, and then
 * throws `CancelledError` if the request was cancelled.
 */

export type ApiShape = Record<string, { params: unknown; result: unknown; progress?: unknown }>;

type RequestMsg = { id: number; method: string; params: unknown } | { id: number; cancel: true };
type ResponseMsg =
  | { id: number; kind: 'result'; result: unknown }
  | { id: number; kind: 'error'; error: { name: string; message: string } }
  | { id: number; kind: 'progress'; progress: unknown };

export class CancelledError extends Error {
  constructor() {
    super('Cancelled');
    this.name = 'AbortError';
  }
}

export interface HandlerContext<P> {
  progress: (p: P) => void;
  checkpoint: () => Promise<void>;
  readonly cancelled: boolean;
}

const TRANSFER = Symbol('transfer');
export interface WithTransfer<T> {
  value: T;
  [TRANSFER]: Transferable[];
}
export function withTransfer<T>(value: T, transfer: Transferable[]): WithTransfer<T> {
  return { value, [TRANSFER]: transfer };
}

type Handlers<A extends ApiShape> = {
  [K in keyof A]: (
    params: A[K]['params'],
    ctx: HandlerContext<A[K]['progress']>,
  ) => Promise<A[K]['result'] | WithTransfer<A[K]['result']>> | A[K]['result'] | WithTransfer<A[K]['result']>;
};

interface WorkerScope {
  onmessage: ((e: MessageEvent<RequestMsg>) => void) | null;
  postMessage(msg: ResponseMsg, transfer?: Transferable[]): void;
}

export function exposeWorker<A extends ApiShape>(
  handlers: Handlers<A>,
  scope: WorkerScope = self as unknown as WorkerScope,
) {
  const cancelled = new Set<number>();
  scope.onmessage = async (e) => {
    const msg = e.data;
    if ('cancel' in msg) {
      cancelled.add(msg.id);
      return;
    }
    const handler = handlers[msg.method as keyof A];
    if (!handler) {
      scope.postMessage({
        id: msg.id,
        kind: 'error',
        error: { name: 'Error', message: `unknown method ${msg.method}` },
      });
      return;
    }
    const ctx: HandlerContext<unknown> = {
      progress: (p) => scope.postMessage({ id: msg.id, kind: 'progress', progress: p }),
      checkpoint: async () => {
        await new Promise((r) => setTimeout(r, 0));
        if (cancelled.has(msg.id)) throw new CancelledError();
      },
      get cancelled() {
        return cancelled.has(msg.id);
      },
    };
    try {
      const out = await handler(msg.params as never, ctx as never);
      if (out && typeof out === 'object' && TRANSFER in out) {
        const t = out as WithTransfer<unknown>;
        scope.postMessage({ id: msg.id, kind: 'result', result: t.value }, t[TRANSFER]);
      } else {
        scope.postMessage({ id: msg.id, kind: 'result', result: out });
      }
    } catch (err) {
      const e2 = err instanceof Error ? err : new Error(String(err));
      scope.postMessage({ id: msg.id, kind: 'error', error: { name: e2.name, message: e2.message } });
    } finally {
      cancelled.delete(msg.id);
    }
  };
}

export interface CallOptions<P> {
  signal?: AbortSignal;
  onProgress?: (p: P) => void;
}

export class WorkerClient<A extends ApiShape> {
  private worker: Worker | null = null;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void; onProgress?: (p: unknown) => void }
  >();

  constructor(private readonly factory: () => Worker) {}

  private ensure(): Worker {
    if (this.worker) return this.worker;
    const w = this.factory();
    w.onmessage = (e: MessageEvent<ResponseMsg>) => {
      const msg = e.data;
      const p = this.pending.get(msg.id);
      if (!p) return;
      if (msg.kind === 'progress') return p.onProgress?.(msg.progress);
      this.pending.delete(msg.id);
      if (msg.kind === 'result') p.resolve(msg.result);
      else {
        const err = new Error(msg.error.message);
        err.name = msg.error.name;
        p.reject(err);
      }
    };
    w.onerror = (e) => {
      const err = new Error(`Worker failed: ${e.message || 'script error'}`);
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
      this.worker?.terminate();
      this.worker = null;
    };
    this.worker = w;
    return w;
  }

  call<K extends keyof A & string>(
    method: K,
    params: A[K]['params'],
    opts: CallOptions<A[K]['progress']> = {},
    transfer: Transferable[] = [],
  ): Promise<A[K]['result']> {
    const w = this.ensure();
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      if (opts.signal?.aborted) return reject(new CancelledError());
      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
        onProgress: opts.onProgress as ((p: unknown) => void) | undefined,
      });
      opts.signal?.addEventListener(
        'abort',
        () => {
          w.postMessage({ id, cancel: true } satisfies RequestMsg);
        },
        { once: true },
      );
      w.postMessage({ id, method, params } satisfies RequestMsg, transfer);
    });
  }

  terminate(): void {
    this.worker?.terminate();
    this.worker = null;
    for (const p of this.pending.values()) p.reject(new CancelledError());
    this.pending.clear();
  }
}

export function isAbortError(e: unknown): boolean {
  return e instanceof Error && e.name === 'AbortError';
}
