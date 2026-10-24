import { WorkerClient } from '@/workers/rpc';
import type { ModelApi } from '@/workers/modelApi';
import type { DataApi } from '@/workers/dataApi';

let model: WorkerClient<ModelApi> | null = null;
let data: WorkerClient<DataApi> | null = null;

/** Worker that owns the Pythia-14m weights and runs forward passes. */
export function modelWorker(): WorkerClient<ModelApi> {
  model ??= new WorkerClient<ModelApi>(
    () =>
      new Worker(new URL('../workers/model.worker.ts', import.meta.url), { type: 'module', name: 'model' }),
  );
  return model;
}

/** Worker for artifact parsing/verification, metrics and tokenizers. */
export function dataWorker(): WorkerClient<DataApi> {
  data ??= new WorkerClient<DataApi>(
    () => new Worker(new URL('../workers/data.worker.ts', import.meta.url), { type: 'module', name: 'data' }),
  );
  return data;
}

/** Drop the model worker and its weights from memory. */
export function releaseModelWorker(): void {
  model?.terminate();
  model = null;
}
