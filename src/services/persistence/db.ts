import { Dexie, type EntityTable } from 'dexie';
import type { ExperimentSession } from '@/types/contracts';

class AttentionScopeDb extends Dexie {
  experiments!: EntityTable<ExperimentSession, 'id'>;

  constructor() {
    super('attentionscope');
    this.version(1).stores({ experiments: 'id, createdAt, name' });
  }
}

let db: AttentionScopeDb | null = null;
function open(): AttentionScopeDb {
  if (typeof indexedDB === 'undefined') throw new Error('IndexedDB is not available in this browser context');
  db ??= new AttentionScopeDb();
  return db;
}

export async function saveExperiment(s: ExperimentSession): Promise<void> {
  await open().experiments.put(s);
}

export async function listExperiments(): Promise<ExperimentSession[]> {
  return open().experiments.orderBy('createdAt').reverse().toArray();
}

export async function deleteExperiment(id: string): Promise<void> {
  await open().experiments.delete(id);
}

export async function clearExperiments(): Promise<void> {
  await open().experiments.clear();
}
