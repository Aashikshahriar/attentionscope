import { create } from 'zustand';

export interface LogEntry {
  id: number;
  time: string;
  level: 'info' | 'warn' | 'error';
  scope: 'demo' | 'model' | 'run' | 'intervention' | 'export' | 'session' | 'tokenizer' | 'metrics';
  message: string;
  durationMs?: number;
}

interface LogState {
  entries: LogEntry[];
  add: (e: Omit<LogEntry, 'id' | 'time'>) => void;
  clear: () => void;
}

let next = 1;

/** Computation log: every entry corresponds to a real event (fetch, verification, forward pass…). */
export const useLog = create<LogState>((set) => ({
  entries: [],
  add: (e) =>
    set((s) => ({
      entries: [...s.entries.slice(-199), { ...e, id: next++, time: new Date().toISOString() }],
    })),
  clear: () => set({ entries: [] }),
}));

export const log = (e: Omit<LogEntry, 'id' | 'time'>) => useLog.getState().add(e);
