export type Matrix = number[][];

export function shapeOf(m: Matrix): [number, number] {
  return [m.length, m[0]?.length ?? 0];
}

export function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
}

export function matmul(a: Matrix, b: Matrix): Matrix {
  const [n, k] = shapeOf(a);
  const [k2, m] = shapeOf(b);
  if (k !== k2) throw new Error(`matmul shape mismatch: (${n}×${k}) · (${k2}×${m})`);
  const out = zeros(n, m);
  for (let i = 0; i < n; i++) {
    const ai = a[i]!;
    const oi = out[i]!;
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += ai[p]! * b[p]![j]!;
      oi[j] = s;
    }
  }
  return out;
}

export function transpose(a: Matrix): Matrix {
  const [n, m] = shapeOf(a);
  const out = zeros(m, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[j]![i] = a[i]![j]!;
  return out;
}

export function hconcat(blocks: Matrix[]): Matrix {
  if (blocks.length === 0) return [];
  const rows = blocks[0]!.length;
  return Array.from({ length: rows }, (_, i) => blocks.flatMap((b) => b[i]!));
}

export function mapMatrix(a: Matrix, f: (v: number, i: number, j: number) => number): Matrix {
  return a.map((row, i) => row.map((v, j) => f(v, i, j)));
}

export function cloneMatrix(a: Matrix): Matrix {
  return a.map((r) => r.slice());
}
