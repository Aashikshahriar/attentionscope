import { describe, expect, it, beforeEach } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import type { AttentionRun } from '@/types/contracts';
import type { DemoArtifact } from '@/types/artifacts';
import { useWorkspace } from '@/stores/workspace';
import { Inspector } from '@/features/workspace/Inspector';
import { QkvLab } from '@/features/qkv/QkvLab';
import { TooltipProvider } from '@/components/ui/controls';
import { headCsv, headJson, runJson } from '@/services/export';
import { makeSession, parseSession, shareHash } from '@/services/session';
import { deleteExperiment, listExperiments, saveExperiment } from '@/services/persistence/db';
import { APP_INFO } from '@/services/provenance';
import { readArrayBuffer, readJson } from '../helpers';

/** Build an AttentionRun from a bundled artifact exactly as the data worker + runs service do. */
function demoRun(id: string): AttentionRun {
  const a = readJson<DemoArtifact>(`public/demos/${id}.json`);
  const data = new Float32Array(readArrayBuffer(`public/demos/${a.attention.file}`));
  return {
    id: `demo-${a.id}`,
    title: a.title,
    source: {
      kind: 'precomputed',
      demoId: a.id,
      artifactFile: `demos/${id}.json`,
      artifactSha256: 'x',
      attentionSha256: a.attention.sha256,
    },
    model: { ...a.model },
    tokenizer: a.tokenizer,
    input: { text: a.input.text, tokens: a.input.tokens, normalization: 'none' },
    attention: {
      shape: a.attention.shape,
      layout: ['layer', 'head', 'query', 'key'],
      kind: 'post-softmax',
      mask: 'causal',
      data,
    },
    predictions: a.predictions,
    ablation: a.ablation,
    suggestedView: a.suggestedView,
    provenance: {
      app: APP_INFO,
      precomputed: true,
      generator: a.generator.script,
      computePrecision: 'float32',
      storedPrecision: 'float32',
      deterministic: true,
      seed: 0,
      computedAt: a.generator.generatedAt,
    },
  };
}

const wrap = (ui: React.ReactNode) =>
  render(
    <MemoryRouter>
      <TooltipProvider>{ui}</TooltipProvider>
    </MemoryRouter>,
  );

describe('demo → inspect → export', () => {
  beforeEach(() => {
    const run = demoRun('induction');
    useWorkspace.setState({ run, runStatus: 'ready', layer: 5, head: 5, queryIndex: 18, keyIndex: null });
  });

  it('inspector shows the selected token, its attention row and tensor shape', () => {
    wrap(<Inspector run={useWorkspace.getState().run!} />);
    expect(screen.getByText(/position 18 · id 16931/)).toBeInTheDocument();
    const row = screen.getByRole('region', { name: /Attention row/ });
    expect(within(row).getByText('0.9538')).toBeInTheDocument();
    expect(within(row).getByText('1.000000')).toBeInTheDocument(); // row sum
    expect(screen.getByText('[1, 12, 12, 19, 19]')).toBeInTheDocument();
  });

  it('selecting a key in the inspector updates the shared store', () => {
    wrap(<Inspector run={useWorkspace.getState().run!} />);
    fireEvent.click(screen.getAllByRole('button', { name: /dog/ })[0]!);
    expect(useWorkspace.getState().keyIndex).toBe(9);
    expect(screen.getByText(/Selected cell q18 → k9/)).toBeInTheDocument();
  });

  it('exports include provenance and the exact tensor values', () => {
    const run = useWorkspace.getState().run!;
    const csv = headCsv(run, 5, 5);
    expect(csv).toContain('openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e');
    expect(
      csv
        .trim()
        .split('\n')
        .filter((l) => !l.startsWith('#')),
    ).toHaveLength(1 + 19);
    const head = JSON.parse(headJson(run, 5, 5));
    expect(head.attention[18][9]).toBeCloseTo(0.9538, 4);
    expect(head.provenance.tokenizer.prependBos).toBe(true);
    const full = JSON.parse(runJson(run));
    expect(full.attention).toHaveLength(12);
    expect(full.attention[0]).toHaveLength(12);
    expect(full.precomputedAblation.method).toBe('zero-head-output');
  });
});

describe('sessions', () => {
  it('round-trips through JSON and IndexedDB, and builds share links', async () => {
    const run = demoRun('ioi');
    const s = makeSession(run, { layer: 9, head: 9, queryIndex: 14 }, { name: 'IOI', notes: 'n' });
    const parsed = parseSession(JSON.parse(JSON.stringify(s)));
    expect(parsed).toEqual(s);
    await saveExperiment(s);
    expect((await listExperiments()).map((x) => x.id)).toContain(s.id);
    await deleteExperiment(s.id);
    expect((await listExperiments()).map((x) => x.id)).not.toContain(s.id);
    expect(shareHash(run, { layer: 9, head: 9, queryIndex: 14 }, 'heatmap', false)).toBe(
      '#/lab?demo=ioi&l=9&h=9&q=14&tab=heatmap',
    );
  });

  it('rejects malformed experiment files', () => {
    expect(() => parseSession([])).toThrow(/JSON object/);
    expect(() => parseSession({ schemaVersion: 2 })).toThrow(/schemaVersion/);
    const run = demoRun('ioi');
    const s = makeSession(run, { layer: 0, head: 0, queryIndex: 0 }, { name: 'x' });
    expect(() => parseSession({ ...s, source: { kind: 'precomputed', demoId: '../../etc' } })).toThrow(
      /demoId/,
    );
    expect(() => parseSession({ ...s, input: { text: 'a', tokenIds: [1.5] } })).toThrow(/tokenIds/);
  });
});

describe('Q/K/V lab', () => {
  it('recomputes attention when the temperature changes and keeps rows normalized', () => {
    wrap(<QkvLab />);
    const attn = screen.getByText('A = softmax(rows)').closest('figure')!;
    const before = attn.textContent;
    fireEvent.change(screen.getByLabelText(/Temperature/), { target: { value: '3' } });
    expect(attn.textContent).not.toBe(before);
    const sums = screen.getByText('row sums').closest('figure')!;
    expect(within(sums).getAllByText('1.000000')).toHaveLength(4);
    expect(screen.getByText(/not a pretrained language model/)).toBeInTheDocument();
  });

  it('editing an input embedding changes Q', () => {
    wrap(<QkvLab />);
    const q = screen.getByText(/^Q = X W_Q/).closest('figure')!;
    const before = q.textContent;
    fireEvent.change(screen.getByLabelText('X (T×d = 4×4), token 0, column 0'), { target: { value: '5' } });
    expect(q.textContent).not.toBe(before);
  });
});
