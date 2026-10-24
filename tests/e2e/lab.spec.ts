import { readFileSync } from 'node:fs';
import { test, expect, expectNoA11yViolations, openDemo } from './fixtures';

test('precomputed demo loads, verifies and is keyboard navigable', async ({ page, consoleErrors }) => {
  await openDemo(page, 'induction', '&l=5&h=5&q=18');
  await expect(page.getByText('precomputed', { exact: true }).first()).toBeVisible();
  // Inspector reflects the authentic induction pattern: " lazy" (2nd copy) -> " dog" (1st copy).
  const inspector = page.getByRole('complementary', { name: 'Inspector' });
  await expect(inspector.getByText('position 18 · id 16931')).toBeVisible();
  await expect(inspector.getByText('0.9538')).toBeVisible();

  const heatmap = page.getByRole('group', { name: /layer 5, head 5/ });
  await heatmap.focus();
  await page.keyboard.press('ArrowLeft');
  await expect(page.getByText(/Query 18 " lazy", key 17 " the": attention probability/)).toBeAttached();
  await page.keyboard.press('ArrowUp');
  await expect(inspector.getByText('position 17 · id 262')).toBeVisible();

  // layer/head shortcuts
  await page.locator('body').click({ position: { x: 5, y: 600 } });
  await page.keyboard.press(']');
  await expect(page.getByRole('heading', { name: 'Layer 6 · Head 5' })).toBeVisible();

  // accessible table alternative
  await page.getByRole('switch', { name: 'Table view' }).click();
  await expect(page.getByRole('table', { name: /Attention probabilities/ })).toBeVisible();
  await expectNoA11yViolations(page);
  expect(consoleErrors).toEqual([]);
});

test('every tab renders for a precomputed demo', async ({ page, consoleErrors }) => {
  await openDemo(page, 'ioi');
  for (const [tab, marker] of [
    ['Arcs', /Show all queries/],
    ['Tokens', /Why tokens are not words/],
    ['Compare', /Attention rollout/],
    ['Interventions', /Precomputed offline/],
    ['Q/K/V Lab', /not a pretrained language model/],
  ] as const) {
    await page.getByRole('tab', { name: tab }).click();
    await expect(page.getByText(marker).first()).toBeVisible();
  }
  await page.getByRole('tab', { name: 'Compare' }).click();
  await expect(page.getByRole('grid', { name: /144 heads/ })).toBeVisible();
  await expectNoA11yViolations(page);
  expect(consoleErrors).toEqual([]);
});

test('CSV and JSON exports carry provenance', async ({ page, consoleErrors }) => {
  await openDemo(page, 'factual', '&l=9&h=8');
  const [csv] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'CSV' }).click(),
  ]);
  const text = readFileSync((await csv.path())!, 'utf8');
  expect(text).toContain('layer 9, head 8');
  expect(text).toContain('openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e');
  expect(text).toMatch(/attention sha256 [0-9a-f]{64}/);
  expect(text).toContain('# token ids: 50256 464');
  const [json] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'JSON (head)' }).click(),
  ]);
  const obj = JSON.parse(readFileSync((await json.path())!, 'utf8'));
  expect(obj.provenance.precomputed).toBe(true);
  expect(obj.provenance.input.text).toBe('The Eiffel Tower is located in the city of');
  expect(obj.attention).toHaveLength(12);
  obj.attention.forEach((row: number[]) => expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5));
  const [png] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'PNG' }).click(),
  ]);
  expect(png.suggestedFilename()).toMatch(/\.png$/);
  expect(consoleErrors).toEqual([]);
});

test('experiments save to IndexedDB and invalid imports are rejected', async ({ page, consoleErrors }) => {
  await openDemo(page, 'agreement');
  await page.getByLabel('Experiment name').fill('Agreement check');
  await page.getByRole('button', { name: 'Save', exact: true }).click();
  await expect(page.getByText('Saved “Agreement check”.')).toBeVisible();
  await page.getByRole('tab', { name: 'Experiments' }).click();
  await expect(page.getByText('Agreement check')).toBeVisible();
  await page.locator('#import-session').setInputFiles({
    name: 'bad.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify({ schemaVersion: 1, id: 'x', name: 'x', createdAt: 'nope' })),
  });
  await expect(page.getByText(/Invalid experiment file: invalid createdAt/)).toBeVisible();
  expect(consoleErrors).toEqual([]);
});

test('light theme and high contrast pass accessibility checks', async ({ page, consoleErrors }) => {
  await openDemo(page, 'code');
  await page.getByRole('button', { name: 'Display and accessibility settings' }).click();
  await page.getByRole('radio', { name: 'Light' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.getByRole('switch', { name: 'High contrast' }).click();
  await page.keyboard.press('Escape');
  await expect(page.locator('html')).toHaveAttribute('data-contrast', 'high');
  await expectNoA11yViolations(page);
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  expect(consoleErrors).toEqual([]);
});

test('tokenizers and docs pages pass accessibility checks', async ({ page, consoleErrors }) => {
  await page.goto('#/tokenizers');
  await page.getByRole('button', { name: 'Korean' }).click();
  await expect(page.getByText('tokens', { exact: true }).first()).toBeVisible();
  await expect(page.getByText(/tokenized in .* ms \(Web Worker\)/).first()).toBeVisible();
  await expectNoA11yViolations(page);
  await page.goto('#/docs/provenance');
  await expect(page.getByText(/SHA-256/).first()).toBeVisible();
  await expectNoA11yViolations(page);
  expect(consoleErrors).toEqual([]);
});
