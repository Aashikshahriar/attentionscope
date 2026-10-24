import { test, expect, expectNoA11yViolations } from './fixtures';

test.describe.configure({ mode: 'serial' });

test('live Pythia-14m: verified load, forward pass, intervention and sweep', async ({
  page,
  consoleErrors,
}) => {
  test.setTimeout(180_000);
  await page.goto('#/lab?src=live');
  await expect(page.getByRole('button', { name: /Load model \(28\.1 MB\)/ })).toBeVisible();
  await page.getByRole('button', { name: /Load model/ }).click();
  await expect(page.getByText('Pythia-14m ready')).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText(/Weights SHA-256 verified/)).toBeVisible();

  await page.getByLabel('Input text').fill('The cat sat on the mat. The cat sat on the');
  await expect(page.getByText(/tokens of 64 max/)).toBeVisible();
  await page.getByRole('button', { name: 'Run forward pass' }).click();
  const heatmap = page.getByRole('group', { name: /Attention probabilities for Pythia-14m/ });
  await expect(heatmap).toBeVisible();
  await expect(page.getByText('live', { exact: true }).first()).toBeVisible();

  // pre-softmax scores are available for live runs
  await page.getByRole('radio', { name: 'Pre-softmax' }).click();
  await expect(page.getByRole('group', { name: /Pre-softmax scores for Pythia-14m/ })).toBeVisible();

  // intervention re-runs the model in the worker
  await page.getByRole('tab', { name: 'Interventions' }).click();
  await expect(page.getByText('Executed in your browser')).toBeVisible();
  await page.getByRole('button', { name: /Run on L\d H\d/ }).click();
  await expect(page.getByText(/KL\(base ‖ intervened\)/)).toBeVisible();
  await expect(page.getByText('No replacement matrix (head-output ablation).')).toBeVisible();

  // attention replacement reports its validation
  await page.getByRole('combobox', { name: 'Intervention' }).click();
  await page.getByRole('option', { name: 'Replace attention: uniform' }).click();
  await page.getByRole('button', { name: /Run on L\d H\d/ }).click();
  await expect(page.getByText('replacement valid')).toBeVisible();

  await page.getByRole('button', { name: /Sweep all 24 heads/ }).click();
  await expect(page.getByText(/24 forward passes in/)).toBeVisible({ timeout: 60_000 });
  await expectNoA11yViolations(page);

  // log records real events
  await page.getByRole('tab', { name: /^Log/ }).click();
  await expect(page.getByText(/Forward pass on Pythia-14m: T=\d+/)).toBeVisible();
  expect(consoleErrors).toEqual([]);
});
