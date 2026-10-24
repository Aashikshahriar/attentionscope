import { test, expect, openDemo } from './fixtures';

test('Q/K/V lab recomputes and reports fully masked rows', async ({ page, consoleErrors }) => {
  await page.goto('#/lab?demo=ioi&tab=qkv');
  await expect(page.getByText(/not a pretrained language model/)).toBeVisible();
  const sums = page.getByRole('figure').filter({ hasText: 'row sums' });
  await expect(sums.getByText('1.000000')).toHaveCount(4);
  const attn = page.getByRole('figure').filter({ hasText: 'A = softmax(rows)' });
  const before = await attn.innerText();
  await page.getByLabel('Temperature τ = 1.00').fill('3');
  await expect.poll(async () => attn.innerText()).not.toBe(before);
  await page.getByRole('button', { name: 'The', exact: true }).click(); // pad the first key
  await expect(page.getByText('Fully masked rows')).toBeVisible();
  expect(consoleErrors).toEqual([]);
});

test('mobile layout has no horizontal page overflow', async ({ page, consoleErrors }) => {
  await openDemo(page, 'induction');
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  await page.goto('./');
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  const overflowLanding = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflowLanding).toBeLessThanOrEqual(1);
  expect(consoleErrors).toEqual([]);
});

test('in-app navigation between lab URLs applies the new configuration', async ({ page, consoleErrors }) => {
  await openDemo(page, 'induction');
  await page.goto('#/lab?demo=code&l=2&h=3');
  await expect(page.getByRole('heading', { name: 'Layer 2 · Head 3' })).toBeVisible();
  await page.goto('#/lab?src=live');
  await expect(page.getByRole('button', { name: /Load model/ })).toBeVisible();
  expect(consoleErrors).toEqual([]);
});
