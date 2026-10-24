import { test, expect, expectNoA11yViolations } from './fixtures';

test('landing page renders real attention and links into the lab', async ({ page, consoleErrors }) => {
  await page.goto('./');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Inside the mind');
  await expect(page.getByRole('img', { name: /GPT-2 small layer 5 head 5/ })).toBeVisible();
  await expect(page.getByRole('link', { name: /Induction: repeated sentence/ })).toBeVisible();
  await expectNoA11yViolations(page);
  await page.getByRole('link', { name: 'Launch Lab' }).click();
  await expect(page).toHaveURL(/#\/lab/);
  await expect(page.getByRole('tab', { name: 'Heatmap' })).toBeVisible();
  expect(consoleErrors).toEqual([]);
});

test('hash routes survive a full reload under the GitHub Pages sub-path', async ({ page, consoleErrors }) => {
  await page.goto('#/docs/privacy');
  await expect(page.getByRole('heading', { name: 'Privacy' })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Privacy' })).toBeVisible();
  expect(new URL(page.url()).pathname).toBe('/attentionscope/');
  expect(consoleErrors).toEqual([]);
});

test('unknown non-hash paths are redirected by 404.html', async ({ page, baseURL }) => {
  test.skip(
    !!process.env.E2E_BASE_URL === false && !baseURL?.includes('github.io'),
    'vite preview serves index.html for unknown paths',
  );
  await page.goto('lab');
  await expect(page).toHaveURL(/#\/lab/);
});
