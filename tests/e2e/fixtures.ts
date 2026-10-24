import { test as base, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

/** Fails the test on uncaught page errors or console errors. */
export const test = base.extend<{ consoleErrors: string[] }>({
  consoleErrors: async ({ page }, provide) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
    page.on('console', (m) => {
      if (m.type() === 'error') errors.push(`console: ${m.text()}`);
    });
    await provide(errors);
    expect(errors, 'no console or page errors').toEqual([]);
  },
});

export { expect };

export async function expectNoA11yViolations(page: Page, exclude: string[] = []) {
  let builder = new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']);
  for (const e of exclude) builder = builder.exclude(e);
  const results = await builder.analyze();
  const summary = results.violations.map(
    (v) => `${v.id}: ${v.help} (${v.nodes.length}) ${v.nodes[0]?.target.join(' ')}`,
  );
  expect(summary, 'axe violations').toEqual([]);
}

export async function openDemo(page: Page, id = 'induction', extra = '') {
  await page.goto(`#/lab?demo=${id}${extra}`);
  await expect(page.getByRole('group', { name: /Attention probabilities for GPT-2 small/ })).toBeVisible();
}
