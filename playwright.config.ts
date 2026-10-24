import { defineConfig, devices } from '@playwright/test';

const PORT = 4318;

// E2E runs against the production build served under the GitHub Pages
// sub-path, so base-path, worker and asset URL bugs surface before deploy.
export default defineConfig({
  testDir: './tests/e2e',
  timeout: 90_000,
  expect: { timeout: 15_000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  use: {
    baseURL: process.env.E2E_BASE_URL ?? `http://localhost:${PORT}/attentionscope/`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'] } },
    {
      name: 'mobile',
      use: { ...devices['Pixel 7'] },
      testMatch: /mobile|landing/,
      grepInvert: /in-app navigation/,
    },
  ],
  webServer: process.env.E2E_BASE_URL
    ? undefined
    : {
        command: 'npm run preview',
        url: `http://localhost:${PORT}/attentionscope/`,
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
      },
});
