# Deploying to GitHub Pages

The site is built for `https://<user>.github.io/attentionscope/` (Vite `base: '/attentionscope/'`).

## One-time setup

1. Create a public repository named **`attentionscope`** under your account and push this project to `main`:
   ```bash
   git remote add origin https://github.com/<user>/attentionscope.git
   git push -u origin main
   ```
2. In the repository: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. The push triggers **CI & Deploy to GitHub Pages** (`.github/workflows/deploy.yml`). When it finishes, the
   `deploy` job shows the URL and the `smoke-test` job runs the landing and lab E2E tests against the live site.

No secrets or environment variables are required.

## What the workflow does

`npm ci` → verify artifact checksums → typecheck → lint → format check → Vitest → build → Playwright + axe
against the production build under `/attentionscope/` → `actions/configure-pages` → `actions/upload-pages-artifact`
→ `actions/deploy-pages` → smoke test on the deployed URL. Pull requests run everything except deployment.

## Renaming the repository

If the repository is not called `attentionscope`, change `base` in `vite.config.ts`, the `base` constant in
`public/404.html`, URLs in `index.html` (`og:url`, `og:image`, favicon) and `src/services/paths.ts`, and the
`/attentionscope/` paths in `playwright.config.ts` and the E2E tests.

## Custom domain (optional)

1. Add the domain under **Settings → Pages → Custom domain** and create the DNS records GitHub shows
   (`CNAME` to `<user>.github.io` for a subdomain).
2. Add `public/CNAME` containing the domain so it survives deployments.
3. Serving from a domain root requires `base: '/'` in `vite.config.ts`, `const base = '/'` in `public/404.html`,
   and updated absolute URLs in `index.html`, `src/services/paths.ts` and the Playwright config.
4. Enable **Enforce HTTPS** (Web Crypto checksum verification requires a secure context).

## Verifying a deployment manually

```bash
E2E_BASE_URL=https://<user>.github.io/attentionscope/ npx playwright test --project=desktop
```

Check: landing hero renders real attention; `#/lab?demo=induction` loads; refreshing `#/docs/privacy` works;
`/attentionscope/lab` (no hash) redirects; **Load model** downloads 28.1 MB and reports "SHA-256 verified".

## Hosting limits

The deployed site is ~34 MB plus ~4 MB of source maps (28.1 MB is the model weights). GitHub Pages allows sites up to 1 GB and files up to
100 MB; no Git LFS is needed.
