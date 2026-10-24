# Contributing

Thanks for helping. AttentionScope is a scientific tool, so correctness comes first.

## Ground rules

- **Never fabricate data.** Attention values, logits and predictions must come from a real forward pass, either
  offline via `scripts/generate-demo-artifacts/` or live via an implementation validated against a reference.
- **Label provenance.** Anything precomputed must say so in the UI and in exports.
- **Expose interventions only where the forward pass is actually modified**, and test them against PyTorch hooks.
- Keep claims modest: attention is not explanation.

## Development

```bash
npm ci
npm run dev
npm run check            # artifacts, typecheck, lint, tests, build
npx playwright install chromium && npm run test:e2e
```

- TypeScript is strict; prefer pure functions in `src/core/` with unit tests.
- New math needs a test against an independent reference (NumPy or PyTorch fixture in `tests/fixtures/`, generated
  by `scripts/generate-demo-artifacts/generate_fixtures.py`).
- UI changes must keep the axe checks green and remain keyboard-operable.
- Run `npm run format` before committing.

## Adding a demo

1. Add an entry to `DEMOS` in `scripts/generate-demo-artifacts/generate_gpt2_demos.py` (single-token target and a
   focus query/key for the suggested head).
2. Run the script, then `npm run verify:artifacts && npm test`.
3. Commit the new `public/demos/*` files and the updated manifest.

## Adding a browser model

A model qualifies only if its license permits redistribution, its tokenizer is supported by
`src/core/tokenizer/bpe.ts` (or a new tested tokenizer), its architecture is implemented in
`src/core/model-adapters/`, and a scientific test matches it to a float64 PyTorch reference, including attention
probabilities and every exposed intervention.

## Pull requests

Describe what changed, how it was validated (commands and results), and any scientific caveats. CI must pass.
