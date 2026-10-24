# Limitations, browser support and reproducibility

## Limitations

- Browser inference is limited to Pythia-14m, the one model whose forward pass AttentionScope implements and
  validates. Arbitrary Hugging Face models are not supported.
- Live input: at most 64 tokens. Demos are fixed prompts.
- Pythia-14m is tiny; its predictions are often poor. Use it to study mechanisms, not capabilities.
- Pre-softmax scores are available for live runs only (not stored in precomputed artifacts).
- Residual-stream norms and logit lens are provided; per-neuron MLP inspection, activation patching between
  prompts, causal tracing, cross-attention (encoder–decoder) and model comparison are **not implemented**.
- Single sequences (batch 1, no padding). Padding masks are demonstrated in the Q/K/V lab only.
- Multilingual support covers tokenization; neither model handles Bengali, Hindi or Korean well.
- Precomputed GPT-2 ablations are zero-ablations on single prompts; they are not executed in the browser.

## Browser support

Current Chrome, Edge, Firefox and Safari (2023+) with ES2022, module Web Workers, Web Crypto (`crypto.subtle`,
available on HTTPS and localhost), IndexedDB and Cache Storage. The live model needs about 150 MB of free memory.
E2E tests run on Chromium (desktop and Pixel 7 emulation).

## Offline use

After a visit, browsers may serve the app from their HTTP cache. The model is stored for offline use only if you
enable it. There is no service worker.

## Reproducibility

- Model and tokenizer revisions are pinned by commit hash; artifact checksums are verified at load time.
- Every export records model repo and revision, tokenizer and BOS setting, input text and token ids, source
  (precomputed artifact + SHA-256 or live weights SHA-256), compute/storage precision, seed (where applicable),
  intervention spec and app version/commit.
- Live runs are deterministic: identical inputs give bit-identical outputs in the same browser.
- Regenerate artifacts with the scripts in `scripts/generate-demo-artifacts/` (see `docs/PROVENANCE.md`).
