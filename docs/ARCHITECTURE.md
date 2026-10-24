# Architecture, decisions and delivery status

## Architecture decision records

### ADR-1 · Two attention pathways plus a toy module

**Decision.** Ship (a) precomputed GPT-2 small artifacts generated offline from real forward passes and (b) one
small model executed in the browser by our own implementation. The Q/K/V lab is a separately labelled toy.
**Why.** GitHub Pages is static; GPT-2 small (124 M params, ~500 MB) is too large for in-browser JS inference, but
its heads are the best-studied in the literature. A tiny model gives genuine live computation and interventions.
**Consequence.** Every view carries a `precomputed` / `live` / `toy` label, and exports record which one.

### ADR-2 · Our own GPT-NeoX forward pass instead of Transformers.js / ONNX Runtime Web

**Decision.** Implement Pythia-14m's forward pass in TypeScript (`src/core/model-adapters/gptNeox.ts`).
**Why.** Generic pipelines return logits/hidden states; fused ONNX attention kernels do not expose attention
probabilities and cannot be edited between softmax and `AV`. Pythia-14m is small enough (6 layers, d=128) that a
plain JS implementation runs a 64-token forward pass in ~0.1 s (next-token logits only) or ~0.5 s with
logits at every position, pre-softmax scores and logit lens (Node 24, desktop CPU).
**Validation.** An independent NumPy float64 implementation and transformers in exact float64 (custom attention
function + float64 rotary tables) agree to 4e-12; the TypeScript implementation matches them to 1e-8 (logits).

### ADR-3 · Model choice: Pythia-14m (Apache-2.0)

Candidates considered: GPT-2 small (too large), distilgpt2 (82 M, too large), TinyStories-1M/3M (no license
declared on the model repos — rejected), Pythia-14m (Apache-2.0, published in float16, rotary + parallel residual,
used in interpretability research). Weights are shipped bit-exactly (28.1 MB float16).

### ADR-4 · float64 activations

Pythia-14m produces pre-softmax scores up to ~180. Stock transformers eager attention computes softmax and the
rotary tables in float32, which shifts final logits by ~6e-3 from exact arithmetic. JavaScript arithmetic is
float64, so the browser keeps activations in `Float64Array` and stores attention as `Float32Array`.

### ADR-5 · Byte-level BPE in TypeScript

Both tokenizers are byte-level BPE `tokenizer.json` files. A ~300-line implementation reproduces the Rust
pipeline (raw added tokens → NFC → normalized added tokens → GPT-2 regex → BPE) and is tested token-for-token on
21 inputs × 2 tokenizers, including whitespace runs, emoji ZWJ sequences, Bengali, Hindi, Korean, Arabic and
decomposed accents. Offsets are derived from the byte stream, never guessed.

### ADR-6 · Static hosting

Hash routing (`createHashRouter`) so refreshes never 404; `public/404.html` forwards non-hash paths. Vite
`base: '/attentionscope/'`; workers are created with `new URL(..., import.meta.url)` and receive absolute asset
URLs. No secrets, no external services, no third-party requests at runtime.

### ADR-7 · Workers and cancellation

A typed RPC (`src/workers/rpc.ts`) supports progress messages, transferable results and cooperative
cancellation: long computations `await ctx.checkpoint()` between layers/heads, which yields to the worker event
loop so a `cancel` message can arrive. Downloads are cancelled by terminating the worker.

### ADR-8 · Rendering

Heatmaps use canvas (up to 64×64 cells per head, 144 mini-maps in the compare grid). Arc and dot plots use SVG.
Every visual has a keyboard path and a table alternative.

## Feature prioritization and status

| Priority | Feature                                                                                | Status                                                                                   |
| -------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| P0       | Landing page, responsive workspace, theme and accessibility controls                   | Done                                                                                     |
| P0       | Curated, licensed, versioned, checksummed demo artifacts; precomputed status disclosed | Done (6 GPT-2 demos)                                                                     |
| P0       | Input text + tokenization for verified browser tokenizers                              | Done (GPT-2, GPT-NeoX)                                                                   |
| P0       | Layer/head selection, heatmap with labelled axes                                       | Done                                                                                     |
| P0       | Arc view and token-to-token highlighting, synchronized selection                       | Done                                                                                     |
| P0       | Causal/padding masks, normalization, tensor dimensions                                 | Done (padding is demonstrated in the Q/K/V lab; real runs are single unpadded sequences) |
| P0       | Mathematical Q/K/V explorer with transparent weights                                   | Done                                                                                     |
| P0       | Artifact manifest with checksum and provenance                                         | Done                                                                                     |
| P0       | PNG / CSV / JSON export with provenance; table alternative                             | Done                                                                                     |
| P0       | Worker computation, cancellation, loading/error states                                 | Done                                                                                     |
| P1       | Real browser inference with attention tensors                                          | Done (Pythia-14m)                                                                        |
| P1       | Layer/head comparison, entropy, distance, token statistics                             | Done                                                                                     |
| P1       | Head ablation and attention intervention on a modifiable forward pass                  | Done (4 intervention kinds + sweep)                                                      |
| P1       | Deterministic before/after probability comparison                                      | Done                                                                                     |
| P1       | Experiment history (IndexedDB), import/export, share links                             | Done                                                                                     |
| P1       | Guided math walkthrough and caveats                                                    | Done (Q/K/V worked example, docs)                                                        |
| P2       | Attention rollout with explicit assumptions                                            | Done                                                                                     |
| P2       | Residual stream inspection                                                             | Partial: residual norms and logit lens; no per-neuron MLP inspection                     |
| P2       | Multilingual tokenization comparison                                                   | Done (tokenization only)                                                                 |
| P2       | Cross-attention (encoder–decoder)                                                      | Not implemented                                                                          |
| P2       | Activation patching between prompts, causal tracing, model comparison                  | Not implemented                                                                          |

## Risk register

| Risk                                                     | Impact   | Mitigation                                                                                                         | Residual |
| -------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------ | -------- |
| Fabricated or mislabelled attention                      | Critical | All tensors come from scripted forward passes; checksums verified in browser + CI; labels on every view and export | Low      |
| Browser implementation diverges from the reference model | High     | Scientific tests vs exact-float64 transformers incl. interventions; export-time NumPy check                        | Low      |
| Tokenizer mismatch                                       | High     | Token-for-token tests vs Rust tokenizers; demo token ids re-tokenized in tests                                     | Low      |
| Large download on mobile / low memory                    | Medium   | Model loads only on explicit click with size shown; progress, cancel and release; 64-token limit                   | Medium   |
| Misinterpretation of attention as explanation            | Medium   | Caveats in inspector, docs, intervention results; no "reasoning path" claims                                       | Medium   |
| GitHub Pages sub-path / refresh bugs                     | Medium   | Hash routing; E2E runs against the built site under `/attentionscope/`; post-deploy smoke test                     | Low      |
| Dependency drift                                         | Low      | `package-lock.json` + `npm ci`; Python versions recorded in each artifact                                          | Low      |
| License incompatibility                                  | Medium   | Only MIT / Apache-2.0 model assets; notices in THIRD_PARTY_NOTICES.md                                              | Low      |

## Dependency verification plan

- **Model/tokenizer revisions** pinned by commit SHA in `scripts/generate-demo-artifacts/_common.py`.
- **Generation environment** (Python, torch, transformers, numpy, platform) is written into every artifact.
- **npm** dependencies locked by `package-lock.json`; CI uses `npm ci`.
- **Runtime artifacts** verified by SHA-256 before use (`data.worker`, `model.worker`) and in CI
  (`scripts/verify-artifacts.mjs`).

## Phase log

| Phase            | Outcome                                                                                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 Feasibility    | GPT-2 (MIT) and Pythia-14m (Apache-2.0) verified; float32-softmax discrepancy in transformers found and characterized; exact-float64 reference built |
| 1 Foundation     | Vite/React/TS strict, design tokens, router, shell, demo manifest                                                                                    |
| 2 Visualization  | Heatmap, arcs, tokens, inspector, exports; axes verified by tests (`headOffset`, demo suggested-view argmax)                                         |
| 3 Education      | Q/K/V lab validated against NumPy fixtures                                                                                                           |
| 4 Live model     | GPT-NeoX in a worker, validated against PyTorch                                                                                                      |
| 5 Research tools | Metrics, rollout, logit lens, interventions, sweeps, IndexedDB sessions                                                                              |
| 6 Hardening      | axe checks, mobile layout, CI/CD, docs                                                                                                               |
