# Artifact generation

Offline scripts that produce every bundled artifact from pinned model revisions. Outputs are committed, so you
only need these to regenerate or extend them.

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r scripts/generate-demo-artifacts/requirements.txt
```

| Script                   | Output                                                                | Notes                                                                                                                                                                                              |
| ------------------------ | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `export_pythia.py`       | `public/models/pythia-14m/{weights.bin,tokenizer.json,manifest.json}` | Bit-exact float16 copy; refuses to write the manifest unless an independent NumPy float64 forward pass matches exact-float64 transformers (< 1e-8 logits)                                          |
| `export_tokenizers.py`   | `public/tokenizers/`                                                  | Copies the GPT-2 tokenizer; manifest covers both tokenizers                                                                                                                                        |
| `generate_gpt2_demos.py` | `public/demos/`                                                       | GPT-2 small, eager attention, float32, CPU, eval mode; attention, predictions, logit lens, 144-head zero-ablation; checks row sums and causality; checks that a no-op hook leaves logits unchanged |
| `generate_fixtures.py`   | `tests/fixtures/`                                                     | References for the TypeScript tests: tokenizer ids, exact-float64 Pythia outputs (incl. hooked interventions), NumPy toy attention                                                                 |

`_common.py` pins model revisions and provides `load_pythia_exact_float64()`, which replaces transformers'
float32 softmax and float32 rotary tables with float64 versions so the model can serve as a tight reference.

Run order: `export_pythia.py` → `export_tokenizers.py` → `generate_gpt2_demos.py` → `generate_fixtures.py`, then
`npm run verify:artifacts && npm test`. Set `HF_HUB_OFFLINE=1` to reuse a populated Hugging Face cache.
