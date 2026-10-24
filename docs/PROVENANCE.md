# Model and data provenance

| Asset                  | Source (pinned revision)                                             | License    | Files                                                                                   |
| ---------------------- | -------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------- |
| GPT-2 small            | `openai-community/gpt2` @ `607a30d783dfa663caf39e06633721c8d4cfcd7e` | MIT        | used offline only; outputs in `public/demos/`                                           |
| GPT-2 tokenizer        | same                                                                 | MIT        | `public/tokenizers/gpt2/tokenizer.json`                                                 |
| Pythia-14m             | `EleutherAI/pythia-14m` @ `cf967c0a9a04383db6f7b1108d86b2962634b4ac` | Apache-2.0 | `public/models/pythia-14m/weights.bin` (float16, bit-exact copy of `model.safetensors`) |
| GPT-NeoX-20B tokenizer | same                                                                 | Apache-2.0 | `public/models/pythia-14m/tokenizer.json`                                               |

Checksums live in `public/demos/manifest.json`, `public/models/pythia-14m/manifest.json` and
`public/tokenizers/manifest.json`, are verified by the app before use, and by `npm run verify:artifacts` in CI.

## Demo prompts

All demos prepend `<|endoftext|>` (GPT-2's BOS/EOS token) and were authored for this project.

| id          | text                                                                                 | measured target                              |
| ----------- | ------------------------------------------------------------------------------------ | -------------------------------------------- |
| induction   | The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy | " dog"                                       |
| ioi         | When Mary and John went to the store, John gave a drink to                           | " Mary"                                      |
| factual     | The Eiffel Tower is located in the city of                                           | " Paris" (GPT-2 small ranks " London" first) |
| coreference | The trophy would not fit in the brown suitcase because it was too                    | " big"                                       |
| agreement   | The keys to the old wooden cabinet                                                   | " are"                                       |
| code        | `def add(a, b):\n    return a +`                                                     | " b"                                         |

Each artifact's suggested head is chosen by a stated rule: the head with the largest attention from a named
query token to a named key token (recorded in `suggestedView.criterion`).

## Generation environment

Recorded in each artifact (`generator.environment`, `exportEnvironment`). The committed artifacts were produced
with Python 3.13, torch 2.7.1, transformers 5.17.0, numpy 2.1.3 on CPU.

## Reproducing

```bash
pip install -r scripts/generate-demo-artifacts/requirements.txt
python scripts/generate-demo-artifacts/export_pythia.py
python scripts/generate-demo-artifacts/export_tokenizers.py
python scripts/generate-demo-artifacts/generate_gpt2_demos.py
python scripts/generate-demo-artifacts/generate_fixtures.py
npm run verify:artifacts && npm test
```

Regenerated demo JSON differs only in timestamps; attention binaries are deterministic on the same
torch/CPU combination (other BLAS backends may differ in the last float32 bits, changing checksums).
