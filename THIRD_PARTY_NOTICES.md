# Third-party notices

The AttentionScope source code is MIT licensed. The following bundled assets and dependencies keep their own
licenses.

## Model assets

| Asset                                                                                                                     | Copyright  | License                                                          |
| ------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------- |
| GPT-2 small outputs and tokenizer (`public/demos/*`, `public/tokenizers/gpt2/`) — derived from `openai-community/gpt2`    | OpenAI     | MIT (https://github.com/openai/gpt-2/blob/master/LICENSE)        |
| Pythia-14m weights (`public/models/pythia-14m/weights.bin`) — from `EleutherAI/pythia-14m`, unmodified values, repackaged | EleutherAI | Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0) |
| GPT-NeoX-20B tokenizer (`public/models/pythia-14m/tokenizer.json`)                                                        | EleutherAI | Apache License 2.0                                               |

Modification notice (Apache-2.0 §4b): the Pythia-14m float16 tensors were extracted from `model.safetensors` and
concatenated into `weights.bin` with a JSON tensor table; tensor values are unchanged.

Citations: Radford et al. (2019), _Language Models are Unsupervised Multitask Learners_; Biderman et al. (2023),
_Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling_.

## Runtime dependencies (bundled into the site)

| Package                                      | License    |
| -------------------------------------------- | ---------- |
| react, react-dom, react-router-dom           | MIT        |
| zustand                                      | MIT        |
| radix-ui                                     | MIT        |
| katex (including its fonts)                  | MIT        |
| clsx                                         | MIT        |
| lucide-react                                 | ISC        |
| d3-scale-chromatic, d3-interpolate, d3-color | ISC        |
| dexie                                        | Apache-2.0 |

Run `npx license-checker --production --summary` for the complete transitive list.
