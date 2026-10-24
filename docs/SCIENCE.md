# Scientific notes

## Attention

For each head h with input X (after LayerNorm):

```
Q_h = X W_Q^h     K_h = X W_K^h     V_h = X W_V^h            (T × d_k)
S_h = Q_h K_h^T / sqrt(d_k) + M                               pre-softmax scores
A_h = softmax_row(S_h)                                        post-softmax probabilities
O_h = A_h V_h
MHA(X) = [O_1 | … | O_H] W_O + b_O
```

- Rows index **queries** (the position reading), columns index **keys** (the position read).
- Causal mask: `M_ij = 0` if `j ≤ i`, else `−∞`. Masked probabilities are exactly 0; each row sums to 1.
- Softmax is computed stably by subtracting the row maximum over visible keys.
- A row with no visible key (possible only with padding) is undefined; the Q/K/V lab returns zeros and flags it.
- Dropout is disabled (evaluation mode). All runs are deterministic.
- Both models use standard multi-head attention (one K/V head per query head). Grouped-/multi-query checkpoints
  are rejected by the loader rather than visualized as if each head had its own keys.

|                  | GPT-2 small         | Pythia-14m                                        |
| ---------------- | ------------------- | ------------------------------------------------- |
| Layers × heads   | 12 × 12             | 6 × 4                                             |
| d_model / d_head | 768 / 64            | 128 / 32                                          |
| Positions        | learned absolute    | rotary on first 8 of 32 dims (NeoX `rotate_half`) |
| Residual         | sequential, pre-LN  | parallel: h + Attn(LN₁h) + MLP(LN₂h)              |
| Activation       | GELU (tanh approx.) | GELU (erf)                                        |
| Scale            | 1/√64               | 1/√32                                             |

## Metrics

- Entropy `H_i = −Σ_{j≤i} a_ij log(a_ij + 1e-12)` in nats; normalized by `log(i+1)`. Head values average rows i ≥ 1.
- Mean attention distance `Σ_j a_ij (i − j)`.
- Previous-token, first-token (sink) and self scores: means of `a_{i,i−1}`, `a_{i,0}`, `a_{i,i}` over i ≥ 1.
- Similarity: `1 − mean_i JSD(a_i, b_i)` (bits) or Pearson r over visible entries. Similar patterns do not imply
  similar function (values and W_O differ).
- Attention rollout (Abnar & Zuidema 2020): `R = Â_{L−1} ⋯ Â_0`, `Â = rownorm(αI + (1−α) mean_h A_h)`. Ignores
  values, W_O, MLPs and LayerNorm — a heuristic, not a causal measurement.
- Logit lens: final LayerNorm + unembedding applied to the residual stream after each layer.

## Interventions (live model only)

| Kind                              | Definition                                  | Location                         |
| --------------------------------- | ------------------------------------------- | -------------------------------- |
| Zero head output                  | `O_h ← 0`; pattern unchanged; W_O bias kept | input slice of `attention.dense` |
| Replace attention: uniform        | `A'_ij = 1/(i+1)` for `j ≤ i`               | after softmax, before `A V`      |
| Replace attention: previous token | one-hot at `i−1` (row 0 → itself)           | after softmax, before `A V`      |
| Replace attention: first token    | one-hot at 0                                | after softmax, before `A V`      |

Replacements are validated (shape, row sums = 1, no mass on masked keys, non-negative) before running. Results
report the next-token distribution before/after at the last position, Δ log P(target), and
`KL(P_base ‖ P_intervened)` in nats. Raw logits are not compared in the UI because logits are only defined up to an
additive constant per position (GPT-2's shift by ~200 under some ablations).

Zero ablation and pattern replacement move activations off-distribution; an effect is evidence about this
intervention on this input only.

## Validation

| Check             | Reference                                                                                           | Result                                                                    |
| ----------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Pythia export     | NumPy float64 forward vs transformers in exact float64                                              | logits 4e-12, attention 2e-12                                             |
| Browser GPT-NeoX  | same reference, 3 inputs                                                                            | logits < 1e-8; attention < 1e-7 (float32 storage); scores relative < 1e-7 |
| Interventions     | PyTorch forward pre-hook on `attention.dense`; custom attention function substituting probabilities | logits < 1e-8                                                             |
| Ablation locality | layers ≤ intervention bit-identical, downstream changed                                             | passes                                                                    |
| Tokenizers        | Rust `tokenizers` via transformers                                                                  | 42/42 exact id sequences; decode identical                                |
| Toy attention     | NumPy                                                                                               | every intermediate < 1e-12                                                |
| Demo artifacts    | checksums, row sums ≤ 1e-5, zero upper triangle, token ids re-tokenized                             | passes                                                                    |

A note on the reference: stock transformers eager attention computes softmax in float32 and builds rotary tables
in float32 even for float64 models. With Pythia-14m's large pre-softmax scores (up to ~180) this changes final
logits by ~6e-3 relative to exact arithmetic. The exact reference (`_common.load_pythia_exact_float64`) registers a
float64 attention function and float64 rotary tables.

## Interpretation

Attention describes where a head reads from, not why the model predicts what it does. A head's effect is
`A_h V_h W_O^h`; heavily attended tokens may contribute little, many heads park mass on the first token as a no-op
("attention sink"), and information has already mixed across positions in earlier layers. Labels such as
"induction head" are hypotheses checked with metrics and interventions on specific inputs.

References: Vaswani et al. 2017; Jain & Wallace 2019; Wiegreffe & Pinter 2019; Clark et al. 2019; Abnar & Zuidema
2020; Elhage et al. 2021; Olsson et al. 2022; Wang et al. 2022; Meng et al. 2022; Biderman et al. 2023.
