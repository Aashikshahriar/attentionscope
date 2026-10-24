"""Generate independent reference fixtures for the TypeScript test-suite.

* tests/fixtures/tokenizer-cases.json  - Hugging Face (Rust) tokenizer ids for
  GPT-2 and GPT-NeoX on English, whitespace, Unicode and multilingual inputs.
* tests/fixtures/pythia-reference.json - exact float64 transformers outputs of
  Pythia-14m, including interventions implemented with PyTorch hooks.
* tests/fixtures/toy-attention.json    - NumPy scaled dot-product attention
  intermediates for the educational Q/K/V module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402
from _common import (  # noqa: E402
    FIXTURE_DIR,
    GPT2_REPO,
    GPT2_REVISION,
    PYTHIA_REPO,
    PYTHIA_REVISION,
    environment_info,
    load_pythia_exact_float64,
    write_json,
)

TOKENIZER_CASES = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    " leading space and trailing space ",
    "I'm sure they'll say we've done it, didn't they?",
    "Numbers: 3.14159, 2026, 1,000,000 and 0x1F.",
    "def add(a, b):\n    return a + b\n",
    "tabs\tand\nnew\n\nlines",
    "two  spaces, three   spaces, eight        spaces.",
    "   \n  \t ",
    "emoji 🙂🚀 and ZWJ 👩‍💻",
    "naïve café résumé",
    "Café decomposed accent",
    "আমি বাংলায় গান গাই",
    "नमस्ते दुनिया, यह एक परीक्षण है।",
    "안녕하세요, 세계! 트랜스포머는 주의를 사용합니다.",
    "مرحبا بالعالم",
    "日本語のテキストも試します。",
    "<|endoftext|>Document boundary<|endoftext|> next",
    "URL https://example.org/path?q=attention&x=1",
    "MiXeD CaSe AND ALLCAPS words",
    "a" * 40,
]


def tokenizer_fixture() -> dict:
    from transformers import AutoTokenizer

    gpt2 = AutoTokenizer.from_pretrained(GPT2_REPO, revision=GPT2_REVISION)
    neox = AutoTokenizer.from_pretrained(PYTHIA_REPO, revision=PYTHIA_REVISION)
    cases = []
    for text in TOKENIZER_CASES:
        cases.append(
            {
                "text": text,
                "gpt2": gpt2(text, add_special_tokens=False)["input_ids"],
                "neox": neox(text, add_special_tokens=False)["input_ids"],
                "gpt2Decoded": gpt2.decode(gpt2(text, add_special_tokens=False)["input_ids"]),
                "neoxDecoded": neox.decode(neox(text, add_special_tokens=False)["input_ids"]),
            }
        )
    return {
        "source": "transformers AutoTokenizer (Rust tokenizers backend)",
        "gpt2": {"repo": GPT2_REPO, "revision": GPT2_REVISION},
        "neox": {"repo": PYTHIA_REPO, "revision": PYTHIA_REVISION},
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Pythia reference with interventions
# ---------------------------------------------------------------------------

_PATCH: dict = {}


def _patched_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Exact float64 attention that can replace one head's probabilities."""
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    q_len, k_len = weights.shape[-2], weights.shape[-1]
    causal = torch.ones(q_len, k_len, dtype=torch.bool).tril(k_len - q_len)
    weights = weights.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(weights, dim=-1)
    if _PATCH and _PATCH["layer"] == module.layer_idx:
        probs = probs.clone()
        probs[0, _PATCH["head"]] = _PATCH["matrix"]
    output = torch.matmul(probs, value).transpose(1, 2).contiguous()
    return output, probs


def uniform_causal(T: int) -> torch.Tensor:
    m = torch.tril(torch.ones(T, T, dtype=torch.float64))
    return m / m.sum(-1, keepdim=True)


def previous_token(T: int) -> torch.Tensor:
    m = torch.zeros(T, T, dtype=torch.float64)
    m[0, 0] = 1.0
    for i in range(1, T):
        m[i, i - 1] = 1.0
    return m


def pythia_fixture() -> dict:
    from transformers import AttentionInterface, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(PYTHIA_REPO, revision=PYTHIA_REVISION)
    model = load_pythia_exact_float64()
    AttentionInterface.register("eager_float64_patch", _patched_attention)

    def logits_subset(logits: torch.Tensor, ids: list[int]) -> list[float]:
        return [float(logits[i]) for i in ids]

    cases = []
    texts = [
        ("The quick brown fox jumps over the lazy dog.", True),
        ("When Mary and John went to the store, John gave a drink to", True),
        ("Attention is not explanation", False),
    ]
    for text, bos in texts:
        ids = ([0] if bos else []) + tok(text, add_special_tokens=False)["input_ids"]
        T = len(ids)
        x = torch.tensor([ids])
        model.config._attn_implementation = "eager_float64"
        residuals: list[torch.Tensor] = []
        hooks = [
            layer.register_forward_hook(lambda m, a, o: residuals.append((o[0] if isinstance(o, tuple) else o)[0].detach()))
            for layer in model.gpt_neox.layers
        ]
        with torch.no_grad():
            out = model(x, output_attentions=True)
        for h in hooks:
            h.remove()
        logits = out.logits[0, -1]
        top = logits.topk(40).indices.tolist()
        probe_ids = sorted(set(top + list(range(0, model.config.vocab_size, 997))))
        attn = torch.stack([a[0] for a in out.attentions])  # (L, H, T, T)

        # pre-softmax scores of layer 0, head 0, recomputed with hooks from q/k
        import transformers.models.gpt_neox.modeling_gpt_neox as neox_mod

        qk: dict = {}
        orig = neox_mod.apply_rotary_pos_emb

        def spy(q, k, cos, sin, unsqueeze_dim=1):
            r = orig(q, k, cos, sin, unsqueeze_dim)
            if "q" not in qk:
                qk["q"], qk["k"] = r[0][0, 0].clone(), r[1][0, 0].clone()
            return r

        neox_mod.apply_rotary_pos_emb = spy
        with torch.no_grad():
            model(x)
        neox_mod.apply_rotary_pos_emb = orig
        hd = model.config.hidden_size // model.config.num_attention_heads
        scores00 = (qk["q"] @ qk["k"].T) / np.sqrt(hd)
        scores00 = scores00.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))

        # logit lens at the final position
        lens = []
        for li, r in enumerate(residuals):
            with torch.no_grad():
                ll = model.get_output_embeddings()(model.gpt_neox.final_layer_norm(r[-1]))
            lens.append({"layer": li, "top5": ll.topk(5).indices.tolist(), "logitsAtProbe": logits_subset(ll, probe_ids)})

        interventions = []
        # (a) zero the output of one head before the output projection (attention.dense)
        L, H = 2, 1

        def zero_hook(module, args, lo=H * hd, hi=(H + 1) * hd):
            y = args[0].clone()
            y[..., lo:hi] = 0.0
            return (y,)

        handle = model.gpt_neox.layers[L].attention.dense.register_forward_pre_hook(zero_hook)
        with torch.no_grad():
            o2 = model(x, output_attentions=True)
        handle.remove()
        interventions.append(
            {
                "spec": {"kind": "zero-head-output", "layer": L, "head": H},
                "logitsAtProbe": logits_subset(o2.logits[0, -1], probe_ids),
                "attentionLayer3Head0Row": o2.attentions[3][0, 0, -1].tolist(),
            }
        )
        # (b, c) replace one head's attention probabilities with a valid row-stochastic matrix
        for kind, layer, head, mat in [
            ("replace-attention-uniform", 1, 0, uniform_causal(T)),
            ("replace-attention-previous-token", 3, 2, previous_token(T)),
        ]:
            _PATCH.update({"layer": layer, "head": head, "matrix": mat})
            model.config._attn_implementation = "eager_float64_patch"
            with torch.no_grad():
                o3 = model(x, output_attentions=True)
            _PATCH.clear()
            model.config._attn_implementation = "eager_float64"
            interventions.append(
                {
                    "spec": {"kind": kind, "layer": layer, "head": head},
                    "logitsAtProbe": logits_subset(o3.logits[0, -1], probe_ids),
                    "patchedAttention": o3.attentions[layer][0, head].tolist(),
                }
            )

        cases.append(
            {
                "text": text,
                "prependBos": bos,
                "ids": ids,
                "probeIds": probe_ids,
                "logitsAtProbe": logits_subset(logits, probe_ids),
                "top10": logits.topk(10).indices.tolist(),
                "attention": attn.tolist(),
                "scoresLayer0Head0": [[(float(v) if np.isfinite(float(v)) else None) for v in row] for row in scores00.tolist()],
                "logitLens": lens,
                "interventions": interventions,
            }
        )
    return {
        "source": "transformers GPT-NeoX, exact float64 (see _common.load_pythia_exact_float64)",
        "repo": PYTHIA_REPO,
        "revision": PYTHIA_REVISION,
        "environment": environment_info(),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Toy attention reference (NumPy)
# ---------------------------------------------------------------------------


def toy_reference(X, WQ, WK, WV, WO, temperature, use_scaling, causal, padding):
    T = X.shape[0]
    heads = []
    for h in range(len(WQ)):
        Q, K, V = X @ WQ[h], X @ WK[h], X @ WV[h]
        dk = Q.shape[1]
        scale = (1.0 / np.sqrt(dk) if use_scaling else 1.0) / temperature
        raw = Q @ K.T
        scaled = raw * scale
        allowed = np.ones((T, T), dtype=bool)
        if causal:
            allowed &= np.tril(np.ones((T, T), dtype=bool))
        allowed &= ~np.asarray(padding, dtype=bool)[None, :]
        masked = np.where(allowed, scaled, -np.inf)
        A = np.zeros((T, T))
        for i in range(T):
            if allowed[i].any():
                row = masked[i] - masked[i][allowed[i]].max()
                e = np.where(allowed[i], np.exp(row), 0.0)
                A[i] = e / e.sum()
        O = A @ V
        heads.append(
            {
                "Q": Q.tolist(),
                "K": K.tolist(),
                "V": V.tolist(),
                "rawScores": raw.tolist(),
                "scaledScores": scaled.tolist(),
                "allowed": allowed.tolist(),
                "attention": A.tolist(),
                "output": O.tolist(),
                "scale": scale,
            }
        )
    concat = np.concatenate([np.array(h["output"]) for h in heads], axis=1)
    Y = concat @ WO
    return {"heads": heads, "concat": concat.tolist(), "projected": Y.tolist()}


def toy_fixture() -> dict:
    rng = np.random.default_rng(20240607)
    cases = []
    configs = [
        ("single head, no mask", 4, 4, 1, 2, 1.0, True, False, [0, 0, 0, 0]),
        ("single head, causal", 5, 4, 1, 3, 1.0, True, True, [0, 0, 0, 0, 0]),
        ("causal + padding, fully masked first row", 4, 4, 1, 2, 1.0, True, True, [1, 0, 0, 1]),
        ("two heads, temperature 2", 5, 6, 2, 3, 2.0, True, False, [0, 0, 0, 0, 1]),
        ("two heads, no scaling, causal", 6, 4, 2, 2, 0.5, False, True, [0, 0, 0, 0, 0, 0]),
    ]
    for name, T, d, H, dk, temp, scaling, causal, padding in configs:
        X = rng.normal(size=(T, d)).round(3)
        WQ = [rng.normal(size=(d, dk)).round(3) for _ in range(H)]
        WK = [rng.normal(size=(d, dk)).round(3) for _ in range(H)]
        WV = [rng.normal(size=(d, dk)).round(3) for _ in range(H)]
        WO = rng.normal(size=(H * dk, d)).round(3)
        ref = toy_reference(X, WQ, WK, WV, WO, temp, scaling, causal, padding)
        cases.append(
            {
                "name": name,
                "config": {"temperature": temp, "useScaling": scaling, "causal": causal, "padding": [bool(p) for p in padding]},
                "X": X.tolist(),
                "WQ": [w.tolist() for w in WQ],
                "WK": [w.tolist() for w in WK],
                "WV": [w.tolist() for w in WV],
                "WO": WO.tolist(),
                "expected": ref,
            }
        )
    return {"source": "NumPy float64 reference (generate_fixtures.toy_reference)", "cases": cases}


def main() -> None:
    torch.manual_seed(0)
    write_json(FIXTURE_DIR / "tokenizer-cases.json", tokenizer_fixture())
    write_json(FIXTURE_DIR / "toy-attention.json", toy_fixture(), compact=True)
    write_json(FIXTURE_DIR / "pythia-reference.json", pythia_fixture(), compact=True)
    for p in sorted(FIXTURE_DIR.glob("*.json")):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    _ = _common
    main()
