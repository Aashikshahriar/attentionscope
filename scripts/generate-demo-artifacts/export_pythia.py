"""Export EleutherAI/pythia-14m for in-browser execution.

Writes public/models/pythia-14m/{weights.bin, tokenizer.json, manifest.json}.

The weights are copied bit-exactly from the pinned float16 safetensors file (the
checkpoint is published in float16, so no precision is lost). After export the
script runs an independent NumPy forward pass over the exported bytes and
compares logits and attention probabilities with the Hugging Face eager
implementation. Export fails if they disagree.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    PUBLIC_DIR,
    PYTHIA_REPO,
    PYTHIA_REVISION,
    environment_info,
    load_pythia_exact_float64,
    sha256_file,
    utc_now,
    write_json,
)

OUT = PUBLIC_DIR / "models" / "pythia-14m"
MAX_SEQUENCE_LENGTH = 64


def tensor_order(num_layers: int) -> list[str]:
    names = ["gpt_neox.embed_in.weight"]
    for i in range(num_layers):
        p = f"gpt_neox.layers.{i}."
        names += [
            p + "input_layernorm.weight",
            p + "input_layernorm.bias",
            p + "post_attention_layernorm.weight",
            p + "post_attention_layernorm.bias",
            p + "attention.query_key_value.weight",
            p + "attention.query_key_value.bias",
            p + "attention.dense.weight",
            p + "attention.dense.bias",
            p + "mlp.dense_h_to_4h.weight",
            p + "mlp.dense_h_to_4h.bias",
            p + "mlp.dense_4h_to_h.weight",
            p + "mlp.dense_4h_to_h.bias",
        ]
    names += ["gpt_neox.final_layer_norm.weight", "gpt_neox.final_layer_norm.bias", "embed_out.weight"]
    return names


# ---------------------------------------------------------------------------
# Independent NumPy reference (float64) used to validate the exported layout.
# ---------------------------------------------------------------------------


def _layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _gelu(x):
    from math import erf

    v = np.vectorize(erf)
    return 0.5 * x * (1.0 + v(x / np.sqrt(2.0)))


def numpy_forward(weights: dict[str, np.ndarray], cfg: dict, ids: list[int]):
    """Exact float64 forward pass written directly from the GPT-NeoX equations."""
    W = {k: v.astype(np.float64) for k, v in weights.items()}
    T = len(ids)
    H = cfg["num_attention_heads"]
    D = cfg["hidden_size"]
    hd = D // H
    rot = int(hd * cfg["rotary_pct"])
    eps = cfg["layer_norm_eps"]
    inv_freq = 1.0 / (cfg["rotary_emb_base"] ** (np.arange(0, rot, 2) / rot))
    freqs = np.outer(np.arange(T), inv_freq)
    emb = np.concatenate([freqs, freqs], -1)
    cos, sin = np.cos(emb), np.sin(emb)

    def rope(x):  # x: (H, T, hd)
        xr, xp = x[..., :rot], x[..., rot:]
        half = rot // 2
        rotated = np.concatenate([-xr[..., half:], xr[..., :half]], -1)
        return np.concatenate([xr * cos + rotated * sin, xp], -1)

    h = W["gpt_neox.embed_in.weight"][ids]
    attentions = []
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    for i in range(cfg["num_hidden_layers"]):
        p = f"gpt_neox.layers.{i}."
        x1 = _layer_norm(h, W[p + "input_layernorm.weight"], W[p + "input_layernorm.bias"], eps)
        qkv = x1 @ W[p + "attention.query_key_value.weight"].T + W[p + "attention.query_key_value.bias"]
        qkv = qkv.reshape(T, H, 3 * hd).transpose(1, 0, 2)
        q, k, v = qkv[..., :hd], qkv[..., hd : 2 * hd], qkv[..., 2 * hd :]
        q, k = rope(q), rope(k)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
        scores = np.where(mask, -np.inf, scores)
        scores = scores - scores.max(-1, keepdims=True)
        a = np.exp(scores)
        a = a / a.sum(-1, keepdims=True)
        attentions.append(a)
        o = (a @ v).transpose(1, 0, 2).reshape(T, D)
        attn_out = o @ W[p + "attention.dense.weight"].T + W[p + "attention.dense.bias"]
        x2 = _layer_norm(h, W[p + "post_attention_layernorm.weight"], W[p + "post_attention_layernorm.bias"], eps)
        m = _gelu(x2 @ W[p + "mlp.dense_h_to_4h.weight"].T + W[p + "mlp.dense_h_to_4h.bias"])
        mlp_out = m @ W[p + "mlp.dense_4h_to_h.weight"].T + W[p + "mlp.dense_4h_to_h.bias"]
        h = h + attn_out + mlp_out
    h = _layer_norm(h, W["gpt_neox.final_layer_norm.weight"], W["gpt_neox.final_layer_norm.bias"], eps)
    logits = h @ W["embed_out.weight"].T
    return logits, np.stack(attentions)


def main() -> None:
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    cfg_path = hf_hub_download(PYTHIA_REPO, "config.json", revision=PYTHIA_REVISION)
    st_path = hf_hub_download(PYTHIA_REPO, "model.safetensors", revision=PYTHIA_REVISION)
    tok_path = hf_hub_download(PYTHIA_REPO, "tokenizer.json", revision=PYTHIA_REVISION)
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    weights = load_file(st_path)

    OUT.mkdir(parents=True, exist_ok=True)
    order = tensor_order(cfg["num_hidden_layers"])
    missing = [n for n in order if n not in weights]
    if missing:
        raise SystemExit(f"missing tensors: {missing}")
    ignored = sorted(set(weights) - set(order))

    tensors = []
    offset = 0
    with open(OUT / "weights.bin", "wb") as f:
        for name in order:
            arr = weights[name]
            if arr.dtype != np.float16:
                raise SystemExit(f"{name} is {arr.dtype}; expected float16 checkpoint")
            data = np.ascontiguousarray(arr, dtype="<f2").tobytes()
            f.write(data)
            tensors.append({"name": name, "shape": list(arr.shape), "byteOffset": offset, "elementCount": int(arr.size)})
            offset += len(data)

    shutil.copyfile(tok_path, OUT / "tokenizer.json")

    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    manifest = {
        "schemaVersion": 1,
        "id": "pythia-14m",
        "displayName": "Pythia-14m",
        "kind": "browser-inference",
        "description": "Smallest model of the EleutherAI Pythia suite (6 layers, 4 heads). Executed in your browser by AttentionScope's own GPT-NeoX implementation, with every attention probability exposed and editable.",
        "source": {
            "repo": PYTHIA_REPO,
            "revision": PYTHIA_REVISION,
            "url": f"https://huggingface.co/{PYTHIA_REPO}/tree/{PYTHIA_REVISION}",
            "license": "Apache-2.0",
            "citation": "Biderman et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. ICML.",
        },
        "architecture": {
            "family": "gpt-neox",
            "numLayers": cfg["num_hidden_layers"],
            "numHeads": cfg["num_attention_heads"],
            "numKeyValueHeads": cfg["num_attention_heads"],
            "attentionType": "multi-head",
            "hiddenSize": cfg["hidden_size"],
            "headDim": hd,
            "intermediateSize": cfg["intermediate_size"],
            "vocabSize": cfg["vocab_size"],
            "maxPositions": cfg["max_position_embeddings"],
            "positional": "rotary",
            "rotaryDims": int(hd * cfg["rotary_pct"]),
            "rotaryBase": cfg["rotary_emb_base"],
            "parallelResidual": cfg["use_parallel_residual"],
            "layerNormEps": cfg["layer_norm_eps"],
            "activation": "gelu-erf",
            "attentionScale": f"1/sqrt({hd})",
            "tiedEmbeddings": cfg["tie_word_embeddings"],
            "qkvLayout": "per-head interleaved [q|k|v] in query_key_value (out = heads * 3 * headDim)",
        },
        "tokenizer": {
            "file": "tokenizer.json",
            "sha256": sha256_file(OUT / "tokenizer.json"),
            "type": "byte-level-bpe",
            "name": "GPT-NeoX-20B tokenizer",
            "bosToken": {"id": 0, "text": "<|endoftext|>"},
        },
        "weights": {
            "file": "weights.bin",
            "dtype": "float16",
            "byteOrder": "little-endian",
            "byteLength": offset,
            "sha256": sha256_file(OUT / "weights.bin"),
            "tensors": tensors,
            "ignoredCheckpointEntries": ignored,
        },
        "runtime": {"maxSequenceLength": MAX_SEQUENCE_LENGTH, "computePrecision": "float32", "deterministic": True},
        "capabilities": {
            "attentionWeights": True,
            "preSoftmaxScores": True,
            "headOutputAblation": True,
            "attentionReplacement": True,
            "logitLens": True,
        },
        "exportedAt": utc_now(),
        "exportEnvironment": environment_info(),
    }

    # --- validation: NumPy reference on exported bytes vs Hugging Face eager ---
    raw = np.fromfile(OUT / "weights.bin", dtype="<f2")
    exported = {
        t["name"]: raw[t["byteOffset"] // 2 : t["byteOffset"] // 2 + t["elementCount"]].reshape(t["shape"])
        for t in tensors
    }
    tok = AutoTokenizer.from_pretrained(PYTHIA_REPO, revision=PYTHIA_REVISION)
    text = "The quick brown fox jumps over the lazy dog."
    ids = [0] + tok(text)["input_ids"]

    hf64 = load_pythia_exact_float64()
    with torch.no_grad():
        out = hf64(torch.tensor([ids]), output_attentions=True)
    hf_logits = out.logits[0].numpy()
    hf_attn = np.stack([x[0].numpy() for x in out.attentions])
    hf32 = GPTNeoXForCausalLM.from_pretrained(
        PYTHIA_REPO, revision=PYTHIA_REVISION, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        out32 = hf32(torch.tensor([ids]), output_attentions=True)
    ref_logits, ref_attn = numpy_forward(exported, cfg, ids)
    logit_err = float(np.abs(hf_logits - ref_logits).max())
    attn_err = float(np.abs(hf_attn - ref_attn).max())
    fp32_logit_gap = float(np.abs(out32.logits[0].double().numpy() - ref_logits).max())
    fp32_attn_gap = float(max(np.abs(x[0].double().numpy() - ref_attn[i]).max() for i, x in enumerate(out32.attentions)))
    print(f"validation (exact float64 HF vs NumPy reference): logits {logit_err:.3e}, attention {attn_err:.3e}")
    print(f"information (stock HF float32 eager vs float64): logits {fp32_logit_gap:.3e}, attention {fp32_attn_gap:.3e}")
    if logit_err > 1e-8 or attn_err > 1e-10:
        raise SystemExit("NumPy reference disagrees with Hugging Face; refusing to export")
    if int(np.argmax(out32.logits[0, -1])) != int(np.argmax(ref_logits[-1])):
        raise SystemExit("float32 and float64 disagree on the top-1 prediction")
    manifest["validation"] = {
        "reference": "independent NumPy float64 forward pass over the exported bytes",
        "comparedAgainst": "transformers GPTNeoXForCausalLM in float64 with float64 rotary tables and a float64 eager softmax (see _common.load_pythia_exact_float64)",
        "text": text,
        "prependBos": True,
        "maxAbsLogitDiff": logit_err,
        "maxAbsAttentionDiff": attn_err,
        "float32NumericalGap": {
            "note": "Stock transformers eager attention computes softmax and rotary tables in float32. Pre-softmax scores in Pythia-14m reach ~180, so that rounding produces these differences from exact float64. AttentionScope computes activations in float64 in the browser.",
            "maxAbsLogitDiff": fp32_logit_gap,
            "maxAbsAttentionDiff": fp32_attn_gap,
        },
    }
    write_json(OUT / "manifest.json", manifest)
    print(f"wrote {OUT} ({offset / 1e6:.1f} MB weights)")


if __name__ == "__main__":
    main()
