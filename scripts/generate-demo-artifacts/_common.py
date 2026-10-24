"""Shared helpers for AttentionScope offline artifact generation.

Every artifact written by these scripts is derived from a real forward pass of a
pinned, licensed model revision. Nothing here synthesizes attention values.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DIR = REPO_ROOT / "public"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

GPT2_REPO = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
PYTHIA_REPO = "EleutherAI/pythia-14m"
PYTHIA_REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if compact:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def to_float16_bytes(arr: np.ndarray) -> bytes:
    """Little-endian IEEE-754 binary16 bytes."""
    return np.ascontiguousarray(arr, dtype="<f2").tobytes()


def environment_info() -> dict[str, str]:
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Byte-level BPE helpers (GPT-2 style byte <-> unicode mapping)
# ---------------------------------------------------------------------------


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_DECODER = {v: k for k, v in bytes_to_unicode().items()}


def token_bytes(tokenizer, token_id: int) -> bytes:
    """Raw bytes represented by a byte-level BPE token (special tokens -> UTF-8 of content)."""
    piece = tokenizer.convert_ids_to_tokens(int(token_id))
    if token_id in tokenizer.all_special_ids or any(ch not in _BYTE_DECODER for ch in piece):
        return piece.encode("utf-8")
    return bytes(_BYTE_DECODER[ch] for ch in piece)


def build_token_records(text: str, tokenizer, ids: Iterable[int], special_ids: set[int]) -> list[dict]:
    """Token records with exact byte spans and UTF-16 spans into ``text``.

    Byte spans come from concatenating token byte strings; they are only
    reported when the concatenation reproduces the UTF-8 encoding of ``text``
    exactly (prepended special tokens are given empty spans). UTF-16 spans are
    used by the browser for highlighting.
    """
    ids = [int(i) for i in ids]
    encoded = text.encode("utf-8")
    # Map byte index -> UTF-16 code-unit index (at char starts) for highlighting.
    byte_to_u16: dict[int, int] = {}
    b = 0
    u = 0
    for ch in text:
        byte_to_u16[b] = u
        b += len(ch.encode("utf-8"))
        u += len(ch.encode("utf-16-le")) // 2
    byte_to_u16[b] = u

    def u16_floor(byte_index: int) -> int:
        while byte_index not in byte_to_u16:
            byte_index -= 1
        return byte_to_u16[byte_index]

    def u16_ceil(byte_index: int) -> int:
        while byte_index not in byte_to_u16:
            byte_index += 1
        return byte_to_u16[byte_index]

    records = []
    cursor = 0
    stream = b""
    for index, tid in enumerate(ids):
        raw = token_bytes(tokenizer, tid)
        is_special = tid in special_ids
        prepended = is_special and encoded[cursor : cursor + len(raw)] != raw
        if prepended:
            start = end = cursor
        else:
            start, end = cursor, cursor + len(raw)
            cursor = end
            stream += raw
        partial = (start not in byte_to_u16) or (end not in byte_to_u16)
        records.append(
            {
                "index": index,
                "id": tid,
                "piece": tokenizer.convert_ids_to_tokens(tid),
                "text": raw.decode("utf-8", errors="replace"),
                "bytesHex": raw.hex(),
                "byteStart": start,
                "byteEnd": end,
                "u16Start": u16_floor(start),
                "u16End": u16_ceil(end),
                "special": is_special,
                "synthetic": prepended,
                "partialCharacter": partial,
            }
        )
    if stream != encoded:
        raise ValueError("token bytes do not reproduce the input text; offsets would be invalid")
    return records


def ensure_dir(path: Path) -> Path:
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Exact float64 Hugging Face reference
# ---------------------------------------------------------------------------


def _eager_float64(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    import torch

    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is None:
        # transformers omits the mask for non-eager backends when causality can be
        # implied; build it explicitly (no KV cache, so query/key positions align).
        q_len, k_len = weights.shape[-2], weights.shape[-1]
        causal = torch.ones(q_len, k_len, dtype=torch.bool, device=weights.device).tril(k_len - q_len)
        weights = weights.masked_fill(~causal, float("-inf"))
    else:
        if attention_mask.dtype == torch.bool:
            weights = weights.masked_fill(~attention_mask, float("-inf"))
        else:
            weights = weights + attention_mask
    weights = torch.softmax(weights, dim=-1)  # stays in the input dtype (float64)
    output = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return output, weights


def load_pythia_exact_float64():
    """transformers GPT-NeoX with every operation in float64.

    Stock eager attention casts the softmax to float32 and the rotary module
    builds cos/sin in float32; both are replaced here so the model can serve as
    a tight reference for independent float64 implementations.
    """
    import types

    import torch
    from transformers import AttentionInterface, GPTNeoXForCausalLM

    AttentionInterface.register("eager_float64", _eager_float64)
    model = GPTNeoXForCausalLM.from_pretrained(
        PYTHIA_REPO, revision=PYTHIA_REVISION, torch_dtype=torch.float64, attn_implementation="eager_float64"
    ).eval()
    rope = model.gpt_neox.rotary_emb
    params = model.config.rope_parameters
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    dim = int(head_dim * params.get("partial_rotary_factor", 1.0))
    base = params["rope_theta"]

    def exact_forward(self, x, position_ids):
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        freqs = position_ids[..., None].to(torch.float64) * inv
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    rope.forward = types.MethodType(exact_forward, rope)
    return model
