"""Generate the precomputed GPT-2 small demonstration artifacts.

For each curated prompt this runs openai-community/gpt2 (pinned revision,
float32, eager attention, eval mode, no dropout) on CPU and records:

* tokenization (ids, byte-level pieces, exact byte and UTF-16 spans)
* post-softmax attention probabilities for every layer and head
  (float32, layout [layer, head, query, key])
* next-token distribution at the final position and top-3 at every position
* logit lens at the final position (unembedding of each block's residual output)
* zero-ablation of every head's output (its slice of the merged head outputs,
  i.e. the input to attn.c_proj), with a full forward pass per head

Outputs public/demos/<id>.json, public/demos/<id>.attention.bin and
public/demos/manifest.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    GPT2_REPO,
    GPT2_REVISION,
    PUBLIC_DIR,
    build_token_records,
    environment_info,
    sha256_file,
    utc_now,
    write_json,
)

OUT = PUBLIC_DIR / "demos"
SCRIPT = "scripts/generate-demo-artifacts/generate_gpt2_demos.py"

DEMOS = [
    {
        "id": "induction",
        "title": "Induction: repeated sentence",
        "category": "In-context copying",
        "text": "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy",
        "target": " dog",
        "summary": "A sentence repeated twice. On the second copy, some heads attend from each token to the token that followed its earlier occurrence.",
        "focus": {"query": None, "key": ' dog'},
        "literature": "Olsson et al. (2022) describe 'induction heads' that implement this prefix-matching and copying pattern; for GPT-2 small, heads such as 5.1, 5.5 and 6.9 are commonly reported. Verify against the pattern here rather than assuming it.",
    },
    {
        "id": "ioi",
        "title": "Indirect object identification",
        "category": "Circuits",
        "text": "When Mary and John went to the store, John gave a drink to",
        "target": " Mary",
        "summary": "The model must predict the name that is not repeated. Compare attention from the final token to both names, and inspect the ablation table.",
        "focus": {"query": None, "key": ' Mary'},
        "literature": "Wang et al. (2022) identify 'name mover' heads in GPT-2 small (e.g. 9.9, 9.6, 10.0) using path patching on a distribution of prompts. A single-prompt zero-ablation is a much weaker test.",
    },
    {
        "id": "factual",
        "title": "Factual recall: landmark location",
        "category": "Knowledge",
        "text": "The Eiffel Tower is located in the city of",
        "target": " Paris",
        "summary": "A factual completion. Attention to the subject tokens is visible in several heads, but attention alone does not localize where the fact is stored.",
        "focus": {"query": None, "key": ' Tower'},
        "literature": "Meng et al. (2022) argue that factual associations in GPT models are mediated largely by mid-layer MLPs at the subject's last token, which attention maps cannot show.",
    },
    {
        "id": "coreference",
        "title": "Pronoun resolution",
        "category": "Coreference",
        "text": "The trophy would not fit in the brown suitcase because it was too",
        "target": " big",
        "summary": "A Winograd-style sentence. Check which nouns the pronoun 'it' attends to in different heads, and whether the next-token prediction depends on them.",
        "focus": {"query": ' it', "key": ' trophy'},
        "literature": "Clark et al. (2019) found BERT heads whose attention tracks coreference; similar behavior in GPT-2 should be verified, not assumed.",
    },
    {
        "id": "agreement",
        "title": "Subject–verb agreement",
        "category": "Syntax",
        "text": "The keys to the old wooden cabinet",
        "target": " are",
        "summary": "The verb must agree with 'keys', not the closer noun 'cabinet'. Compare the probability of ' are' with ' is' after ablating heads.",
        "focus": {"query": None, "key": ' keys'},
        "literature": "Linzen et al. (2016) introduced agreement across attractor nouns as a probe of syntactic sensitivity in language models.",
    },
    {
        "id": "code",
        "title": "Code: function body",
        "category": "Code",
        "text": "def add(a, b):\n    return a +",
        "target": " b",
        "summary": "Python source with a newline and indentation. Inspect how whitespace is tokenized and how the final token attends to the parameter list.",
        "focus": {"query": None, "key": ' b'},
        "literature": "GPT-2 was not trained specifically on code; its tokenizer splits runs of spaces into individual tokens.",
    },
]


def main() -> None:
    from transformers import AutoTokenizer, GPT2LMHeadModel

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    tok = AutoTokenizer.from_pretrained(GPT2_REPO, revision=GPT2_REVISION)
    model = GPT2LMHeadModel.from_pretrained(
        GPT2_REPO, revision=GPT2_REVISION, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cfg = model.config
    n_layer, n_head, d_model = cfg.n_layer, cfg.n_head, cfg.n_embd
    head_dim = d_model // n_head
    bos = cfg.bos_token_id
    special_ids = set(tok.all_special_ids)
    env = environment_info()
    OUT.mkdir(parents=True, exist_ok=True)

    def decode(i: int) -> str:
        return tok.decode([int(i)])

    def top_entries(probs: torch.Tensor, logits: torch.Tensor, k: int) -> list[dict]:
        vals, idx = probs.topk(k)
        return [
            {"id": int(i), "text": decode(i), "prob": float(v), "logit": float(logits[i])}
            for v, i in zip(vals.tolist(), idx.tolist())
        ]

    manifest_entries = []
    for demo in DEMOS:
        text = demo["text"]
        ids = [bos] + tok(text)["input_ids"]
        target_ids = tok(demo["target"])["input_ids"]
        if len(target_ids) != 1:
            raise SystemExit(f"target {demo['target']!r} is not a single token")
        target = target_ids[0]
        tokens = build_token_records(text, tok, ids, special_ids)
        T = len(ids)
        input_ids = torch.tensor([ids])

        residuals: list[torch.Tensor] = []
        def keep_residual(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            residuals.append(hidden[0].detach())

        hooks = [blk.register_forward_hook(keep_residual) for blk in model.transformer.h]
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)
        for h in hooks:
            h.remove()

        attn = torch.stack([a[0] for a in out.attentions]).numpy().astype(np.float32)  # (L, H, T, T)
        assert attn.shape == (n_layer, n_head, T, T)
        row_err = float(np.abs(attn.sum(-1) - 1.0).max())
        upper = float(np.triu(attn, 1).max(initial=0.0))
        if row_err > 1e-5 or upper != 0.0:
            raise SystemExit(f"{demo['id']}: attention fails normalization/causality checks")

        logits = out.logits[0]
        probs = torch.softmax(logits.double(), -1)
        last_probs, last_logits = probs[-1], logits[-1]
        target_rank = int((last_probs > last_probs[target]).sum())

        per_position = []
        for t in range(T):
            per_position.append({"top": top_entries(probs[t], logits[t], 3)})

        logit_lens = []
        for layer, resid in enumerate(residuals):
            with torch.no_grad():
                ll = model.lm_head(model.transformer.ln_f(resid[-1]))
            lp = torch.softmax(ll.double(), -1)
            logit_lens.append(
                {
                    "layer": layer,
                    "top": top_entries(lp, ll, 5),
                    "targetProb": float(lp[target]),
                    "targetRank": int((lp > lp[target]).sum()),
                }
            )
        residual_norms = [[float(x) for x in r.norm(dim=-1).tolist()] for r in residuals]
        with torch.no_grad():
            emb = model.transformer.wte(input_ids)[0] + model.transformer.wpe(torch.arange(T))
        residual_norms.insert(0, [float(x) for x in emb.norm(dim=-1).tolist()])

        # --- head zero-ablation sweep ---
        base_logp = torch.log_softmax(last_logits.double(), -1)
        ablations = []
        for layer in range(n_layer):
            row = []
            for head in range(n_head):
                lo, hi = head * head_dim, (head + 1) * head_dim

                def pre_hook(module, args, lo=lo, hi=hi):
                    x = args[0].clone()
                    x[..., lo:hi] = 0.0
                    return (x,)

                handle = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(pre_hook)
                with torch.no_grad():
                    abl_logits = model(input_ids).logits[0, -1]
                handle.remove()
                abl_logp = torch.log_softmax(abl_logits.double(), -1)
                abl_p = abl_logp.exp()
                kl = float((base_logp.exp() * (base_logp - abl_logp)).sum())
                top_id = int(abl_p.argmax())
                row.append(
                    {
                        "targetProb": float(abl_p[target]),
                        "targetLogit": float(abl_logits[target]),
                        "top1": {"id": top_id, "text": decode(top_id), "prob": float(abl_p[top_id])},
                        "klFromBaseline": max(kl, 0.0),
                    }
                )
            ablations.append(row)
        # sanity: a no-op hook reproduces the baseline exactly
        handle = model.transformer.h[0].attn.c_proj.register_forward_pre_hook(lambda m, a: (a[0].clone(),))
        with torch.no_grad():
            noop = model(input_ids).logits[0, -1]
        handle.remove()
        if not torch.equal(noop, last_logits):
            raise SystemExit("no-op hook changed logits; ablation harness is not trustworthy")

        # Suggested starting view: the head with the largest attention from the
        # focus query token (default: final token) to the first occurrence of
        # the focus key token. Chosen from the data, not from the literature.
        texts = [t["text"] for t in tokens]
        q_text, k_text = demo["focus"]["query"], demo["focus"]["key"]
        q_idx = T - 1 if q_text is None else texts.index(q_text)
        k_idx = texts.index(k_text)
        flat = int(np.argmax(attn[:, :, q_idx, k_idx]))
        best_l, best_h = divmod(flat, n_head)
        suggested = {
            "layer": best_l,
            "head": best_h,
            "queryIndex": q_idx,
            "keyIndex": k_idx,
            "weight": float(attn[best_l, best_h, q_idx, k_idx]),
            "criterion": f"head with the largest attention from token {q_idx} ({texts[q_idx]!r}) to token {k_idx} ({texts[k_idx]!r})",
        }

        bin_name = f"{demo['id']}.attention.bin"
        attn.astype("<f4").tofile(OUT / bin_name)
        artifact = {
            "schemaVersion": 1,
            "id": demo["id"],
            "title": demo["title"],
            "category": demo["category"],
            "summary": demo["summary"],
            "literature": demo["literature"],
            "suggestedView": suggested,
            "precomputed": True,
            "model": {
                "id": "gpt2",
                "displayName": "GPT-2 small",
                "repo": GPT2_REPO,
                "revision": GPT2_REVISION,
                "url": f"https://huggingface.co/{GPT2_REPO}/tree/{GPT2_REVISION}",
                "license": "MIT",
                "citation": "Radford et al. (2019). Language Models are Unsupervised Multitask Learners.",
                "architecture": {
                    "family": "gpt2",
                    "numLayers": n_layer,
                    "numHeads": n_head,
                    "numKeyValueHeads": n_head,
                    "attentionType": "multi-head",
                    "hiddenSize": d_model,
                    "headDim": head_dim,
                    "intermediateSize": 4 * d_model,
                    "vocabSize": cfg.vocab_size,
                    "maxPositions": cfg.n_positions,
                    "positional": "learned-absolute",
                    "parallelResidual": False,
                    "layerNormEps": cfg.layer_norm_epsilon,
                    "activation": cfg.activation_function,
                    "attentionScale": f"1/sqrt({head_dim})",
                    "tiedEmbeddings": True,
                },
            },
            "tokenizer": {"name": "GPT-2 byte-level BPE", "repo": GPT2_REPO, "revision": GPT2_REVISION, "prependBos": True, "bosToken": {"id": bos, "text": "<|endoftext|>"}},
            "generator": {
                "script": SCRIPT,
                "framework": "transformers GPT2LMHeadModel",
                "attnImplementation": "eager",
                "dtype": "float32",
                "device": "cpu",
                "mode": "eval (dropout disabled)",
                "deterministicAlgorithms": True,
                "seed": 0,
                "generatedAt": utc_now(),
                "environment": env,
            },
            "input": {"text": text, "tokens": tokens},
            "attention": {
                "file": bin_name,
                "dtype": "float32",
                "byteOrder": "little-endian",
                "layout": ["layer", "head", "query", "key"],
                "shape": {"batch": 1, "layers": n_layer, "heads": n_head, "queryLength": T, "keyLength": T},
                "kind": "post-softmax",
                "mask": "causal",
                "normalization": "softmax over keys (each query row sums to 1)",
                "sha256": sha256_file(OUT / bin_name),
                "checks": {"maxRowSumError": row_err, "maxUpperTriangle": upper},
            },
            "predictions": {
                "position": T - 1,
                "top": top_entries(last_probs, last_logits, 10),
                "target": {"id": target, "text": decode(target), "prob": float(last_probs[target]), "logit": float(last_logits[target]), "rank": target_rank},
                "perPosition": per_position,
            },
            "logitLens": {
                "position": T - 1,
                "method": "lm_head(ln_f(residual stream after block l)); the final layer matches the model output",
                "layers": logit_lens,
            },
            "residualNorms": {
                "method": "L2 norm of the residual stream: index 0 = token + position embedding, index l+1 = after block l",
                "values": residual_norms,
            },
            "ablation": {
                "precomputed": True,
                "method": "zero-head-output",
                "location": "head slice of the merged attention output, i.e. the input to attn.c_proj (W_O), at every position",
                "measuredAt": {"position": T - 1, "target": {"id": target, "text": decode(target)}},
                "baseline": {"targetProb": float(last_probs[target]), "targetLogit": float(last_logits[target])},
                "caveat": "Zero-ablation moves activations off-distribution; effects are evidence about this prompt and this intervention only.",
                "heads": ablations,
            },
        }
        json_name = f"{demo['id']}.json"
        write_json(OUT / json_name, artifact, compact=True)
        manifest_entries.append(
            {
                "id": demo["id"],
                "title": demo["title"],
                "category": demo["category"],
                "summary": demo["summary"],
                "model": "gpt2",
                "modelDisplayName": "GPT-2 small",
                "tokens": T,
                "file": json_name,
                "sha256": sha256_file(OUT / json_name),
                "attentionFile": bin_name,
                "attentionSha256": artifact["attention"]["sha256"],
                "bytes": (OUT / json_name).stat().st_size + (OUT / bin_name).stat().st_size,
            }
        )
        top = artifact["predictions"]["top"][0]
        print(f"  suggested {best_l}.{best_h} ({suggested['weight']:.3f}) {suggested['criterion']}")
        print(f"{demo['id']:12s} T={T:3d} top1={top['text']!r} p={top['prob']:.3f} target={demo['target']!r} p={last_probs[target]:.3f} rank={target_rank}")

    write_json(
        OUT / "manifest.json",
        {
            "schemaVersion": 1,
            "generatedAt": utc_now(),
            "generator": SCRIPT,
            "notice": "Precomputed from real forward passes of GPT-2 small. Regenerate with the script above to reproduce.",
            "demos": manifest_entries,
        },
    )


if __name__ == "__main__":
    main()
