"""Copy pinned tokenizer.json files used for in-browser tokenization comparisons.

Writes public/tokenizers/<id>/tokenizer.json and public/tokenizers/manifest.json.
The GPT-NeoX tokenizer is shared with the Pythia-14m model directory.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import GPT2_REPO, GPT2_REVISION, PUBLIC_DIR, PYTHIA_REPO, PYTHIA_REVISION, sha256_file, utc_now, write_json  # noqa: E402

OUT = PUBLIC_DIR / "tokenizers"


def main() -> None:
    entries = []
    gpt2_dir = OUT / "gpt2"
    gpt2_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(hf_hub_download(GPT2_REPO, "tokenizer.json", revision=GPT2_REVISION), gpt2_dir / "tokenizer.json")
    entries.append(
        {
            "id": "gpt2",
            "name": "GPT-2 byte-level BPE",
            "usedBy": "GPT-2 small (precomputed demos)",
            "repo": GPT2_REPO,
            "revision": GPT2_REVISION,
            "license": "MIT",
            "path": "tokenizers/gpt2/tokenizer.json",
            "sha256": sha256_file(gpt2_dir / "tokenizer.json"),
            "bos": {"id": 50256, "text": "<|endoftext|>"},
        }
    )
    neox_file = PUBLIC_DIR / "models" / "pythia-14m" / "tokenizer.json"
    if not neox_file.exists():
        raise SystemExit("run export_pythia.py first")
    entries.append(
        {
            "id": "gpt-neox",
            "name": "GPT-NeoX-20B byte-level BPE",
            "usedBy": "Pythia-14m (live in-browser model)",
            "repo": PYTHIA_REPO,
            "revision": PYTHIA_REVISION,
            "license": "Apache-2.0",
            "path": "models/pythia-14m/tokenizer.json",
            "sha256": sha256_file(neox_file),
            "bos": {"id": 0, "text": "<|endoftext|>"},
        }
    )
    write_json(OUT / "manifest.json", {"schemaVersion": 1, "generatedAt": utc_now(), "tokenizers": entries})
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
