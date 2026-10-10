"""Precompute embeddings for the approved-copy anchor corpus.

bge-m3 via local Ollama (/api/embed). Scope: copy_registry rows with
in_voc_scope True (727). Embedded text is full_copy (slash-joined
lines as-is — matches how candidate copies will be entered).

Output data/copy/copy_embeddings.json:
  {model, dim, items: [{num, vec}]}
Vectors are L2-normalized here so the dashboard does cosine via plain
dot product. Registry metadata is NOT duplicated — join by num.

Read-only over inputs; --write saves the JSON (standalone run).
Requires: ollama serve running, bge-m3 pulled.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import requests

REGISTRY = Path("data/copy/copy_registry.parquet")
OUT = Path("data/copy/copy_embeddings.json")
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "bge-m3"
BATCH = 32


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = requests.post(OLLAMA_URL, json={"model": MODEL, "input": texts},
                         timeout=120)
    resp.raise_for_status()
    return resp.json()["embeddings"]


def l2norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [round(x / n, 6) for x in v]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    reg = pd.read_parquet(REGISTRY)
    rows = reg[reg["in_voc_scope"]].reset_index(drop=True)
    texts = rows["full_copy"].astype(str).tolist()
    nums = rows["num"].astype(int).tolist()
    print(f"anchors to embed: {len(texts)} (model {MODEL})")

    vecs: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        vecs.extend(embed_batch(chunk))
        print(f"  embedded {min(i + BATCH, len(texts))}/{len(texts)}")
    dim = len(vecs[0])
    vecs = [l2norm(v) for v in vecs]
    print(f"dim: {dim}")

    sanity = sum(a * b for a, b in zip(vecs[0], vecs[1]))
    print(f"sanity cos(anchor0, anchor1) = {sanity:.4f}")

    if args.write:
        payload = {"model": MODEL, "dim": dim,
                   "items": [{"num": n, "vec": v} for n, v in zip(nums, vecs)]}
        OUT.write_text(json.dumps(payload, ensure_ascii=False))
        size_mb = OUT.stat().st_size / 1e6
        print(f"written: {OUT} ({size_mb:.1f} MB)")
    else:
        print("dry-run only, nothing written")


if __name__ == "__main__":
    main()