"""Per-format token usage and cost of a finished Message Batch.

Reads batch_id from --out-dir, pulls results and aggregates usage per fmt
(custom_id prefix gen-<fmt>- or sup-<fmt>-). Read-only unless --write, which
saves usage.parquet into --out-dir. Prices are the constants below; compare
the printed total against the console figure before relying on them.
"""
import argparse
import re
from pathlib import Path

import anthropic
import pandas as pd

PRICE_IN = 3.0
PRICE_OUT = 15.0
PRICE_CACHE_WRITE = 3.75
PRICE_CACHE_READ = 0.30
BATCH_DISCOUNT = 0.5


def row_cost(r) -> float:
    base = (r["input_tokens"] * PRICE_IN + r["output_tokens"] * PRICE_OUT
            + r["cache_write"] * PRICE_CACHE_WRITE + r["cache_read"] * PRICE_CACHE_READ)
    return base / 1e6 * BATCH_DISCOUNT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    batch_id = (out / "batch_id").read_text().strip()
    client = anthropic.Anthropic()
    rows = []
    for res in client.messages.batches.results(batch_id):
        m = re.match(r"^(?:gen|sup)-([a-z0-9]+)-", res.custom_id)
        fmt = m.group(1) if m else "?"
        if res.result.type != "succeeded":
            rows.append({"custom_id": res.custom_id, "fmt": fmt, "status": res.result.type,
                         "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_write": 0})
            continue
        u = res.result.message.usage
        rows.append({"custom_id": res.custom_id, "fmt": fmt, "status": "succeeded",
                     "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                     "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                     "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0})
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "succeeded"].copy()
    ok["cost_usd"] = ok.apply(row_cost, axis=1)
    print(f"batch {batch_id}: {len(df)} results | succeeded {len(ok)}")
    g = ok.groupby("fmt").agg(n=("custom_id", "size"), in_mean=("input_tokens", "mean"),
                              out_mean=("output_tokens", "mean"), cache_read_mean=("cache_read", "mean"),
                              cache_write_mean=("cache_write", "mean"), cost_usd=("cost_usd", "sum"))
    print(g.round(1).to_string())
    print(f"total cost (computed, batch {int(BATCH_DISCOUNT * 100)}% off): ${ok['cost_usd'].sum():.3f}")
    print(f"per-req mean: ${ok['cost_usd'].mean():.4f} | per-item (5/req): ${ok['cost_usd'].mean() / 5:.4f}")
    if args.write:
        dest = out / "usage.parquet"
        if dest.exists():
            print(f"skip: {dest} exists")
            return
        ok.to_parquet(dest, index=False)
        print(f"saved {dest}")


if __name__ == "__main__":
    main()
