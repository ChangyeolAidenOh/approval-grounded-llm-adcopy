"""Merge v1 judgments with the v1.1 supplement judgments.

Dry-run prints band distributions per track, the aspect-coverage check
(every fmt/series/line cell should now carry 3 aspects), and the track B
comparison: natively generated pack-vocabulary copy versus the v2
substituted copy on the same cells. --write saves judgments_v1_1.parquet
(refuses to overwrite). v1 rows are tagged track "v2".
"""
import argparse
from pathlib import Path

import pandas as pd

V1 = Path("data/copy/catalog/judgments_v1n.parquet")
SUP = Path("data/copy/catalog/judgments_sup.parquet")
OUT = Path("data/copy/catalog/judgments_v1_1.parquet")
BANDS = ["승인동등", "실집행근접", "분포내", "신중"]


def band_table(groups: dict) -> pd.DataFrame:
    t = pd.DataFrame({k: v["band"].value_counts() for k, v in groups.items()})
    return t.reindex(BANDS).fillna(0).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    v1 = pd.read_parquet(V1)
    sup = pd.read_parquet(SUP)
    if "track" not in v1.columns:
        v1["track"] = "v2"
    v1["track"] = v1["track"].replace("", "v2")
    missing = [c for c in v1.columns if c not in sup.columns]
    if missing:
        raise SystemExit(f"supplement lacks columns {missing}")
    both = pd.concat([v1, sup[v1.columns]], ignore_index=True)
    dup = int(both.duplicated(["custom_id", "item_idx"]).sum())
    print(f"v1 {len(v1)} + sup {len(sup)} = {len(both)} | duplicate ids {dup} | risk>=1 {int((both['n_risk'] > 0).sum())}")
    print("\nbands by track:")
    print(band_table({t: g for t, g in both.groupby("track")}).to_string())

    cov = both.groupby(["fmt", "series", "line"])["aspect"].nunique()
    print(f"\ncells {len(cov)} | with <3 aspects: {cov[cov < 3].to_dict()}")

    b = sup[sup["track"] == "b"]
    keys = set(zip(b["fmt"], b["series"], b["line"], b["aspect"]))
    base = v1[[k in keys for k in zip(v1["fmt"], v1["series"], v1["line"], v1["aspect"])]]
    print(f"\ntrack b native ({len(b)}) vs v2 substituted ({len(base)}) on the same cells:")
    print(band_table({"native": b, "substituted": base}).to_string())
    print(f"mean best_sim native {b['best_sim'].mean():.3f} | substituted {base['best_sim'].mean():.3f}")
    a = sup[sup["track"] == "a"]
    print(f"\ntrack a ({len(a)}): mean best_sim {a['best_sim'].mean():.3f} | top band share "
          f"{(a['band'].isin(BANDS[:2])).mean() * 100:.0f}% (v2 {(v1['band'].isin(BANDS[:2])).mean() * 100:.0f}%)")

    if not args.write:
        print("\ndry-run: pass --write to save")
        return
    if OUT.exists():
        print(f"stop: {OUT} exists")
        return
    both.to_parquet(OUT, index=False)
    print(f"saved {OUT} ({len(both)})")


if __name__ == "__main__":
    main()
