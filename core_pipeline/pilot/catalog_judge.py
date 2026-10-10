"""카탈로그 판정 사전 계산 — 1,348 유효 산출물 x 3축.

근접도: bge-m3 x 727 앵커, 표시용 dedup = (full_copy, line) 그룹 접기.
라인 전이: Top5 동일 라인 비율. 리스크: 위험 어간 x 효능마커 공존 (전 필드).
밴드: >=0.76 승인동등 / >=0.71 실집행근접 / >=0.58 분포내 / 미만 신중.
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CAND = Path("data/copy/gen_production_v2/candidates_v2_norm.parquet")
EMB = Path("data/copy/copy_embeddings.json")
REG = Path("data/copy/copy_registry.parquet")
RISK = Path("data/copy/risk_lexicon_v0.json")
OUT = Path("data/copy/catalog/judgments_v1n.parquet")
OLLAMA = "http://127.0.0.1:11434/api/embed"

BANDS = [(0.76, "승인동등"), (0.71, "실집행근접"), (0.58, "분포내"), (-1, "신중")]
MAIN_KEYS = {"f1": ["copy1", "copy2"], "sda": ["copy1", "copy2"],
             "meta": ["copy1", "copy2"], "bsmo": ["sub", "main", "desc1", "desc2"],
             "bspc": ["title", "desc1", "desc2"]}


def embed_texts(texts, chunk=64):
    vecs = []
    for i in range(0, len(texts), chunk):
        body = json.dumps({"model": "bge-m3", "input": texts[i:i + chunk]}).encode()
        req = urllib.request.Request(OLLAMA, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            vecs.extend(json.loads(r.read())["embeddings"])
        if (i // chunk) % 5 == 0:
            print(f"  embed {min(i + chunk, len(texts))}/{len(texts)}")
    v = np.array(vecs, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def risk_scan(fields, risk):
    hits = []
    for k, v in fields.items():
        for kw in risk["risk_keywords"]:
            if kw in v:
                for m in risk["efficacy_markers"]:
                    if m in v:
                        hits.append(f"{k}:{kw}x{m}")
    return hits


def band_of(sim):
    for th, name in BANDS:
        if sim >= th:
            return name
    return "신중"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cand", default=str(CAND))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    cand_path, out_path = Path(args.cand), Path(args.out)
    print(f"cand: {cand_path} -> out: {out_path}")
    cand = pd.read_parquet(cand_path)
    valid = cand[cand["n_issues"] == 0].reset_index(drop=True)
    print(f"후보 {len(cand)} → 유효 {len(valid)} (이슈 제외 {len(cand) - len(valid)})")
    reg = pd.read_parquet(REG).set_index("num")
    emb = json.loads(EMB.read_text())
    anchors = np.array([it["vec"] for it in emb["items"]], dtype=np.float32)
    anchors = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9)
    nums = [it["num"] for it in emb["items"]]
    a_line = reg.loc[nums, "line"].tolist()
    a_copy = reg.loc[nums, "full_copy"].tolist()
    a_pdf = reg.loc[nums, "pdf_file"].tolist()
    a_simui = reg.loc[nums, "simui"].astype(str).tolist()
    a_series = reg.loc[nums, "series"].tolist()
    risk = json.loads(RISK.read_text())

    payloads = [json.loads(p) for p in valid["payload"]]
    texts = [" / ".join(str(pl.get(k, "")).strip()
                        for k in MAIN_KEYS[f] if str(pl.get(k, "")).strip())
             for f, pl in zip(valid["fmt"], payloads)]
    work = valid if args.all else valid.head(20)
    print(f"대상: {len(work)} ({'전수' if args.all else '표본 dry-run'})")
    q = embed_texts(texts[:len(work)])
    sims = q @ anchors.T

    rows = []
    for i, (_, r) in enumerate(work.iterrows()):
        pl = payloads[i]
        order = np.argsort(-sims[i])
        seen, top = set(), []
        for j in order:
            key = (a_copy[j], a_series[j], a_line[j])
            if key in seen:
                continue
            seen.add(key)
            top.append({"sim": round(float(sims[i, j]), 4), "line": f"{a_series[j]}/{a_line[j]}", "raw_line": a_line[j], "raw_series": a_series[j],
                        "copy": a_copy[j], "pdf": a_pdf[j], "simui": a_simui[j]})
            if len(top) == 5:
                break
        best = top[0]
        same = sum(1 for t in top if t["raw_series"] == r["series"] and t["raw_line"] == r["line"])
        hits = risk_scan({k: str(v) for k, v in pl.items()}, risk)
        rows.append({"custom_id": r["custom_id"], "item_idx": r["item_idx"],
                     "fmt": r["fmt"], "series": r["series"], "line": r["line"],
                     "forme": r["forme"], "aspect": r["aspect"],
                     "payload": r["payload"], "judge_text": texts[i],
                     "best_sim": best["sim"], "band": band_of(best["sim"]),
                     "best_anchor_line": best["line"],
                     "same_line_top5": same, "n_risk": len(hits),
                     "risk_hits": "|".join(hits),
                     "top5": json.dumps(top, ensure_ascii=False),
                     "track": str(r.get("track", ""))})
    out = pd.DataFrame(rows)
    if not args.all:
        print("\n--- 표본 판정 미리보기 ---")
        for _, r in out.head(8).iterrows():
            print(f"[{r['fmt']}/{r['line']}] {r['judge_text'][:40]}")
            print(f"   {r['band']} ({r['best_sim']:.3f}) | 동일라인 {r['same_line_top5']}/5 | risk {r['n_risk']}")
        print("\ndry-run — 전수는 --all 단독 실행")
        return
    if out_path.exists():
        print(f"중단: {out_path} 이미 존재")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"\n저장: {out_path} ({len(out)}행)")
    print("밴드 분포:", out["band"].value_counts().to_dict())
    print("fmt별 승인동등+실집행근접 비율:")
    for fmt, g in out.groupby("fmt"):
        hi = int(g["band"].isin(["승인동등", "실집행근접"]).sum())
        print(f"  {fmt}: {hi}/{len(g)} ({hi / len(g) * 100:.0f}%)")
    print("라인 전이 주의 (best 앵커 타라인):",
          int((out["best_anchor_line"] != out["series"] + "/" + out["line"]).sum()), "건")
    print("리스크 신호 ≥1:", int((out["n_risk"] > 0).sum()), "건")


if __name__ == "__main__":
    main()