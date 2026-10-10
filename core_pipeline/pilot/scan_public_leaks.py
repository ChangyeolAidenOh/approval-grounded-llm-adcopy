"""Scan a masked catalog build for leaked client identifiers.

Read-only. Forbidden tokens are derived from the real build (v1 HTML), the
registry parquet and the judgments parquet. Pass the public HTML path as the
first argument. Every counted section must report 0 before publishing.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

V1 = Path("data/copy/catalog/__BRAND___copy_catalog_v1.html")
PUB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/copy/catalog/copy_catalog_public.html")
REG = Path("data/copy/live_assets/copy_registry_live_v1.parquet")
JUDGE = Path("data/copy/catalog/judgments_v1n.parquet")

BRAND_TOKENS = [
    "__BRAND__", "__BRAND__", "__BRAND__", "__BRAND__",
    "__PRODUCT_PROBIOTIC__", "__PRODUCT_PROBIOTIC__", "__PRODUCT_PROBIOTIC__",
    "__PRODUCT_VITAMIN__", "__PRODUCT_VITAMIN__", "__PRODUCT_OMEGA__", "__PRODUCT_OMEGA__",
    "알티지 __PRODUCT_OMEGA__", "line_rtg900", "rtg900",
    "TrueVitamin", "Truevitamin", "TRUE VITAMIN", "TrueOmega", "TRUE OMEGA",
    "___APPROVAL_TAG___", "approved_ref.pdf", "DEMO-", "demo build",
]
MASK_VALUES = ["브랜디아", "Brandia", "프로바이오 스토리", "트루비타", "오메가트루", "북유럽",
               "베이직", "눈건강", "여성", "고함량", "엑스트라", "집중", "코엔자임"]
GENERIC_LINES = {"line_pro", "line_active", "line_booster"}
FORBIDDEN_TOP5_KEYS = {"copy", "pdf", "simui", "raw_line", "raw_series"}
MIN_SKELETON = 8


def load(p: Path):
    html = p.read_text(encoding="utf-8")
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    if not m:
        raise SystemExit(f"no data block in {p}")
    return html, json.loads(m.group(1))


def ws(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


def skeleton(s: str) -> str:
    s = str(s)
    for t in sorted(BRAND_TOKENS + MASK_VALUES, key=len, reverse=True):
        s = s.replace(t, "")
    return ws(s)


def snippets(text: str, tok: str, n: int = 2, w: int = 30) -> list:
    out = []
    for m in re.finditer(re.escape(tok), text):
        out.append(text[max(0, m.start() - w): m.end() + w].replace("\n", " "))
        if len(out) >= n:
            break
    return out


def section(title: str):
    print(f"\n== {title}")


def main():
    v1_html, v1 = load(V1)
    pub_html, pub = load(PUB)
    reg = pd.read_parquet(REG)
    judge = pd.read_parquet(JUDGE)
    print(f"public file {PUB} | v1 records {len(v1)} | public records {len(pub)} | registry {len(reg)}")
    total = 0

    section("A. brand tokens in public html (raw bytes)")
    for t in BRAND_TOKENS:
        c = pub_html.count(t)
        if c:
            total += c
            print(f"  {t!r}: {c}")
            for s in snippets(pub_html, t):
                print(f"     ...{s}...")
    print("  title:", re.search(r"<title>(.*?)</title>", pub_html, re.S).group(1).strip())
    print("  filename contains brand:", "__BRAND__" in PUB.name.lower())

    section("B. real simui / pdf values from v1 present in public html")
    real_simui = {str(t["simui"]) for r in v1 for t in r["top5"]}
    real_pdf = {str(t["pdf"]) for r in v1 for t in r["top5"]}
    hit_simui = [s for s in real_simui if s in pub_html]
    hit_pdf = [s for s in real_pdf if s in pub_html]
    total += len(hit_simui) + len(hit_pdf)
    print(f"  real simui unique {len(real_simui)} | simui hits {len(hit_simui)} | pdf hits {len(hit_pdf)}")
    pub_ref = {str(t.get("ref", t.get("simui"))) for r in pub for t in r["top5"]}
    print(f"  public ref unique {len(pub_ref)} | sample {sorted(pub_ref)[:3]}")

    section("C. line / series / id identifiers in public data")
    lines = set(judge["line"].astype(str)) | set(reg["line"].astype(str))
    pub_lines = {str(r["line"]) for r in pub} | {str(t["line"]).split("/")[-1] for r in pub for t in r["top5"]}
    leaked_lines = sorted((pub_lines & lines) - GENERIC_LINES)
    total += len(leaked_lines)
    print(f"  real lines present (non-generic): {leaked_lines}")
    print(f"  public lines: {sorted(pub_lines)}")
    ascii_ids = {m for cid in judge["custom_id"] for m in re.findall(r"gen-[a-z0-9]+-([a-z]+)-([a-z0-9]+)-", cid)[0]}
    pub_ascii = set()
    for r in pub:
        for grp in re.findall(r"gen-[a-z0-9]+-([a-z]+)-([a-z0-9]+)-", r["id"]):
            pub_ascii |= set(grp)
    total += len(pub_ascii & ascii_ids)
    print(f"  ascii line ids in public ids: {sorted(pub_ascii & ascii_ids)} | id sample {pub[0]['id']}")

    section("D. approved anchor text exposure (skeleton match vs registry)")
    reg_skel = {skeleton(c) for c in reg["full_copy"].astype(str)}
    reg_skel = {s for s in reg_skel if len(s) >= MIN_SKELETON}
    pub_anchor = [(r["id"], t.get("copy", "")) for r in pub for t in r["top5"]]
    exposed = [(i, c) for i, c in pub_anchor if c and skeleton(c) in reg_skel]
    total += len(exposed)
    print(f"  public anchor entries {len(pub_anchor)} | with copy text {sum(1 for _, c in pub_anchor if c)} | matching registry {len(exposed)}")

    section("E. generated field values identical to an approved copy")
    gen_dup = [(r["id"], k, v) for r in pub for k, v in r["fields"].items() if skeleton(v) in reg_skel]
    total += len(gen_dup)
    print(f"  generated fields matching registry skeleton {len(gen_dup)}")
    for i, k, v in gen_dup[:3]:
        print(f"     {i} {k}: {v[:40]!r}")

    section("F. masking artefacts")
    dd = pub_html.count("오메가트루트루") + pub_html.count("__PRODUCT_OMEGA__DD") + pub_html.count("브랜디아브랜디아")
    total += dd
    print(f"  double-replace artefacts: {dd}")
    print("  mask value counts:", {v: pub_html.count(v) for v in MASK_VALUES})

    section("G. forbidden keys in public top5 entries")
    bad_keys = [(r["id"], sorted(set(t) & FORBIDDEN_TOP5_KEYS)) for r in pub for t in r["top5"] if set(t) & FORBIDDEN_TOP5_KEYS]
    total += len(bad_keys)
    print(f"  entries with forbidden keys {len(bad_keys)} | key set sample {sorted(pub[0]['top5'][0]) if pub[0]['top5'] else None}")

    print(f"\nTOTAL LEAK HITS (A+B+C+D+E+F+G): {total}")


if __name__ == "__main__":
    main()
