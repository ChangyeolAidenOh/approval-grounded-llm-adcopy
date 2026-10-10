"""단위 어휘 교정 v1 — 비타민 포→팩, 부스터 개→병 (조사 받침 보정 포함).

candidates_v2.parquet 무접촉, candidates_v2_norm.parquet 신규 산출.
dry-run: 패턴별 건수 + before/after 표본. --write: 저장.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SRC = Path("data/copy/gen_production_v2/candidates_v2.parquet")
OUT = Path("data/copy/gen_production_v2/candidates_v2_norm.parquet")

JOSA = {"를": "을", "가": "이", "로": "으로", "와": "과", "면": "이면",
        "요": "이요", "예요": "이에요", "야": "이야"}
KEEP = ["에", "은", "을", "만", "씩", "의", "도", "라", "에서", "까지", "부터", ""]
NUM = r"[한두세네1-9]"


def make_sub(old, new):
    josa_alt = "|".join(sorted(set(list(JOSA) + KEEP), key=len, reverse=True))
    pat = re.compile(rf"(?P<pre>{NUM}\s?){old}(?P<josa>{josa_alt})(?=$|[^가-힣])")

    def repl(m):
        j = m.group("josa")
        return m.group("pre") + new + JOSA.get(j, j)
    return pat, repl


SUBS_VIT = [make_sub("포", "팩")]
SUBS_BOOST = [make_sub("개", "병"), make_sub("포", "병")]
BOOST_PHRASES = [("포장 뜯어", "뚜껑 열어"), ("뜯으면", "열면"), ("뜯고", "열고"), ("뜯어서", "열어서"),
                 ("낱개 포장", "1병 포장"), ("개별 포장", "1병 포장")]


def normalize(text, line):
    subs = SUBS_BOOST if line == "line_booster" else SUBS_VIT
    out, n = text, 0
    for pat, repl in subs:
        out, k = pat.subn(repl, out)
        n += k
    if line == "line_booster":
        for old, new in BOOST_PHRASES:
            if old in out:
                out = out.replace(old, new)
                n += 1
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    df = pd.read_parquet(SRC)
    rows, samples, n_changed, n_subs = [], [], 0, 0
    for _, r in df.iterrows():
        if r["series"] != "비타민":
            rows.append({**r, "unit_normalized": False})
            continue
        pl = json.loads(r["payload"])
        new_pl, cnt = {}, 0
        for k, v in pl.items():
            nv, c = normalize(str(v), r["line"])
            new_pl[k] = nv
            cnt += c
        if cnt:
            n_changed += 1
            n_subs += cnt
            if len(samples) < 15:
                samples.append((r["line"], json.dumps(pl, ensure_ascii=False)[:70],
                                json.dumps(new_pl, ensure_ascii=False)[:70]))
        rows.append({**r, "payload": json.dumps(new_pl, ensure_ascii=False),
                     "unit_normalized": bool(cnt)})
    out = pd.DataFrame(rows)
    vit = df[df["series"] == "비타민"]
    print(f"전체 {len(df)} | 비타민 {len(vit)} | 치환 발생 산출물 {n_changed} | 총 치환 {n_subs}")
    boost = out[(out["line"] == "line_booster") & out["unit_normalized"]]
    print(f"부스터 치환: {len(boost)}건")
    print("\n--- before → after 표본 ---")
    for line, b, a in samples:
        print(f"[{line}]\n  {b}\n  {a}")
    if not args.write:
        print("\ndry-run — 저장은 --write 단독 실행")
        return
    if OUT.exists():
        print(f"중단: {OUT} 이미 존재")
        return
    out.to_parquet(OUT, index=False)
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()