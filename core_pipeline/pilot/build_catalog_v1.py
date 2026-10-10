"""{BRAND} 카피 카탈로그 v1 빌더 — 정적 단일 HTML.

입력: data/copy/catalog/judgments_v1_1.parquet (v1 1,348 + v1.1 보충 판정)
출력: data/copy/catalog/{brand}_copy_catalog_v1.html (사내 정본)
      data/copy/catalog/copy_catalog_public.html (--public, 마스킹 공개판)

인자 없이: 통계·빌드 계획만 출력 (dry-run).
--write: 정본 빌드. --write --public: 공개판 빌드 (둘 다 덮어쓰기 거부).
벡터 미내장 — 판정 사전 계산 결과만 실어 용량 최소화.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

JUDGE = Path("data/copy/catalog/judgments_v1_1.parquet")
OUT = Path("data/copy/catalog/{brand}_copy_catalog_v1_1.html")
OUT_DEMO = Path("data/copy/catalog/copy_catalog_public.html")

FMT_LABEL = {
    "f1": "이미지 소재 (GFA·스페셜DA 메인)",
    "sda": "스페셜DA 520 전용",
    "bspc": "브랜드검색 PC",
    "bsmo": "브랜드검색 MO",
    "meta": "메타 피드",
}
FIELD_LABEL = {
    "copy1": "1행", "copy2": "2행", "title": "타이틀", "desc1": "설명1",
    "desc2": "설명2", "card_sub": "카드 서브", "card_title": "카드 타이틀",
    "card_desc1": "카드 설명1", "card_desc2": "카드 설명2",
    "sub": "서브타이틀", "main": "메인타이틀",
}
DEMO_MAP = {
    "{BRAND}": "브랜디아", "{BRAND}": "Brandia", "{BRAND}": "BRANDIA", "{brand}": "brandia",
    "{PRODUCT_PROBIOTIC}": "프로바이오 스토리", "{PRODUCT_PROBIOTIC}": "프로바이오 스토리",
    "{PRODUCT_PROBIOTIC}": "프로바이오 스토리", "{PRODUCT_PROBIOTIC}": "프로바이오 스토리",
    "{PRODUCT_VITAMIN}": "트루비타", "{PRODUCT_VITAMIN}": "트루비타",
    "{PRODUCT_OMEGA} 900": "오메가트루 900", "{PRODUCT_OMEGA}": "오메가트루", "{PRODUCT_OMEGA}": "오메가트루",
    "{ORIGIN}": "북유럽",
}
LINE_MAP = {
    "line_base": "베이직", "line_i": "눈건강", "line_woman": "여성", "line_pro": "line_pro", "line_active": "line_active",
    "line_rtg900": "고함량", "line_x": "엑스트라", "line_focus": "집중", "line_booster": "line_booster", "line_coq10": "코엔자임",
}
DEMO_N = {"f1": 40, "sda": 20, "bspc": 20, "bsmo": 20, "meta": 20}
_MASK_RE = re.compile("|".join(re.escape(k) for k in sorted(DEMO_MAP, key=len, reverse=True)))
REGISTRY = Path("data/copy/live_assets/copy_registry_live_v1.parquet")


def mask(s: str) -> str:
    return _MASK_RE.sub(lambda m: DEMO_MAP[m.group(0)], s)


def mask_line(line: str) -> str:
    return LINE_MAP.get(line, line)


def ws(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


def build_live_index() -> dict:
    reg = pd.read_parquet(REGISTRY, columns=["full_copy", "series", "line", "live_used"])
    idx = {}
    for c, s, l, u in reg.itertuples(index=False):
        k = ws(c)
        for key in ((k, s, l), (k, l), (k,)):
            idx[key] = bool(idx.get(key, False) or u)
    return idx


def to_records(df: pd.DataFrame, demo: bool) -> list:
    recs = []
    live_idx = build_live_index() if demo else {}
    ref_idx = {}
    miss = 0
    n_anchor = 0
    for n, (_, r) in enumerate(df.iterrows()):
        payload = json.loads(r["payload"])
        top5 = json.loads(r["top5"])
        series, line = r["series"], r["line"]
        rid = f"{r['custom_id']}#{int(r['item_idx'])}"
        if demo:
            payload = {k: mask(str(v)) for k, v in payload.items()}
            pub5 = []
            for t in top5:
                k = ws(t["copy"])
                lu = live_idx.get((k, t["raw_series"], t["raw_line"]),
                                  live_idx.get((k, t["raw_line"]), live_idx.get((k,))))
                if lu is None:
                    miss += 1
                n_anchor += 1
                ref = ref_idx.setdefault(str(t["simui"]), f"REF-{len(ref_idx) + 1:03d}")
                pub5.append({"line": f"{t['raw_series']}/{mask_line(t['raw_line'])}", "sim": t["sim"],
                             "n_chars": len(t["copy"]), "live_used": bool(lu), "ref": ref})
            top5 = pub5
            line = mask_line(line)
            rid = f"pub-{r['fmt']}-{n + 1:04d}"
        recs.append({
            "id": rid,
            "fmt": r["fmt"], "series": series, "line": line,
            "forme": r["forme"], "aspect": r["aspect"],
            "fields": payload,
            "band": r["band"], "sim": float(r["best_sim"]),
            "same5": int(r["same_line_top5"]),
            "transfer": bool(r["best_anchor_line"] != f"{r['series']}/{r['line']}"),
            "top5": top5,
        })
    if demo:
        print(f"public build: anchors {n_anchor} | live join miss {miss} | refs {len(ref_idx)}")
    return recs


HTML = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2330;--sub:#66707f;--line:#e3e7ee;
--blue:#2f6bff;--ok:#0f8a4d;--okbg:#e7f6ee;--near:#8a6d00;--nearbg:#fff7dc;
--mid:#555e6d;--midbg:#eef1f5;--low:#b3361f;--lowbg:#fdecea}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;background:var(--bg);color:var(--ink)}
header{background:#fff;border-bottom:1px solid var(--line);padding:14px 22px}
h1{font-size:17px;margin:0 0 8px}
.notice{background:#fff7dc;border:1px solid #f0dc96;border-radius:8px;padding:9px 12px;font-size:12.5px;color:#6b5600}
.notice b{color:#8a6d00}
.bar{display:flex;flex-wrap:wrap;gap:8px;padding:12px 22px;align-items:center}
select,input[type=search]{padding:7px 9px;border:1px solid var(--line);border-radius:7px;font-size:13px;background:#fff}
input[type=search]{min-width:200px;flex:1}
.stats{padding:0 22px 6px;font-size:12.5px;color:var(--sub)}
main{padding:6px 22px 40px;max-width:1060px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:13px 15px;margin-bottom:10px}
.meta{font-size:11.5px;color:var(--sub);display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
.tag{background:var(--midbg);border-radius:5px;padding:2px 7px}
.flds{margin:2px 0 9px}
.fld{display:flex;gap:8px;font-size:14px;line-height:1.55}
.fld .k{color:var(--sub);font-size:11.5px;min-width:62px;padding-top:2px}
.badges{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px}
.band{border-radius:6px;padding:3px 9px;font-weight:600}
.b0{background:var(--okbg);color:var(--ok)}.b1{background:var(--nearbg);color:var(--near)}
.b2{background:var(--midbg);color:var(--mid)}.b3{background:var(--lowbg);color:var(--low)}
.warn{color:var(--low)}
button.cp{margin-left:auto;border:1px solid var(--line);background:#fff;border-radius:7px;padding:5px 11px;font-size:12px;cursor:pointer}
button.cp:hover{border-color:var(--blue);color:var(--blue)}
details{margin-top:8px;font-size:12.5px}summary{cursor:pointer;color:var(--sub)}
.anc{border-left:3px solid var(--line);padding:5px 10px;margin:6px 0;color:#39414e}
.anc .s{color:var(--sub);font-size:11.5px}
.pager{display:flex;gap:8px;justify-content:center;padding:18px;align-items:center;font-size:13px}
.pager button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:6px 12px;cursor:pointer}
.pager button:disabled{opacity:.4;cursor:default}
.empty{text-align:center;color:var(--sub);padding:50px 0}
</style></head><body>
<header>
<h1>__TITLE__</h1>
<div class="notice">본 카탈로그의 밴드·근접도는 <b>심의 통과 이력과의 근접도와 리스크 신호</b>이며, <b>심의 통과를 보장하지 않습니다.</b> 최종 판단은 심의 절차를 따릅니다. 밴드 임계는 실집행 이미지 카피 분포 기준 — 브랜드검색·스페셜DA 형식은 참고 지표로 해석.</div>
</header>
<div class="bar">
<select id="fFmt"><option value="">소재: 전체</option></select>
<select id="fLine"><option value="">라인: 전체</option></select>
<select id="fForme"><option value="">유형: 전체</option></select>
<select id="fAspect"><option value="">소구: 전체</option></select>
<select id="fBand"><option value="">밴드: 전체</option></select>
<input id="fQ" type="search" placeholder="문안 검색">
</div>
<div class="stats" id="stats"></div>
<main id="list"></main>
<div class="pager"><button id="prev">이전</button><span id="pinfo"></span><button id="next">다음</button></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const FMT=__FMTLABEL__,FLD=__FLDLABEL__,PUB=__PUBLIC__;
const BANDS=["승인동등","실집행근접","분포내","신중"];
const PAGE=50;let page=0;
const $=id=>document.getElementById(id);
function opts(sel,vals,pre){vals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=(pre?pre(v):v);sel.appendChild(o)})}
opts($('fFmt'),Object.keys(FMT),v=>FMT[v]);
opts($('fLine'),[...new Set(D.map(r=>r.series+'/'+r.line))].sort());
opts($('fForme'),[...new Set(D.map(r=>r.forme))].sort());
opts($('fAspect'),[...new Set(D.map(r=>r.aspect))].sort());
opts($('fBand'),BANDS);
function filt(){const f=$('fFmt').value,l=$('fLine').value,fo=$('fForme').value,a=$('fAspect').value,b=$('fBand').value,q=$('fQ').value.trim().replace(/\s+/g,'');
return D.filter(r=>(!f||r.fmt===f)&&(!l||r.series+'/'+r.line===l)&&(!fo||r.forme===fo)&&(!a||r.aspect===a)&&(!b||r.band===b)&&(!q||Object.values(r.fields).join(' ').replace(/\s+/g,'').includes(q)))}
function copyText(r){return Object.entries(r.fields).map(([k,v])=>(FLD[k]||k)+': '+v).join('\\n')}
function card(r){const bi=BANDS.indexOf(r.band);
const flds=Object.entries(r.fields).map(([k,v])=>`<div class="fld"><span class="k">${FLD[k]||k}</span><span>${v}</span></div>`).join('');
const anc=r.top5.map(t=>PUB?`<div class="anc"><div class="s">${t.line} · ${(t.sim*100).toFixed(0)}% · ${t.n_chars}자 · ${t.live_used?'실집행':'심의필'} · ${t.ref}</div></div>`:`<div class="anc">"${t.copy}"<div class="s">${t.line} · ${(t.sim*100).toFixed(0)}% · ${t.pdf} · 심의 ${t.simui}</div></div>`).join('');
return `<div class="card"><div class="meta"><span class="tag">${FMT[r.fmt]}</span><span class="tag">${r.series}/${r.line}</span><span class="tag">${r.forme}</span><span class="tag">${r.aspect}</span></div>
<div class="flds">${flds}</div>
<div class="badges"><span class="band b${bi}">${r.band}</span><span>근접도 ${(r.sim*100).toFixed(0)}%</span><span>동일라인 앵커 ${r.same5}/5</span>${r.transfer?'<span class="warn">최근접 앵커 타라인</span>':''}<button class="cp" data-id="${r.id}">복사</button></div>
<details><summary>최근접 심의필 앵커 Top5${PUB?' — 원문 비공개, 속성만':''}</summary>${anc}</details></div>`}
function render(){const rows=filt();const np=Math.max(1,Math.ceil(rows.length/PAGE));if(page>=np)page=np-1;
const slice=rows.slice(page*PAGE,(page+1)*PAGE);
$('list').innerHTML=slice.length?slice.map(card).join(''):'<div class="empty">조건에 맞는 카피 없음</div>';
$('stats').textContent=`${rows.length}개 / 전체 ${D.length}개 · 승인동등 ${rows.filter(r=>r.band==='승인동등').length} · 실집행근접 ${rows.filter(r=>r.band==='실집행근접').length}`;
$('pinfo').textContent=`${page+1} / ${np}`;$('prev').disabled=page===0;$('next').disabled=page>=np-1;
document.querySelectorAll('.cp').forEach(b=>b.onclick=()=>{const r=rows.find(x=>x.id===b.dataset.id);
navigator.clipboard.writeText(copyText(r)).then(()=>{b.textContent='복사됨';setTimeout(()=>b.textContent='복사',900)})})}
['fFmt','fLine','fForme','fAspect','fBand'].forEach(id=>$(id).onchange=()=>{page=0;render()});
$('fQ').oninput=()=>{page=0;render()};
$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--public", dest="demo", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(JUDGE)
    print(f"판정 데이터: {len(df)}행")
    if args.demo:
        parts = [g.head(DEMO_N.get(f, 20)) for f, g in df.groupby("fmt")]
        df = pd.concat(parts).reset_index(drop=True)
        print(f"데모 표본: {len(df)}행 (마스킹 적용)")
    recs = to_records(df, demo=args.demo)
    data = json.dumps(recs, ensure_ascii=False, separators=(",", ":"))
    title = ("Copy Catalog v1 — public build (client copy text withheld)" if args.demo
             else f"{BRAND} 카피 카탈로그 v1.1 — {len(df):,} 사전 판정")
    html = (HTML.replace("__TITLE__", title)
                .replace("__FMTLABEL__", json.dumps(FMT_LABEL, ensure_ascii=False))
                .replace("__FLDLABEL__", json.dumps(FIELD_LABEL, ensure_ascii=False))
                .replace("__PUBLIC__", "true" if args.demo else "false")
                .replace("__DATA__", data))
    size_mb = len(html.encode()) / 1024 / 1024
    print(f"HTML 크기: {size_mb:.2f}MB (벡터 미내장)")
    dest = OUT_DEMO if args.demo else OUT
    if not args.write:
        print(f"dry-run — 빌드하려면 --write{' --public' if args.demo else ''} 실행 (대상: {dest})")
        return
    if dest.exists():
        print(f"중단: {dest} 이미 존재 (덮어쓰기 거부)")
        return
    dest.write_text(html, encoding="utf-8")
    print(f"저장: {dest}")
    print("빌드 후 검증: 브라우저 열기 + 필터/검색/복사 동작 확인")


if __name__ == "__main__":
    main()