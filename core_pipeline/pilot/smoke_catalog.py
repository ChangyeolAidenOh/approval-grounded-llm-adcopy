"""Browser smoke test for the static copy catalog builds (v1 and public).

Drives the real page in headless Chromium via Playwright. Expected values are
recomputed in Python from the embedded JSON, independently of the page's own
filter code. Clipboard writes are captured through a stubbed
navigator.clipboard (file:// pages have no clipboard permission), so this
verifies the text the button sends, not the OS clipboard.
Read-only; --shot writes a screenshot to /tmp.
"""
import argparse
import json
import random
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

BANDS = ["승인동등", "실집행근접", "분포내", "신중"]
PAGE = 50
SEARCH_TERMS = ["한팩", "한 팩", "1팩", "한 병", "한병", "포 돌파", "xq9z없음"]
PUBLIC_TOP5_KEYS = {"line", "sim", "n_chars", "live_used", "ref"}
CLIP_STUB = ("Object.defineProperty(navigator,'clipboard',{value:{writeText:t=>{window.__copied=t;"
             "return Promise.resolve();}},configurable:true});")
BATCH_JS = r"""(combos) => combos.map(c => {
  $('fFmt').value=c[0]; $('fLine').value=c[1]; $('fForme').value=c[2]; $('fAspect').value=c[3]; $('fBand').value=c[4]; $('fQ').value=c[5];
  page=0; render();
  const m=$('stats').textContent.match(/^(\d+)개/);
  return [parseInt(m[1]), document.querySelectorAll('#list .card').length, $('pinfo').textContent];
})"""
RESET_JS = "() => {['fFmt','fLine','fForme','fAspect','fBand'].forEach(i=>$(i).value='');$('fQ').value='';page=0;render();}"

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def ws(s):
    return re.sub(r"\s+", "", str(s))


def load_data(path: Path):
    html = path.read_text(encoding="utf-8")
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1))


def py_filter(D, f, l, fo, a, b, q):
    qn = ws(q.strip())
    out = []
    for r in D:
        if f and r["fmt"] != f:
            continue
        if l and f"{r['series']}/{r['line']}" != l:
            continue
        if fo and r["forme"] != fo:
            continue
        if a and r["aspect"] != a:
            continue
        if b and r["band"] != b:
            continue
        if qn and qn not in ws(" ".join(str(v) for v in r["fields"].values())):
            continue
        out.append(r)
    return out


def py_copy_text(r, FLD):
    return "\n".join(f"{FLD.get(k, k)}: {v}" for k, v in r["fields"].items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shot", action="store_true")
    args = ap.parse_args()
    path = Path(args.path)
    D = load_data(path)
    byid = {r["id"]: r for r in D}
    rng = random.Random(args.seed)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        ctx.add_init_script(CLIP_STUB)
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(path.resolve().as_uri())
        page.wait_for_selector("#list .card")

        n_page = page.evaluate("() => D.length")
        check("records loaded", n_page == len(D), f"page {n_page} / json {len(D)}")
        FLD = page.evaluate("() => FLD")
        PUB = page.evaluate("() => typeof PUB==='undefined'?null:PUB")
        print("build:", "public" if PUB else ("v1" if PUB is False else "unpatched"))

        hdr = page.locator("header")
        box = hdr.bounding_box()
        txt = hdr.inner_text()
        check("header visible at top", box is not None and box["y"] <= 2 and txt.strip(),
              f"y={box['y'] if box else None} text={txt[:160]!r}")

        fmts = sorted({r["fmt"] for r in D})
        lines = sorted({f"{r['series']}/{r['line']}" for r in D})
        formes = sorted({r["forme"] for r in D})
        aspects = sorted({r["aspect"] for r in D})
        combos = [(f, l, "", "", b, "") for f in [""] + fmts for l in [""] + lines for b in [""] + BANDS]
        combos += [("", "", fo, a, "", "") for fo in [""] + formes for a in [""] + aspects]
        for _ in range(args.random):
            combos.append((rng.choice([""] * 2 + fmts), rng.choice([""] * 2 + lines), rng.choice([""] * 2 + formes),
                           rng.choice([""] * 2 + aspects), rng.choice([""] * 2 + BANDS), rng.choice([""] * 3 + SEARCH_TERMS)))
        combos += [("", "", "", "", "", q) for q in SEARCH_TERMS]
        got = page.evaluate(BATCH_JS, [list(c) for c in combos])
        bad = []
        for c, (n_stat, n_cards, pinfo) in zip(combos, got):
            exp = len(py_filter(D, *c))
            exp_np = max(1, -(-exp // PAGE))
            if n_stat != exp or n_cards != min(exp, PAGE) or pinfo != f"1 / {exp_np}":
                bad.append((c, exp, n_stat, n_cards, pinfo))
        check(f"filter combos ({len(combos)}) match python", not bad, f"mismatch {len(bad)}" + (f" first {bad[0]}" if bad else ""))
        for q in SEARCH_TERMS:
            print(f"  search {q!r}: {len(py_filter(D, '', '', '', '', '', q))}")
        page.evaluate(RESET_JS)

        total = len(D)
        np_ = max(1, -(-total // PAGE))
        prev0 = page.is_disabled("#prev")
        seen = page.locator("#list .card").count()
        clicks = 0
        while not page.is_disabled("#next") and clicks <= np_ + 2:
            page.click("#next")
            clicks += 1
            seen += page.locator("#list .card").count()
        check("pagination walk", clicks == np_ - 1 and seen == total and prev0 and page.inner_text("#pinfo") == f"{np_} / {np_}",
              f"clicks {clicks} seen {seen}/{total} pinfo {page.inner_text('#pinfo')!r}")
        page.select_option("#fBand", "승인동등")
        check("page resets on filter change", page.inner_text("#pinfo").startswith("1 /"), page.inner_text("#pinfo"))
        page.select_option("#fBand", "")

        ids = page.evaluate("() => [...document.querySelectorAll('#list .cp')].map(b=>b.dataset.id)")
        sample = rng.sample(ids, min(5, len(ids)))
        copy_bad, fld_bad = [], []
        for rid in sample:
            btn = page.locator(f'#list .cp[data-id="{rid}"]')
            btn.click()
            copied = page.evaluate("() => window.__copied")
            label = btn.inner_text()
            if copied != py_copy_text(byid[rid], FLD) or label != "복사됨":
                copy_bad.append((rid, label, (copied or "")[:40]))
            rendered = page.evaluate(
                "(id) => {const b=document.querySelector(`#list .cp[data-id=\"${id}\"]`);const card=b.closest('.card');"
                "return [...card.querySelectorAll('.fld')].map(d=>[d.querySelector('.k').textContent,d.querySelectorAll('span')[1].textContent]);}",
                rid)
            exp = [[FLD.get(k, k), str(v)] for k, v in byid[rid]["fields"].items()]
            if rendered != exp:
                fld_bad.append((rid, rendered[:1], exp[:1]))
        check("copy button sends label: value lines", not copy_bad, str(copy_bad[:1]))
        check("card fields render == json", not fld_bad, str(fld_bad[:1]))

        over = [r["id"] for r in D if len(r["top5"]) > 5]
        check("top5 <= 5 (json)", not over, str(over[:3]))
        anc_max = page.evaluate("() => Math.max(...[...document.querySelectorAll('#list .card')].map(c=>c.querySelectorAll('.anc').length))")
        check("top5 <= 5 (rendered)", anc_max <= 5, str(anc_max))
        badge_bad = page.evaluate(
            "(B) => [...document.querySelectorAll('#list .card')].filter(c=>{const e=c.querySelector('.band');"
            "return !e.classList.contains('b'+B.indexOf(e.textContent))}).length", BANDS)
        check("band badge class matches band", badge_bad == 0, str(badge_bad))

        if PUB:
            keys_bad = [r["id"] for r in D for t in r["top5"] if set(t) != PUBLIC_TOP5_KEYS]
            check("public top5 keys whitelist", not keys_bad, str(keys_bad[:3]))
            textnodes = page.evaluate(
                "() => [...document.querySelectorAll('#list .anc')].filter(a=>[...a.childNodes].some(n=>n.nodeType===3&&n.textContent.trim())).length")
            check("public anchors carry no copy text", textnodes == 0, str(textnodes))
            summ = page.evaluate("() => document.querySelector('#list details summary').textContent")
            check("public summary marks withheld", "원문 비공개" in summ, summ)
            live = sum(t["live_used"] for r in D for t in r["top5"])
            print(f"  public anchors live_used {live} / {sum(len(r['top5']) for r in D)}")

        page.fill("#fQ", "xq9z없음")
        check("empty state message", page.inner_text("#list").strip() == "조건에 맞는 카피 없음", page.inner_text("#list")[:40])
        page.fill("#fQ", "")

        check("no page errors", not errors, str(errors[:2]))
        if args.shot:
            out = f"/tmp/smoke_{path.stem}.png"
            page.screenshot(path=out)
            print("screenshot", out)
        browser.close()

    fails = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(fails)}/{len(results)} passed" + (f" | FAIL: {fails}" if fails else ""))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
