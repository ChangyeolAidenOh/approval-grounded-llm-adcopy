import sys, time, hashlib
from pathlib import Path
import pandas as pd
import urllib.request

MANIFEST = Path('data/copy/live_assets/cre_manifest.parquet')
IMG_DIR = Path('data/copy/live_assets/imgs')

def fname_for(url):
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    ext = url.split('?')[0].rsplit('.', 1)[-1].lower()
    return f'{h}.{ext}'

def head_size(url):
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=15) as r:
        return int(r.headers.get('Content-Length', 0))

def main():
    confirm = '--confirm' in sys.argv
    df = pd.read_parquet(MANIFEST)
    urls = df['img_url'].tolist()
    print(f'대상: {len(urls)}장')
    if not confirm:
        sample = urls[::max(1, len(urls) // 30)][:30]
        sizes = []
        for u in sample:
            try:
                sizes.append(head_size(u))
            except Exception as e:
                print('HEAD 실패:', u[:60], type(e).__name__)
        if sizes:
            avg = sum(sizes) / len(sizes)
            print(f'표본 {len(sizes)}장 평균 {avg/1024:.0f}KB → 총 추정 {avg*len(urls)/1024/1024:.0f}MB')
        print('dry-run — 다운로드는 --confirm 단독 실행')
        return
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    done = skip = fail = 0
    fails = []
    for i, u in enumerate(urls, 1):
        dest = IMG_DIR / fname_for(u)
        if dest.exists() and dest.stat().st_size > 0:
            skip += 1
            continue
        try:
            urllib.request.urlretrieve(u, dest)
            done += 1
            time.sleep(0.05)
        except Exception as e:
            fail += 1
            fails.append((u, type(e).__name__))
            if dest.exists():
                dest.unlink()
        if i % 100 == 0:
            print(f'{i}/{len(urls)} (신규 {done} / 스킵 {skip} / 실패 {fail})')
    print(f'완료: 신규 {done} / 스킵 {skip} / 실패 {fail}')
    for u, err in fails[:10]:
        print('실패:', err, u[:80])

if __name__ == '__main__':
    main()