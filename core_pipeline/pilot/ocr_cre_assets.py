import sys, time, hashlib
from pathlib import Path
import pandas as pd
import Quartz
import Vision
from Foundation import NSURL

MANIFEST = Path('data/copy/live_assets/cre_manifest.parquet')
IMG_DIR = Path('data/copy/live_assets/imgs')
OUT = Path('data/copy/live_assets/ocr_lines.parquet')

def fname_for(url):
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    ext = url.split('?')[0].rsplit('.', 1)[-1].lower()
    return f'{h}.{ext}'

def ocr_image(path):
    nsurl = NSURL.fileURLWithPath_(str(path))
    src = Quartz.CGImageSourceCreateWithURL(nsurl, None)
    if src is None:
        return None
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if img is None:
        return None
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    req.setRecognitionLanguages_(['ko-KR', 'en-US'])
    ok = handler.performRequests_error_([req], None)
    if isinstance(ok, tuple):
        ok = ok[0]
    if not ok:
        return None
    lines = []
    for obs in (req.results() or []):
        cands = obs.topCandidates_(1)
        if cands and len(cands):
            box = obs.boundingBox()
            lines.append({
                'text': str(cands[0].string()),
                'conf': float(cands[0].confidence()),
                'y': round(1.0 - float(box.origin.y) - float(box.size.height), 4),
                'x': round(float(box.origin.x), 4),
                'rel_h': round(float(box.size.height), 4),
            })
    lines.sort(key=lambda l: (round(l['y'], 2), l['x']))
    return lines

def main():
    full = '--all' in sys.argv
    df = pd.read_parquet(MANIFEST)
    df['img_file'] = df['img_url'].map(fname_for)
    present = df['img_file'].map(lambda f: (IMG_DIR / f).exists())
    print(f'manifest: {len(df)}행 | 파일 존재: {int(present.sum())} | 누락: {int((~present).sum())}')
    if (~present).any():
        for f in df.loc[~present, 'img_file'].head(5):
            print('  누락:', f)
    work = df[present].reset_index(drop=True)
    if not full:
        sample = pd.concat([
            work[work['media'] == 'GFA'].head(10),
            work[work['media'] == '메타'].head(5),
        ])
        print(f'--- 표본 {len(sample)}장 품질 미리보기 ---')
        for _, r in sample.iterrows():
            lines = ocr_image(IMG_DIR / r['img_file'])
            n = len(lines) if lines else 0
            print(f"\n[{r['media']}/{r['item']}/{r['promo']}] {r['creative_name'][:50]}")
            if lines:
                for l in lines[:8]:
                    print(f"  ({l['conf']:.2f}) {l['text'][:60]}")
                if n > 8:
                    print(f'  ... 외 {n-8}행')
            else:
                print('  (추출 실패 또는 텍스트 없음)')
        print('\ndry-run — 전수 처리는 --all 단독 실행')
        return
    if OUT.exists():
        print(f'중단: {OUT} 이미 존재 (덮어쓰기 거부)')
        return
    rows = []
    t0 = time.time()
    n_fail = 0
    for i, r in work.iterrows():
        lines = ocr_image(IMG_DIR / r['img_file'])
        if lines is None:
            n_fail += 1
            continue
        for seq, l in enumerate(lines):
            rows.append({
                'img_file': r['img_file'], 'creative_id': r['creative_id'],
                'creative_name': r['creative_name'], 'media': r['media'],
                'channel': r['channel'], 'promo': r['promo'], 'item': r['item'],
                'is_on': r['is_on'], 'line_seq': seq, **l,
            })
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f'{i+1}/{len(work)} ({el:.0f}s, 실패 {n_fail})')
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    print(f'저장: {OUT} | 이미지 {len(work)-n_fail}/{len(work)} 처리 | 텍스트 {len(out)}행 | {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()