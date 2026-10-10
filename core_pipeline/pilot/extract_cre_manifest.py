import sys, re, json, glob
from pathlib import Path
import pandas as pd

SRC = Path('data/copy/live_reports')
OUT = Path('data/copy/live_assets')

def load_embedded(path):
    html = open(path, encoding='utf-8', errors='replace').read()
    m = re.search(r'<script[^>]*id="data"[^>]*>(.*?)</script>', html, re.S)
    return json.loads(m.group(1)) if m else None

def main():
    write = '--write' in sys.argv
    rows = []
    for f in sorted(SRC.glob('*.html')):
        data = load_embedded(f)
        if not data or 'cre' not in data:
            print(f'skip (cre 없음): {f.name}')
            continue
        latest = data.get('meta', {}).get('latest', '')
        for r in data['cre']:
            imgs = r.get('imgs', '[]')
            if isinstance(imgs, str):
                imgs = json.loads(imgs.replace("'", '"'))
            for i, url in enumerate(imgs):
                rows.append({
                    'source_report': f.name, 'report_latest': latest,
                    'creative_id': str(r.get('소재ID', '')), 'creative_name': r.get('소재명', ''),
                    'channel': r.get('유통채널', ''), 'media': r.get('매체', ''),
                    'promo': r.get('기획전', ''), 'item': r.get('품목', ''),
                    'campaign': r.get('캠페인', ''), 'adgroup': r.get('광고그룹', ''),
                    'is_on': str(r.get('on', '')), 'img_idx': i, 'img_url': url,
                })
    df = pd.DataFrame(rows)
    uniq = df.drop_duplicates(subset=['img_url']).reset_index(drop=True)
    print(f'레코드 전체: {len(df)} | 유니크 이미지 URL: {len(uniq)}')
    print(f'유니크 소재ID: {uniq["creative_id"].nunique()}')
    print('매체 분포:', uniq['media'].value_counts().to_dict())
    print('품목 분포:', uniq['item'].value_counts().to_dict())
    print('기획전 상위:', uniq['promo'].value_counts().head(12).to_dict())
    ext = uniq['img_url'].str.extract(r'\.(\w+)(?:\?|$)')[0].str.lower()
    print('확장자 분포:', ext.value_counts().to_dict())
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        dest = OUT / 'cre_manifest.parquet'
        if dest.exists():
            print(f'중단: {dest} 이미 존재 (덮어쓰기 거부)')
            return
        uniq.to_parquet(dest, index=False)
        print(f'저장: {dest} ({len(uniq)}행)')
    else:
        print('dry-run — 저장하려면 --write 단독 실행')

if __name__ == '__main__':
    main()