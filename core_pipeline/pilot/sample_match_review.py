import sys
from pathlib import Path
import pandas as pd

SRC = Path('data/copy/live_assets/live_copy_matches.parquet')
OUT = Path('data/copy/live_assets/live_match_review_sample.csv')
BANDS = [(0.85, 1.01, 12), (0.75, 0.85, 12), (0.65, 0.75, 10), (0.50, 0.65, 6)]

def main():
    write = '--write' in sys.argv
    df = pd.read_parquet(SRC)
    u = (df.sort_values('match_score', ascending=False)
           .groupby('text', as_index=False)
           .agg(match_score=('match_score', 'max'), n_imgs=('img_file', 'nunique'),
                media=('media', 'first'), promo=('promo', 'first'), item=('item', 'first'),
                anchor=('anchor_full_copy', 'first'), anchor_line=('anchor_line', 'first')))
    print(f'유니크 문안: {len(u)}')
    parts = []
    for lo, hi, n in BANDS:
        band = u[(u['match_score'] >= lo) & (u['match_score'] < hi)]
        take = band.sample(min(n, len(band)), random_state=20260813)
        take = take.assign(band=f'{lo:.2f}~')
        parts.append(take)
        print(f'구간 {lo:.2f}~{hi if hi<=1 else 1.0:.2f}: 모수 {len(band)} → 표본 {len(take)}')
    s = pd.concat(parts).sort_values('match_score', ascending=False).reset_index(drop=True)
    s['verdict'] = ''
    cols = ['band', 'match_score', 'text', 'anchor', 'anchor_line', 'media', 'promo', 'item', 'n_imgs', 'verdict']
    print()
    for _, r in s.iterrows():
        print(f"[{r['match_score']:.3f}] OCR: {r['text'][:42]}")
        print(f"        앵커({r['anchor_line']}): {str(r['anchor'])[:42]}")
    if write:
        if OUT.exists():
            print(f'중단: {OUT} 이미 존재')
            return
        s[cols].to_csv(OUT, index=False, encoding='utf-8-sig')
        print(f'저장: {OUT} ({len(s)}행)')
    else:
        print('\ndry-run — CSV 저장은 --write 단독 실행')

if __name__ == '__main__':
    main()