import sys, re
from pathlib import Path
import pandas as pd

MATCHES = Path('data/copy/live_assets/live_copy_matches.parquet')
VERIFIED = Path('data/copy/live_assets/live_match_review_sample_verified.csv')
REG = Path('data/copy/registry/copy_registry.parquet') if Path('data/copy/registry/copy_registry.parquet').exists() else Path('data/copy/copy_registry.parquet')
OUT_REG = Path('data/copy/live_assets/copy_registry_live_v1.parquet')
OUT_B = Path('data/copy/live_assets/live_only_copies.parquet')

def norm(s):
    return re.sub(r'[^가-힣a-z0-9]', '', str(s).lower())

def bigram_dice(a, b):
    a, b = norm(a), norm(b)
    A = {a[i:i+2] for i in range(len(a)-1)}
    B = {b[i:i+2] for i in range(len(b)-1)}
    return 2*len(A & B)/(len(A)+len(B)) if A and B else 0.0

def char_sim(text, anchor):
    cands = [str(anchor)] + [p.strip() for p in str(anchor).split('/')]
    return max(bigram_dice(text, c) for c in cands)

def is_A(sem, cs):
    return (sem >= 0.85 and cs >= 0.70) or cs >= 0.80

def hangul_n(s):
    return len(re.findall(r'[가-힣]', str(s)))

def main():
    write = '--write' in sys.argv
    m = pd.read_parquet(MATCHES)
    u = (m.sort_values('conf', ascending=False)
          .groupby('text', as_index=False)
          .agg(sem=('match_score', 'max'), conf_max=('conf', 'max'),
               n_imgs=('img_file', 'nunique'), anchor_num=('anchor_num', 'first'),
               anchor=('anchor_full_copy', 'first'), anchor_line=('anchor_line', 'first'),
               media=('media', lambda s: '|'.join(sorted(set(s)))),
               promo=('promo', lambda s: '|'.join(sorted(set(s))[:6])),
               item=('item', 'first'), is_on=('is_on', lambda s: s.astype(str).max())))
    u['char_sim'] = [char_sim(t, a) for t, a in zip(u['text'], u['anchor'])]
    u['cls'] = ['A' if is_A(s, c) else 'B' for s, c in zip(u['sem'], u['char_sim'])]
    DISCLAIM_END = ('바랍니다', '바랍니다.', '하십시오', '하십시오.', '십시오.', '있습니다', '있습니다.')
    DISCLAIM_IN = ('주의사항', '상담하', '지도해', '복용', '의약품', '첨가물', '단순 스펙', '섭취 시', '밀봉')

    def is_noise_b(text, conf, sem):
        t = str(text).strip()
        if conf < 0.5 or hangul_n(t) < 6 or sem < 0.55:
            return True
        if t.endswith(DISCLAIM_END) or any(c in t for c in DISCLAIM_IN):
            return True
        if ' ' not in t and len(t) <= 8:
            return True
        if t.startswith('*'):
            return True
        return False

    u.loc[(u['cls'] == 'B') & u.apply(lambda r: is_noise_b(r['text'], r['conf_max'], r['sem']), axis=1), 'cls'] = 'C'
    print(f'전체 유니크 문안 {len(u)}: ', u['cls'].value_counts().to_dict())
    v = pd.read_csv(VERIFIED)
    v['char_sim'] = [char_sim(t, a) for t, a in zip(v['text'], v['anchor'])]
    v['pred'] = ['A' if is_A(s, c) else 'B' for s, c in zip(v['match_score'], v['char_sim'])]
    agree_A = int(((v['verdict'] == 'A') & (v['pred'] == 'A')).sum())
    fp = int(((v['verdict'] != 'A') & (v['pred'] == 'A')).sum())
    print(f'표본 재검증: A 적중 {agree_A}/{int((v["verdict"]=="A").sum())}, 오탑재 {fp} (기대 7/7, 0)')
    A, B = u[u['cls'] == 'A'], u[u['cls'] == 'B']
    print(f'\nA(실사용 registry 카피): 문안 {len(A)} / 앵커 {A["anchor_num"].nunique()}종 / 이미지 {int(A["n_imgs"].sum())}장')
    print(f'B(실집행 registry 밖 문안): {len(B)} | 매체 분포:', B['media'].str.split('|').explode().value_counts().to_dict())
    print('\n[캘리브레이션] 실집행 문안의 앵커 근접도 분포 (안전도 도구 임계 입력):')
    for name, g in (('A', A), ('B', B)):
        q = g['sem'].quantile([.1, .25, .5, .75, .9]).round(3)
        print(f'  {name}: p10={q[.1]} p25={q[.25]} p50={q[.5]} p75={q[.75]} p90={q[.9]}')
    print(f'  → 제안 임계: 근접도 ≥{B["sem"].quantile(.25):.2f} = "실집행 분포 내" (B p25), ≥{B["sem"].quantile(.75):.2f} = "근접" (B p75)')
    print('\n[소재 형식 실측] 매체별 카피 행 프로파일 (A∪B 행 기준):')
    cp = m[m['text'].isin(u.loc[u['cls'].isin(['A', 'B']), 'text'])]
    for media, g in cp.groupby('media'):
        per_img = g.groupby('img_file').size()
        chars = g['text'].str.len()
        print(f'  {media}: 이미지 {g["img_file"].nunique()}장 | 행/이미지 p50={per_img.median():.0f} p90={per_img.quantile(.9):.0f} | 행 글자수 p25={chars.quantile(.25):.0f} p50={chars.median():.0f} p90={chars.quantile(.9):.0f}')
    print('\nB 버킷 상위 노출 표본 12 (순도 눈검수):')
    for _, r in B.nlargest(12, 'n_imgs').iterrows():
        print(f'  [{r["n_imgs"]}장|{r["media"]}] {r["text"][:48]}')
    if write:
        for p in (OUT_REG, OUT_B):
            if p.exists():
                print(f'중단: {p} 이미 존재')
                return
        reg = pd.read_parquet(REG)
        flag = A.groupby('anchor_num').agg(n_live_imgs=('n_imgs', 'sum'), live_media=('media', lambda s: '|'.join(sorted(set('|'.join(s).split('|'))))), best_live_sem=('sem', 'max'))
        out = reg.merge(flag, left_on='num', right_index=True, how='left')
        out['live_used'] = out['num'].isin(flag.index)
        out.to_parquet(OUT_REG, index=False)
        B.to_parquet(OUT_B, index=False)
        print(f'\n저장: {OUT_REG} ({len(out)}행, live_used {int(out["live_used"].sum())}종)')
        print(f'저장: {OUT_B} ({len(B)}행)')
    else:
        print('\ndry-run — 저장은 --write 단독 실행')

if __name__ == '__main__':
    main()