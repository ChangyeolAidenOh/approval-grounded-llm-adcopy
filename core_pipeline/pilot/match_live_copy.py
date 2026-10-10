import sys, re, json, time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

OCR = Path('data/copy/live_assets/ocr_lines.parquet')
EMB = Path('data/copy/copy_embeddings.json')
REG = Path('data/copy/copy_registry.parquet')
OUT = Path('data/copy/live_assets/live_copy_matches.parquet')
OLLAMA = 'http://127.0.0.1:11434/api/embed'

NOISE_CUES = ['섭취량', '섭취방법', '섭취 방법', '원료명', '영양', '보관방법', '소비기한',
              '유통기한', '반품', '교환', '품목보고', '고객센터', '부작용', '알레르기',
              '질병의 예방', '의약품이 아', '수입판매원', '제조원', '원산지', '건강기능식품']

def hangul_n(s):
    return len(re.findall(r'[가-힣]', s))

def is_candidate(text, conf):
    t = text.strip()
    if conf < 0.3 or hangul_n(t) < 3:
        return False
    if any(c in t for c in NOISE_CUES):
        return False
    return True

def detect_fields(item):
    vec_key = next((k for k in ('embedding', 'vec', 'emb', 'vector') if k in item), None)
    id_key = next((k for k in ('num', 'id', 'copy_id', 'idx') if k in item), None)
    txt_key = next((k for k in ('full_copy', 'text', 'copy', 'anchor_text') if k in item), None)
    return vec_key, id_key, txt_key

def embed_texts(texts, chunk=64):
    vecs = []
    for i in range(0, len(texts), chunk):
        body = json.dumps({'model': 'bge-m3', 'input': texts[i:i+chunk]}).encode()
        req = urllib.request.Request(OLLAMA, data=body, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        vecs.extend(out['embeddings'])
        if (i // chunk) % 10 == 0:
            print(f'  embed {i+len(out["embeddings"])}/{len(texts)}')
    v = np.array(vecs, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

def main():
    full = '--all' in sys.argv
    df = pd.read_parquet(OCR)
    print(f'OCR 전체: {len(df)}행 / 이미지 {df["img_file"].nunique()}장')
    df['keep'] = [is_candidate(t, c) for t, c in zip(df['text'], df['conf'])]
    kept = df[df['keep']].copy()
    kept['text'] = kept['text'].str.strip()
    print(f'필터 통과: {len(kept)}행 | 유니크 문안: {kept["text"].nunique()}')
    print('매체별 통과 행:', kept['media'].value_counts().to_dict())
    emb = json.loads(open(EMB).read())
    item0 = emb['items'][0]
    vk, ik, tk = detect_fields(item0)
    print(f'embeddings 필드 탐지: vec={vk} id={ik} text={tk} | item keys={list(item0.keys())}')
    if not full:
        print('\n--- 통과 표본 15 ---')
        for t in kept['text'].drop_duplicates().head(15):
            print(' ', t[:60])
        print('--- 탈락 표본 10 ---')
        for t in df[~df['keep']]['text'].drop_duplicates().head(10):
            print(' ', t[:60])
        print('\ndry-run — 매칭 실행은 --all 단독 실행 (ollama serve 필요)')
        return
    if None in (vk, ik):
        print('중단: embeddings 필드 탐지 실패 — 실물 확인 필요')
        return
    if OUT.exists():
        print(f'중단: {OUT} 이미 존재 (덮어쓰기 거부)')
        return
    anchors = np.array([it[vk] for it in emb['items']], dtype=np.float32)
    anchors = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9)
    ids = [it[ik] for it in emb['items']]
    reg = pd.read_parquet(REG).set_index('num')
    texts = kept['text'].drop_duplicates().tolist()
    print(f'임베딩 대상 유니크 문안: {len(texts)}')
    t0 = time.time()
    q = embed_texts(texts)
    sims = q @ anchors.T
    best = sims.argmax(axis=1)
    lookup = {t: (ids[b], float(sims[i, b])) for i, (t, b) in enumerate(zip(texts, best))}
    kept['anchor_num'] = kept['text'].map(lambda t: lookup[t][0])
    kept['match_score'] = kept['text'].map(lambda t: lookup[t][1])
    for col in ('full_copy', 'line', 'product', 'simui', 'forme'):
        if col in reg.columns:
            kept[f'anchor_{col}'] = kept['anchor_num'].map(reg[col])
    kept.drop(columns=['keep']).to_parquet(OUT, index=False)
    s = kept['match_score']
    print(f'저장: {OUT} ({len(kept)}행, {time.time()-t0:.0f}s)')
    print(f'매칭 점수 분포: p50={s.median():.3f} p75={s.quantile(.75):.3f} p90={s.quantile(.9):.3f}')
    for th in (0.85, 0.75, 0.65):
        print(f'  ≥{th}: {int((s >= th).sum())}행 / 이미지 {kept.loc[s >= th, "img_file"].nunique()}장')
    hi = kept.nlargest(5, 'match_score')[['text', 'anchor_full_copy', 'match_score']]
    print('\n상위 매칭 예시:')
    for _, r in hi.iterrows():
        print(f'  {r["match_score"]:.3f} | OCR: {r["text"][:36]} | 앵커: {str(r["anchor_full_copy"])[:36]}')

if __name__ == '__main__':
    main()