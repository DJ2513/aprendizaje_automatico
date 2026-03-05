import numpy as np
from .index import cosine_scores
from .filters import parse_filters, apply_filters
from .profile import blend

def mmr_select(q, M_norm, candidates, k=10, lam=0.78):
    selected = []
    cand = list(map(int, candidates))

    q_scores = cosine_scores(q, M_norm[cand])
    q_score_map = {cand[i]: float(q_scores[i]) for i in range(len(cand))}

    while cand and len(selected) < k:
        if not selected:
            best = max(cand, key=lambda idx: q_score_map[idx])
            selected.append(best)
            cand.remove(best)
            continue

        def score_mmr(idx):
            rel = q_score_map[idx]
            sims = M_norm[selected] @ (M_norm[idx] / (np.linalg.norm(M_norm[idx]) + 1e-12))
            red = float(np.max(sims)) if len(sims) else 0.0
            return lam * rel - (1 - lam) * red

        best = max(cand, key=score_mmr)
        selected.append(best)
        cand.remove(best)

    return selected

def recommend(query_vec, query_text, data, topk=10, mmr_lambda=0.78, use_filters=True, profile=None, mix=0.70):
    M_norm = data["M_norm"]
    years = data["years"]
    genres = data["genres"]

    q = blend(profile, query_vec, mix=mix)

    scores = cosine_scores(q, M_norm)
    top50 = scores.argsort()[-50:][::-1]

    filters_info = None
    candidates = top50

    if use_filters and years is not None and genres is not None:
        year_min, year_max, wanted_genres = parse_filters(query_text)
        filtered = apply_filters(top50, years, genres, year_min, year_max, wanted_genres)
        if len(filtered) >= topk:
            candidates = filtered
            filters_info = (year_min, year_max, wanted_genres)

    selected = mmr_select(q, M_norm, candidates, k=topk, lam=mmr_lambda)
    return selected, scores, filters_info