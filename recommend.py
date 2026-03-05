import os
import re
import numpy as np
from openai import OpenAI

client = OpenAI(os.environ("OPENAI_API_KEY"))
MODEL = "text-embedding-3-small"
INDEX_PATH = "movie_index.npz"
PROFILE_PATH = "profiles/user_profile.npy"

GENRES = [
    "Action","Adventure","Animation","Biography","Comedy","Crime","Drama","Family","Fantasy",
    "History","Horror","Music","Mystery","Romance","Sci-Fi","Thriller","War","Western"
]

SPANISH_GENRE_MAP = {
    "accion":"Action","acción":"Action",
    "aventura":"Adventure",
    "animacion":"Animation","animación":"Animation",
    "biografia":"Biography","biografía":"Biography",
    "comedia":"Comedy",
    "crimen":"Crime",
    "drama":"Drama",
    "familiar":"Family",
    "fantasia":"Fantasy","fantasía":"Fantasy",
    "historia":"History","historica":"History","histórica":"History",
    "terror":"Horror",
    "musica":"Music","música":"Music",
    "misterio":"Mystery",
    "romance":"Romance","romantica":"Romance","romántica":"Romance",
    "ciencia ficcion":"Sci-Fi","ciencia ficción":"Sci-Fi",
    "suspenso":"Thriller",
    "guerra":"War",
    "vaqueros":"Western",
}

def embed_one(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=MODEL, input=[text])
    return np.array(resp.data[0].embedding, dtype=np.float32)

def normalize_rows(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)

def cosine_scores(q: np.ndarray, M_norm: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-12)
    return M_norm @ q

def parse_filters(query: str):
    q = query.lower()

    # décadas en español
    year_min, year_max = None, None
    if re.search(r"(noventas|años\s*90|90s|1990s)", q):
        year_min, year_max = 1990, 1999
    elif re.search(r"(ochentas|años\s*80|80s|1980s)", q):
        year_min, year_max = 1980, 1989
    elif re.search(r"(setentas|años\s*70|70s|1970s)", q):
        year_min, year_max = 1970, 1979
    elif re.search(r"(dosmil|años\s*2000|2000s)", q):
        year_min, year_max = 2000, 2009
    elif re.search(r"(dosmil\s*10|2010s|años\s*2010)", q):
        year_min, year_max = 2010, 2019

    # año específico
    m = re.search(r"\b(19\d{2}|20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        year_min, year_max = y, y

    # género (inglés y español)
    wanted_genres = set()
    for g in GENRES:
        if g.lower() in q:
            wanted_genres.add(g)
    for sp, eng in SPANISH_GENRE_MAP.items():
        if sp in q:
            wanted_genres.add(eng)

    return year_min, year_max, list(wanted_genres)

def apply_filters(candidate_idx, years, genres, year_min, year_max, wanted_genres):
    keep = []
    for i in candidate_idx:
        ok = True
        y = years[i]
        try:
            y_int = int(str(y))
        except:
            y_int = None

        if year_min is not None and y_int is not None:
            if not (year_min <= y_int <= year_max):
                ok = False

        if wanted_genres:
            gstr = str(genres[i])
            if not any(g in gstr for g in wanted_genres):
                ok = False

        if ok:
            keep.append(i)
    return np.array(keep, dtype=int)

def mmr_select(q, M_norm, candidates, k=10, lam=0.75):
    # MMR: maximiza relevancia al query y minimiza redundancia
    selected = []
    cand = list(candidates)

    q_scores = cosine_scores(q, M_norm[cand])  # scores sobre candidatos
    q_score_map = {cand[i]: float(q_scores[i]) for i in range(len(cand))}

    while cand and len(selected) < k:
        if not selected:
            best = max(cand, key=lambda idx: q_score_map[idx])
            selected.append(best)
            cand.remove(best)
            continue

        def score_mmr(idx):
            rel = q_score_map[idx]
            # máxima similitud con ya seleccionadas
            sims = M_norm[selected] @ (M_norm[idx] / (np.linalg.norm(M_norm[idx]) + 1e-12))
            red = float(np.max(sims)) if len(sims) else 0.0
            return lam * rel - (1 - lam) * red

        best = max(cand, key=score_mmr)
        selected.append(best)
        cand.remove(best)

    return selected

def load_profile(dim: int):
    if os.path.exists(PROFILE_PATH):
        p = np.load(PROFILE_PATH)
        if p.shape == (dim,):
            return p.astype(np.float32)
    return None

def save_profile(p: np.ndarray):
    np.save(PROFILE_PATH, p.astype(np.float32))

def main():
    idx = np.load(INDEX_PATH, allow_pickle=True)
    M = idx["embeddings"].astype(np.float32)
    M_norm = normalize_rows(M)

    titles = idx["titles"]
    years = idx["years"]
    genres = idx["genres"]

    dim = M.shape[1]
    profile = load_profile(dim)

    print(" Recomendador listo.")
    print("Comandos: like N | dislike N | reset | exit")
    print("Ejemplo query: 'thriller oscuro noventas' o 'ciencia ficción emocional'")

    last_results = None  # lista de índices globales

    while True:
        query = input("\n Qué quieres ver? > ").strip()
        if not query:
            continue

        if query.lower() == "exit":
            break

        if query.lower() == "reset":
            if os.path.exists(PROFILE_PATH):
                os.remove(PROFILE_PATH)
            profile = None
            print(" Perfil borrado.")
            continue

        m = re.match(r"^(like|dislike)\s+(\d+)$", query.lower())
        if m:
            if not last_results:
                print(" No hay resultados previos. Haz una búsqueda primero.")
                continue
            action = m.group(1)
            n = int(m.group(2))
            if n < 1 or n > len(last_results):
                print(" Número fuera de rango.")
                continue

            chosen_idx = last_results[n - 1]
            vec = M[chosen_idx]

            if profile is None:
                profile = np.zeros((dim,), dtype=np.float32)

            if action == "like":
                profile = profile + vec
                print(f" Like guardado: {titles[chosen_idx]}")
            else:
                profile = profile - vec
                print(f" Dislike guardado: {titles[chosen_idx]}")

            # normaliza perfil para estabilidad
            norm = np.linalg.norm(profile) + 1e-12
            profile = (profile / norm).astype(np.float32)
            save_profile(profile)
            continue

        # ---- búsqueda normal ----
        qv = embed_one(query)

        # mezcla con perfil (si existe)
        if profile is not None:
            qv = (0.70 * profile + 0.30 * qv).astype(np.float32)

        # recuperación inicial
        scores = cosine_scores(qv, M_norm)
        top50 = scores.argsort()[-50:][::-1]

        year_min, year_max, wanted_genres = parse_filters(query)
        filtered = apply_filters(top50, years, genres, year_min, year_max, wanted_genres)
        if len(filtered) < 10:
            # si filtró demasiado, usa top50 sin filtros
            filtered = top50

        # diversidad con MMR
        selected = mmr_select(qv, M_norm, filtered, k=10, lam=0.78)
        last_results = selected

        # imprime
        if year_min is not None or wanted_genres:
            print(f"\n Filtros detectados: "
                  f"{'año/decada=' + (str(year_min) if year_min==year_max else f'{year_min}-{year_max}') if year_min else ''} "
                  f"{'géneros=' + ','.join(wanted_genres) if wanted_genres else ''}".strip())

        print("\n🎬 Recomendaciones (usa 'like N' / 'dislike N'):")
        for r, i in enumerate(selected, 1):
            y = str(years[i]) if str(years[i]).lower() != "nan" else ""
            g = str(genres[i]) if str(genres[i]).lower() != "nan" else ""
            print(f"{r:02d}. {titles[i]} ({y}) | {g} | score={scores[i]:.3f}")

if __name__ == "__main__":
    main()
