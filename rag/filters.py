import re
import numpy as np

def parse_filters(query: str):
    q = query.lower()

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

    m = re.search(r"\b(19\d{2}|20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        year_min, year_max = y, y

    genre_map = {
        "Action": ["action", "accion", "acción"],
        "Adventure": ["adventure", "aventura"],
        "Animation": ["animation", "animacion", "animación"],
        "Biography": ["biography", "biografia", "biografía"],
        "Comedy": ["comedy", "comedia"],
        "Crime": ["crime", "crimen"],
        "Drama": ["drama"],
        "Family": ["family", "familiar"],
        "Fantasy": ["fantasy", "fantasia", "fantasía"],
        "History": ["history", "historica", "histórica", "historia"],
        "Horror": ["horror", "terror"],
        "Music": ["music", "musica", "música"],
        "Mystery": ["mystery", "misterio"],
        "Romance": ["romance", "romantica", "romántica"],
        "Sci-Fi": ["sci-fi", "science fiction", "ciencia ficcion", "ciencia ficción"],
        "Thriller": ["thriller", "suspenso", "suspense"],
        "War": ["war", "guerra"],
        "Western": ["western", "vaqueros"],
    }

    wanted = [g for g, kws in genre_map.items() if any(kw in q for kw in kws)]
    return year_min, year_max, wanted

def apply_filters(candidate_idx, years, genres, year_min, year_max, wanted_genres):
    keep = []
    for i in candidate_idx:
        ok = True

        y = years[i] if years is not None else None
        try:
            y_int = int(str(y))
        except:
            y_int = None

        if year_min is not None and y_int is not None:
            if not (year_min <= y_int <= year_max):
                ok = False

        if wanted_genres and genres is not None:
            gstr = str(genres[i])
            if not any(g in gstr for g in wanted_genres):
                ok = False

        if ok:
            keep.append(int(i))

    return np.array(keep, dtype=int)