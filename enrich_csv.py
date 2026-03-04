import pandas as pd
import re

IN_CSV = "top_300_imdb_movies.csv"
OUT_CSV = "top_300_imdb_movies_enriched.csv"

GENRE_ALIASES = {
    "action": ["accion", "acción", "action"],
    "adventure": ["aventura", "adventure"],
    "animation": ["animacion", "animación", "animation"],
    "biography": ["biografia", "biografía", "biography"],
    "comedy": ["comedia", "comedy"],
    "crime": ["crimen", "crime"],
    "drama": ["drama"],
    "family": ["familiar", "family"],
    "fantasy": ["fantasia", "fantasía", "fantasy"],
    "history": ["historica", "histórica", "history"],
    "horror": ["terror", "horror"],
    "music": ["musica", "música", "music"],
    "mystery": ["misterio", "mystery"],
    "romance": ["romance", "romantica", "romántica"],
    "sci-fi": ["ciencia ficcion", "ciencia ficción", "sci-fi", "science fiction"],
    "thriller": ["thriller", "suspenso", "suspense"],
    "war": ["guerra", "war"],
    "western": ["western", "vaqueros"],
}

MOOD_KEYWORDS = {
    "dark": ["oscura", "oscuro", "dark", "sombrio", "siniestro", "bleak"],
    "uplifting": ["inspiradora", "uplifting", "esperanzadora", "feel-good", "conmovedora"],
    "tense": ["tensa", "tenso", "tension", "tense", "edge", "ansiedad"],
    "funny": ["divertida", "graciosa", "chistosa", "funny", "hilarante"],
    "emotional": ["emocional", "tearjerker", "conmovedora", "heartbreaking"],
    "mind-bending": ["mind-bending", "twist", "giro", "surreal", "paranoia"],
}

THEME_KEYWORDS = {
    "revenge": ["venganza", "revenge"],
    "redemption": ["redencion", "redención", "redemption"],
    "coming-of-age": ["coming of age", "madurar", "adolescencia"],
    "friendship": ["amistad", "friendship"],
    "love": ["amor", "love"],
    "war": ["guerra", "war", "soldado", "battle"],
    "crime": ["mafia", "crimen", "crime", "heist", "robo"],
    "space": ["espacio", "space", "astronaut", "galaxy"],
    "time": ["tiempo", "time travel", "viaje en el tiempo", "timeline"],
    "justice": ["justicia", "justice", "court", "trial", "abogado", "judge"],
}

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    return t

def detect_tags_from_overview(overview: str):
    o = (overview or "").lower()
    moods = []
    themes = []
    for mood, kws in MOOD_KEYWORDS.items():
        if any(k.lower() in o for k in kws):
            moods.append(mood)
    for theme, kws in THEME_KEYWORDS.items():
        if any(k.lower() in o for k in kws):
            themes.append(theme)
    return moods[:4], themes[:6]

def build_doc_text(row) -> str:
    title = normalize(row.get("Title", ""))
    year = row.get("Year", "")
    genres = normalize(row.get("Genres", ""))
    director = normalize(row.get("Director", ""))
    overview = normalize(row.get("Overview", ""))

    moods, themes = detect_tags_from_overview(overview)

    # “Repetición suave” de señales fuertes (título, géneros, director) ayuda a embeddings sin meter ruido.
    parts = [
        f"Title: {title}",
        f"Year: {year}",
        f"Genres: {genres}",
        f"Director: {director}",
    ]
    if moods:
        parts.append("Mood: " + ", ".join(moods))
    if themes:
        parts.append("Themes: " + ", ".join(themes))

    parts.append(f"Overview: {overview}")

    # Boost ligero: repetir géneros una vez ayuda a queries por género
    if genres:
        parts.append(f"Keywords: {genres}")

    return "\n".join([p for p in parts if p and str(p).strip() and str(p).lower() != "nan"])

def main():
    df = pd.read_csv(IN_CSV)
    df["doc_text"] = df.apply(build_doc_text, axis=1)
    df.to_csv(OUT_CSV, index=False)
    print(f"Guardado: {OUT_CSV} (con columna doc_text)")

if __name__ == "__main__":
    main()
