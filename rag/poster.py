import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import requests

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"

# Local cache files
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
POSTER_CACHE_PATH = CACHE_DIR / "poster_cache.json"
CONFIG_CACHE_PATH = CACHE_DIR / "tmdb_config.json"

# Cache TTLs
POSTER_CACHE_TTL_SECONDS = 60 * 60 * 24 * 30   # 30 days
CONFIG_CACHE_TTL_SECONDS = 60 * 60 * 24 * 7    # 7 days

DEFAULT_POSTER_SIZE = "w342"  # good UI size, common TMDB size


def _now() -> int:
    return int(time.time())


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_api_key():
    if not TMDB_API_KEY:
        raise RuntimeError(
            "TMDB_API_KEY is not set. Export it in your environment before running."
        )


def get_tmdb_config() -> Dict[str, Any]:
    """
    Get TMDB configuration (image base_url + available sizes).
    TMDB recommends using /configuration to build image URLs.  :contentReference[oaicite:1]{index=1}
    """
    _ensure_api_key()

    cache = _load_json(CONFIG_CACHE_PATH)
    ts = cache.get("_ts", 0)
    if cache and (_now() - ts) < CONFIG_CACHE_TTL_SECONDS:
        return cache

    r = requests.get(
        f"{TMDB_BASE}/configuration",
        params={"api_key": TMDB_API_KEY},
        timeout=10,
    )
    r.raise_for_status()
    cfg = r.json()
    cfg["_ts"] = _now()
    _save_json(CONFIG_CACHE_PATH, cfg)
    return cfg


def _build_image_url(file_path: str, size: str = DEFAULT_POSTER_SIZE) -> Optional[str]:
    if not file_path:
        return None
    cfg = get_tmdb_config()
    images = cfg.get("images", {})
    base_url = images.get("secure_base_url") or images.get("base_url")
    if not base_url:
        return None

    poster_sizes = images.get("poster_sizes") or []
    if poster_sizes and size not in poster_sizes:
        # fallback to a safe available size
        size = poster_sizes[-1]

    # base_url already ends with /
    return f"{base_url}{size}{file_path}"


def _poster_cache_key(title: str, year: Optional[str]) -> str:
    t = (title or "").strip().lower()
    y = (str(year).strip() if year is not None else "")
    return f"{t}::{y}"


def _search_movie_tmdb(title: str, year: Optional[str]) -> Optional[Dict[str, Any]]:
    _ensure_api_key()
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": "false",
        "language": "en-US",
    }
    # Only pass year if it looks like a year
    if year and str(year).isdigit():
        params["year"] = str(year)

    r = requests.get(f"{TMDB_BASE}/search/movie", params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []
    if not results:
        return None

    # best match is usually first; you can improve with custom scoring if needed
    return results[0]


def get_poster_url(title: str, year: Optional[str] = None, size: str = DEFAULT_POSTER_SIZE) -> Optional[str]:
    """
    Return a full poster URL for a movie title/year using TMDB search, with caching.
    """
    if not title:
        return None
    _ensure_api_key()

    cache = _load_json(POSTER_CACHE_PATH)
    key = _poster_cache_key(title, year)

    # Cache record format: { url: str|None, ts: int }
    rec = cache.get(key)
    if rec and (_now() - rec.get("ts", 0)) < POSTER_CACHE_TTL_SECONDS:
        return rec.get("url")

    movie = _search_movie_tmdb(title, year)
    poster_path = (movie or {}).get("poster_path")
    url = _build_image_url(poster_path, size=size) if poster_path else None

    cache[key] = {"url": url, "ts": _now()}
    _save_json(POSTER_CACHE_PATH, cache)
    return url