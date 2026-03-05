import os

import streamlit as st
from dotenv import load_dotenv
import numpy as np

load_dotenv()

from rag.index import load_index
from rag.embeddings import embed_one
from rag.recommend import recommend
from rag.poster import get_poster_url
from rag.vision import (
    encode_image,
    extract_frames,
    describe_image,
    describe_multiple_frames,
    rerank_and_generate,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)

INDEX_PATH = "movie_index.npz"

st.set_page_config(page_title="RAG PROJECT", layout="wide")
st.title("Movie Recommender")

st.markdown(
    """
    <style>
      div[data-testid="stHorizontalBlock"]{ gap:0.5rem; }
      div[data-testid="stButton"] > button { width:100%; }

      .btn-like div[data-testid="stButton"] > button {
        background: rgba(220, 38, 38, 0.18) !important;
        border: 1px solid rgba(220, 38, 38, 0.55) !important;
      }
      .btn-like div[data-testid="stButton"] > button:hover {
        border-color: rgba(220, 38, 38, 0.85) !important;
      }

      .btn-dislike div[data-testid="stButton"] > button {
        background: rgba(220, 38, 38, 0.18) !important;
        border: 1px solid rgba(220, 38, 38, 0.55) !important;
      }
      .btn-dislike div[data-testid="stButton"] > button:hover {
        border-color: rgba(220, 38, 38, 0.85) !important;
      }

      .btn-clear div[data-testid="stButton"] > button {
        background: rgba(107, 114, 128, 0.12) !important;
        border: 1px solid rgba(107, 114, 128, 0.45) !important;
      }

      div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding-top:0.75rem; padding-bottom:0.75rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_index():
    return load_index(INDEX_PATH)


data = cached_index()
M = data["M"]
titles = data["titles"]
years = data["years"]
genres = data["genres"]
directors = data["directors"]
overviews = data["overviews"]
dim = M.shape[1]

if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}
if "profile_raw" not in st.session_state:
    st.session_state["profile_raw"] = np.zeros((dim,), dtype=np.float32)
if "profile" not in st.session_state:
    st.session_state["profile"] = None
if "last_results" not in st.session_state:
    st.session_state["last_results"] = []
if "last_scores" not in st.session_state:
    st.session_state["last_scores"] = None
if "visual_results" not in st.session_state:
    st.session_state["visual_results"] = None
if "text_gpt_result" not in st.session_state:
    st.session_state["text_gpt_result"] = None


def normalize_profile():
    raw = st.session_state["profile_raw"]
    n = float(np.linalg.norm(raw))
    st.session_state["profile"] = None if n < 1e-8 else (raw / n).astype(np.float32)


def set_state(movie_key: str, new_state: int, vec: np.ndarray):
    prev = int(st.session_state["feedback"].get(movie_key, 0))
    if prev == new_state:
        return
    delta = new_state - prev
    st.session_state["profile_raw"] = (st.session_state["profile_raw"] + delta * vec).astype(np.float32)
    if new_state == 0:
        st.session_state["feedback"].pop(movie_key, None)
    else:
        st.session_state["feedback"][movie_key] = new_state
    normalize_profile()


def build_candidates(indices):
    candidates = []
    for i in indices:
        t = str(titles[i])
        y = str(years[i]) if years is not None and str(years[i]).lower() != "nan" else ""
        g = str(genres[i]) if genres is not None and str(genres[i]).lower() != "nan" else ""
        d = str(directors[i]) if directors is not None and str(directors[i]).lower() != "nan" else ""
        o = str(overviews[i]) if overviews is not None and str(overviews[i]).lower() != "nan" else ""
        candidates.append({"id": str(i), "title": t, "year": y, "genres": g, "director": d, "overview": o})
    return candidates


tab1, tab2 = st.tabs(["Text Search", "Visual Search"])


with tab1:
    with st.form("search_form", clear_on_submit=False):
        r1, r2 = st.columns([6, 1], vertical_alignment="bottom")
        with r1:
            query = st.text_input(
                "Describe what you want to watch",
                placeholder="e.g., thriller oscuro noventas / emotional sci-fi / crime drama 70s",
                label_visibility="collapsed",
            )
        with r2:
            do_search = st.form_submit_button("Recommend", use_container_width=True)

    with st.expander("Advanced controls"):
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.6, 2.0, 2.0, 1.2], vertical_alignment="bottom")
        with c1:
            topk = st.slider("Top K", 5, 20, 10, 1)
        with c2:
            mmr_lambda = st.slider("MMR λ", 0.50, 0.95, 0.78, 0.01)
        with c3:
            mix_profile = st.slider("Profile mix", 0.0, 1.0, 0.70, 0.05)
        with c4:
            use_filters = st.checkbox("Auto-filters (decade/genre)", True)
        with c5:
            if st.button("Reset profile", use_container_width=True):
                st.session_state["profile"] = None
                st.session_state["profile_raw"] = np.zeros((dim,), dtype=np.float32)
                st.session_state["feedback"] = {}
                st.success("Profile cleared.")

    if do_search and query.strip():
        with st.spinner("Searching..."):
            qv = embed_one(query.strip())
            selected, scores, filters_info = recommend(
                query_vec=qv,
                query_text=query,
                data=data,
                topk=max(topk, 15),
                mmr_lambda=mmr_lambda,
                use_filters=use_filters,
                profile=st.session_state["profile"],
                mix=mix_profile,
            )
            st.session_state["last_scores"] = scores

        if filters_info and (filters_info[0] is not None or filters_info[2]):
            year_min, year_max, wanted_genres = filters_info
            parts = []
            if year_min is not None:
                parts.append(f"years: {year_min}" if year_min == year_max else f"years: {year_min}-{year_max}")
            if wanted_genres:
                parts.append("genres: " + ", ".join(wanted_genres))
            st.info("Detected filters → " + " | ".join(parts))

        with st.spinner("Ranking and generating response with GPT-4o-mini..."):
            text_result = rerank_and_generate(query.strip(), build_candidates(selected), top_n=topk, query_type="text")
            st.session_state["last_results"] = [int(m["id"]) for m in text_result["movies"]]
            st.session_state["text_gpt_result"] = text_result

    if st.session_state["text_gpt_result"]:
        tr = st.session_state["text_gpt_result"]
        scores = st.session_state["last_scores"]

        st.markdown("---")
        st.write(tr["response"])
        st.markdown("---")

        st.subheader("Results")
        for rank, movie in enumerate(tr["movies"], 1):
            i = int(movie["id"])
            t, y, g = movie["title"], movie["year"], movie["genres"]
            d, o, reason = movie.get("director", ""), movie.get("overview", ""), movie.get("reason", "")
            s = float(scores[i]) if scores is not None else 0.0
            movie_key = str(i)

            poster_url = None
            try:
                poster_url = get_poster_url(t, y, size="w342")
            except Exception:
                pass

            with st.container(border=True):
                left, right = st.columns([220, 780], gap="large")

                with left:
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.caption("No poster found")

                    b1, b2, b3 = st.columns([1, 1, 1])
                    with b1:
                        st.markdown('<div class="btn-like">', unsafe_allow_html=True)
                        if st.button("👍", key=f"like_{i}_{rank}", use_container_width=True):
                            set_state(movie_key, 1, M[i])
                            st.toast(f"Liked: {t} ({y})", icon="👍")
                        st.markdown("</div>", unsafe_allow_html=True)
                    with b2:
                        st.markdown('<div class="btn-dislike">', unsafe_allow_html=True)
                        if st.button("👎", key=f"dislike_{i}_{rank}", use_container_width=True):
                            set_state(movie_key, -1, M[i])
                            st.toast(f"Disliked: {t} ({y})", icon="👎")
                        st.markdown("</div>", unsafe_allow_html=True)
                    with b3:
                        st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
                        if st.button("↩️", key=f"clear_{i}_{rank}", use_container_width=True):
                            set_state(movie_key, 0, M[i])
                            st.toast(f"Cleared: {t} ({y})", icon="↩️")
                        st.markdown("</div>", unsafe_allow_html=True)

                with right:
                    st.markdown(f"### {rank:02d}. {t} ({y})")
                    st.markdown(f"**Director:** {d if d else '—'}")
                    st.markdown(f"**Genre:** {g if g else '—'}")
                    st.markdown(f"**Score:** {s:.3f}")
                    if reason:
                        st.markdown(f"**Why it fits:** {reason}")
                    if o:
                        st.markdown("**Synopsis:**")
                        st.write(o)
    else:
        st.caption("Type a query and click Recommend.")


with tab2:
    st.markdown(
        "Upload an image or a short video clip and the system will analyze its cinematic "
        "style to find matching movies."
    )
    st.info(
        "**Video note:** Please upload short clips only (trailers or scenes), "
        "no longer than **3 minutes**. Longer videos may exceed the upload limit or "
        "take too long to process."
    )

    accepted_types = (
        [ext.lstrip(".") for ext in sorted(IMAGE_EXTENSIONS)]
        + [ext.lstrip(".") for ext in sorted(VIDEO_EXTENSIONS)]
    )

    with st.form("visual_form", clear_on_submit=False):
        uploaded_file = st.file_uploader(
            "Upload image or video",
            type=accepted_types,
            label_visibility="collapsed",
        )
        extra_context = st.text_input(
            "Extra context (optional)",
            placeholder="e.g., I want something horror / looking for romantic movies",
        )
        v1, v2 = st.columns([1.5, 1], vertical_alignment="bottom")
        with v1:
            visual_top_n = st.slider("Number of recommendations", 3, 10, 5, 1)
        with v2:
            do_visual = st.form_submit_button("Analyze & Recommend", use_container_width=True)

    if do_visual:
        if uploaded_file is None:
            st.warning("Please upload an image or video first.")
        else:
            file_bytes = uploaded_file.read()
            ext = os.path.splitext(uploaded_file.name)[1].lower()

            try:
                if ext in IMAGE_EXTENSIONS:
                    with st.spinner("Analyzing image with GPT-4o Vision..."):
                        description = describe_image(encode_image(file_bytes), extra_context=extra_context)

                elif ext in VIDEO_EXTENSIONS:
                    with st.spinner("Extracting frames from video..."):
                        frames, duration_sec = extract_frames(file_bytes)
                        st.caption(f"Video duration: {duration_sec:.1f}s — {len(frames)} frames extracted.")
                    with st.spinner("Analyzing frames with GPT-4o Vision..."):
                        description = describe_multiple_frames([encode_image(f) for f in frames], extra_context=extra_context)

                else:
                    st.error(f"Unsupported file type: {ext}")
                    st.stop()

                with st.spinner("Finding matching movies..."):
                    qv = embed_one(description)
                    selected, _, _ = recommend(
                        query_vec=qv,
                        query_text=description,
                        data=data,
                        topk=15,
                        mmr_lambda=0.60,
                        use_filters=False,
                        profile=None,
                        mix=0.0,
                    )

                with st.spinner("Reranking and generating response with GPT-4o-mini..."):
                    visual_result = rerank_and_generate(description, build_candidates(selected), top_n=visual_top_n)
                    st.session_state["visual_results"] = visual_result

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    if st.session_state["visual_results"]:
        vr = st.session_state["visual_results"]

        with st.expander("What GPT-4o saw in your media", expanded=True):
            st.write(vr["description"])

        st.markdown("---")
        st.write(vr["response"])
        st.markdown("---")

        st.subheader("Recommended Movies")
        for rank, movie in enumerate(vr["movies"], 1):
            t, y, g = movie["title"], movie["year"], movie["genres"]
            d, o, reason = movie.get("director", ""), movie.get("overview", ""), movie.get("reason", "")

            poster_url = None
            try:
                poster_url = get_poster_url(t, y, size="w342")
            except Exception:
                pass

            with st.container(border=True):
                left, right = st.columns([220, 780], gap="large")

                with left:
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.caption("No poster found")

                with right:
                    st.markdown(f"### {rank:02d}. {t} ({y})")
                    st.markdown(f"**Director:** {d if d else '—'}")
                    st.markdown(f"**Genre:** {g if g else '—'}")
                    if reason:
                        st.markdown(f"**Why it fits:** {reason}")
                    if o:
                        st.markdown("**Synopsis:**")
                        st.write(o)
    else:
        st.caption("Upload an image or video and click Analyze & Recommend.")
