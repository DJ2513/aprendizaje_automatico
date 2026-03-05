import os
import base64
import io
import json
import tempfile

import numpy as np
from PIL import Image
from openai import OpenAI

VISION_MODEL = "gpt-4o"
GENERATE_MODEL = "gpt-4o-mini"
VIDEO_FRAMES = 5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

VISION_SYSTEM_PROMPT = (
    "You are a cinematographic analyst specialized in movie recommendations. "
    "Your task is to analyze images and describe them in terms useful for finding similar movies. "
    "Focus on: visual tone (dark/light/vibrant), genre cues (thriller/romance/action/sci-fi), "
    "setting and atmosphere (urban night, wilderness, futuristic, historical), "
    "emotional mood (tense, melancholic, joyful, mysterious), "
    "and any narrative elements visible (conflict, intimacy, adventure). "
    "Write 3-5 sentences. Be specific and cinematographic. "
    "Do not mention brand names, actors, or specific movie titles you recognize."
)


def encode_image(source, max_size: int = 1024) -> str:
    if isinstance(source, bytes):
        img = Image.open(io.BytesIO(source)).convert("RGB")
    elif isinstance(source, str):
        img = Image.open(source).convert("RGB")
    elif isinstance(source, np.ndarray):
        import cv2
        img = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    else:
        img = source.convert("RGB")

    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_frames(video_bytes: bytes, n_frames: int = VIDEO_FRAMES):
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        if total == 0:
            cap.release()
            raise ValueError("Could not read video: no frames found.")

        duration_sec = total / fps
        start = int(total * 0.05)
        end = int(total * 0.95)
        positions = np.linspace(start, end, n_frames, dtype=int)

        frames = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
    finally:
        os.unlink(tmp_path)

    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    return frames, duration_sec


def describe_image(image_b64: str, extra_context: str = "") -> str:
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "low"},
        },
        {
            "type": "text",
            "text": (
                "Analyze this image cinematographically for movie recommendation purposes."
                + (f" Additional context from user: {extra_context}" if extra_context else "")
            ),
        },
    ]
    resp = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def describe_multiple_frames(frames_b64: list, extra_context: str = "") -> str:
    user_content = []
    for i, b64 in enumerate(frames_b64):
        user_content.append({"type": "text", "text": f"Frame {i + 1} of {len(frames_b64)}:"})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })

    user_content.append({
        "type": "text",
        "text": (
            f"These are {len(frames_b64)} frames extracted at regular intervals from a video clip. "
            "Analyze the visual progression and describe the overall cinematographic style, "
            "tone, genre, atmosphere, and narrative for movie recommendation purposes. "
            "Consider how the visuals evolve across the frames."
            + (f" Additional user context: {extra_context}" if extra_context else "")
        ),
    })

    resp = _client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=400,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def rerank_and_generate(
    query_description: str,
    candidates: list,
    top_n: int = 5,
    query_type: str = "visual",
) -> dict:
    candidates_text = ""
    for i, c in enumerate(candidates):
        candidates_text += (
            f"{i + 1}. ID={c['id']} | {c['title']} ({c['year']}) | "
            f"Genres: {c['genres']}\n"
            f"   Overview: {str(c['overview'])[:200]}\n\n"
        )

    if query_type == "text":
        system_rerank = (
            "You are a movie recommendation expert. "
            "The user described what they want to watch in text. "
            "Rerank the candidates selecting the best matches for their request. "
            f"Select exactly {top_n} movies. "
            'Return ONLY a valid JSON array: [{"id": "movie_id", "reason": "one sentence why"}, ...]'
        )
        user_rerank = (
            f"User query:\n{query_description}\n\n"
            f"Candidates:\n{candidates_text}"
            f"Return JSON with the best {top_n} movies."
        )
        system_gen = (
            "You are a friendly movie recommender. "
            "The user described what they want to watch. "
            "Write a warm response in English explaining why these movies are a great match for what they asked for. "
            "Keep it under 300 words."
        )
        user_gen = f"User query:\n{query_description}\n\nSelected movies:\n{{context}}"
    else:
        system_rerank = (
            "You are a movie recommendation expert. "
            "The user provided a visual description (from an image or video) as their query. "
            "Rerank the candidates selecting the best visual and tonal matches. "
            f"Select exactly {top_n} movies. "
            'Return ONLY a valid JSON array: [{"id": "movie_id", "reason": "one sentence why"}, ...]'
        )
        user_rerank = (
            f"Visual query description:\n{query_description}\n\n"
            f"Candidates:\n{candidates_text}"
            f"Return JSON with the best {top_n} movies."
        )
        system_gen = (
            "You are a friendly movie recommender. "
            "The user submitted a visual input (image or video) and you analyzed it. "
            "Write a warm response in English explaining what you saw visually and why these movies match that vibe. "
            "Keep it under 300 words."
        )
        user_gen = f"Visual analysis of the submitted media:\n{query_description}\n\nSelected movies:\n{{context}}"

    try:
        resp = _client.chat.completions.create(
            model=GENERATE_MODEL,
            messages=[
                {"role": "system", "content": system_rerank},
                {"role": "user", "content": user_rerank},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        reranked_data = json.loads(raw)
    except Exception:
        reranked_data = [{"id": c["id"], "reason": ""} for c in candidates[:top_n]]

    candidates_by_id = {c["id"]: c for c in candidates}
    reranked = []
    for item in reranked_data:
        mid = str(item.get("id", ""))
        if mid in candidates_by_id:
            movie = dict(candidates_by_id[mid])
            movie["reason"] = item.get("reason", "")
            reranked.append(movie)
    reranked = reranked[:top_n]

    context = ""
    for i, movie in enumerate(reranked, 1):
        context += (
            f"{i}. {movie['title']} ({movie['year']})\n"
            f"   Overview: {str(movie['overview'])[:250]}\n"
            f"   Why it fits: {movie['reason']}\n\n"
        )

    gen_resp = _client.chat.completions.create(
        model=GENERATE_MODEL,
        messages=[
            {"role": "system", "content": system_gen},
            {"role": "user", "content": user_gen.format(context=context)},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    return {
        "response": gen_resp.choices[0].message.content.strip(),
        "movies": reranked,
        "description": query_description,
    }
