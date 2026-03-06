# Movie Recommender — RAG + Embeddings

A movie recommendation system that combines semantic search, user profiling, and multimodal AI. It supports both **text queries** and **visual inputs** (images and videos) to find matching movies from a curated dataset of the top 300 IMDb films.

---

## How It Works — High Level

```
                        ┌─────────────────────────┐
                        │   Data Preparation       │
                        │   (contructor.py)        │
                        │                          │
                        │  CSV → Embed → .npz      │
                        └──────────┬──────────────┘
                                   │ movie_index.npz
                    ┌──────────────┴──────────────┐
                    │                             │
           Text Query Flow               Visual Query Flow
                    │                             │
           User types query            User uploads image/video
                    │                             │
           OpenAI Embedding            GPT-4o Vision describes it
                    │                             │
           Cosine Similarity           OpenAI Embedding
                    │                             │
           Filters + MMR              Cosine Similarity + MMR
                    │                             │
           GPT-4o-mini reranks        GPT-4o-mini reranks
                    │                             │
           Movie cards + posters + reasons (Streamlit UI)
```

---

## Project Structure

```
embeding/
├── app.py              # Streamlit UI — main entry point
├── contructor.py       # Builds the vector index from the CSV
├── recommend.py        # Legacy CLI recommender (standalone)
├── enrich_csv.py       # CSV enrichment utilities
├── movie_index.npz     # Pre-built vector store (embeddings + metadata)
├── data/
│   └── top_300_imdb_movies.csv
├── profiles/           # Saved user profile vectors
└── rag/
    ├── embeddings.py   # OpenAI embedding wrapper
    ├── index.py        # Loads .npz and computes cosine scores
    ├── filters.py      # Decade and genre filter parsing
    ├── recommend.py    # Core recommendation logic (blend + MMR)
    ├── profile.py      # Profile vector blending
    ├── vision.py       # GPT-4o Vision + frame extraction + reranking
    └── poster.py       # Movie poster fetching
```

---

## Step 0 — Data Preparation (`contructor.py`)

Before anything runs, the movie database must be converted into a vector index.

1. Reads `data/top_300_imdb_movies.csv`
2. For each movie, builds a text document combining title, year, genres, director, IMDb rating, and overview
3. Sends all documents to **OpenAI `text-embedding-3-small`** in batches of 128
4. Saves the resulting 1536-dimensional vectors alongside movie metadata into `movie_index.npz`

This step only needs to run once. The resulting `.npz` file is loaded at startup by the app.

---

## Text Query Flow

This is the main search mode. The user types a natural language description of what they want to watch — in English or Spanish.

### 1. Embedding the Query (`rag/embeddings.py`)
The query string is sent to OpenAI's `text-embedding-3-small` model, which returns a 1536-dimensional vector representing its semantic meaning.

### 2. Profile Blending (`rag/profile.py`)
If the user has liked or disliked movies in the session, a **profile vector** is maintained as a running sum of those movie embeddings. The final search vector is a weighted blend:

```
search_vector = 70% profile + 30% query
```

This makes results increasingly personalized as the user interacts.

### 3. Cosine Similarity (`rag/index.py`)
The search vector is compared against all 300 movie embeddings using cosine similarity. The top 50 candidates are retrieved.

### 4. Filtering (`rag/filters.py`)
The query text is parsed for decade and genre mentions — in both English and Spanish (e.g., "noventas", "terror", "80s", "sci-fi"). If filters are detected and enough results pass them, the candidate pool is narrowed down.

### 5. MMR Diversification (`rag/recommend.py`)
**Maximal Marginal Relevance (MMR)** re-orders the candidates to balance relevance against redundancy. This prevents the results from being too similar to each other.

```
MMR score = λ × relevance_to_query − (1 − λ) × max_similarity_to_already_selected
```

Default λ is 0.78 (adjustable in the UI).

### 6. GPT-4o-mini Reranking & Generation (`rag/vision.py`)
The top candidates are sent to **GPT-4o-mini**, which:
- Reranks them based on how well they match the user's request
- Returns a one-sentence reason for each pick
- Generates a warm, conversational recommendation response

### 7. Display (`app.py`)
Results are shown as movie cards with:
- Poster image (fetched from TMDB via `rag/poster.py`)
- Title, year, genre, director, score
- "Why it fits" explanation from GPT
- Like / Dislike / Clear buttons to update the profile

---

## Visual Query Flow

The user uploads an image (JPG, PNG, WEBP) or a short video clip (MP4, MOV, etc.).

### 1. Media Processing (`rag/vision.py`)
- **Image:** Resized and encoded as base64 JPEG
- **Video:** 5 frames are extracted at evenly spaced intervals (skipping the first and last 5% to avoid black frames), then each frame is encoded

### 2. GPT-4o Vision Description
The encoded frames are sent to **GPT-4o** with a cinematographic analysis prompt. It describes the visual tone, genre cues, atmosphere, emotional mood, and narrative elements — without naming specific movies or actors. This produces a rich text description of the media's cinematic style.

### 3. Embedding + Search
The visual description is embedded with `text-embedding-3-small` and searched against the vector index using the same cosine similarity pipeline — without profile blending or filters.

### 4. GPT-4o-mini Reranking & Generation
Same as the text flow: candidates are reranked, reasons are generated, and a response is written explaining what the AI saw and why the movies match.

---

## User Profile System

The user profile is a session-level vector that adapts recommendations over time.

| Action | Effect |
|---|---|
| Like a movie | Its embedding vector is added to the profile |
| Dislike a movie | Its embedding vector is subtracted |
| Clear a rating | The delta is reversed |

After each change the profile vector is L2-normalized for stability. On the next search, it is blended with the query vector (70%/30% by default, adjustable via the UI slider).

Clicking **Reset profile** zeroes the vector and clears all feedback for the session.

---

## Key Technologies

| Component | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-3-small` |
| Vision analysis | OpenAI `gpt-4o` |
| Reranking & generation | OpenAI `gpt-4o-mini` |
| Similarity search | NumPy cosine similarity |
| Diversity | Maximal Marginal Relevance (MMR) |
| UI | Streamlit |
| Movie posters | TMDB API |
| Video frame extraction | OpenCV |

---

## Running the App

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

# 3. Build the vector index (only once)
python contructor.py

# 4. Launch the app
streamlit run app.py
```
