import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="CineMate",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_models():

    movie_embeddings = joblib.load(
        "models/movie_embeddings.pkl"
    )

    user_factors = joblib.load(
        "models/user_factors.pkl"
    )

    movie_factors = joblib.load(
        "models/movie_factors.pkl"
    )

    user_movie_matrix = joblib.load(
        "models/user_movie_matrix.pkl"
    )

    movies_text = pd.read_pickle(
        "models/movies_text.pkl"
    )

    return (
        movie_embeddings,
        user_factors,
        movie_factors,
        user_movie_matrix,
        movies_text
    )


(
    movie_embeddings,
    user_factors,
    movie_factors,
    user_movie_matrix,
    movies_text
) = load_models()

@st.cache_resource
def get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("GEMINI_API_KEY was not found.")
        st.stop()

    return genai.Client(api_key=api_key)


client = get_gemini_client()

def extract_preferences(user_message):

    prompt = f"""
You are the preference extraction system for a movie recommendation app.

Analyze the user's movie request and return ONLY valid JSON.

Use exactly these fields:

{{
    "genres": [],
    "moods": [],
    "themes": [],
    "exclude": []
}}

Rules:
- genres: movie genres explicitly requested or strongly implied
- moods: adjectives describing the desired feeling or tone
- themes: story elements, topics, settings, or concepts the user wants
- exclude: genres, themes, or types of movies the user does not want
- If something is not specified, return an empty list.
- Do not recommend movies.
- Do not explain anything.

User request:
{user_message}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    return json.loads(text)

genre_map = {
    "Science Fiction": "Sci-Fi",
    "Sci Fi": "Sci-Fi",
    "Science-Fiction": "Sci-Fi",
    "Thriller": "Thriller",
    "Mystery": "Mystery",
    "Action": "Action",
    "Adventure": "Adventure",
    "Comedy": "Comedy",
    "Drama": "Drama",
    "Horror": "Horror",
    "Romance": "Romance",
    "Fantasy": "Fantasy",
    "Animation": "Animation",
    "Children": "Children",
    "Crime": "Crime",
    "Documentary": "Documentary",
    "Musical": "Musical",
    "War": "War",
    "Western": "Western",
    "Film-Noir": "Film-Noir",
    "(no genres listed)": None
}


def get_preferences(user_message):

    preferences = extract_preferences(user_message)

    preferences["genres"] = [
        genre_map[genre]
        for genre in preferences["genres"]
        if genre in genre_map
        and genre_map[genre] is not None
    ]

    return preferences

def filter_movies(preferences):

    candidates = movies_text.copy()

    # Require every requested genre
    for genre in preferences["genres"]:

        candidates = candidates[
            candidates["genres"].str.contains(
                genre,
                case=False,
                na=False
            )
        ]

    for excluded in preferences["exclude"]:

        candidates = candidates[
            ~candidates["text"].str.contains(
                excluded,
                case=False,
                na=False
            )
        ]

    return candidates

def rank_candidates(query, candidates, top_n=20):

    query_embedding = embedding_model.encode([query])

    candidate_indices = candidates.index

    candidate_embeddings = movie_embeddings[
        candidate_indices
    ]

    scores = cosine_similarity(
        query_embedding,
        candidate_embeddings
    )[0]

    results = candidates.copy()

    results["semantic_score"] = scores

    return (
        results
        .sort_values(
            "semantic_score",
            ascending=False
        )
        .head(top_n)
    )

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2"
    )


embedding_model = load_embedding_model()

movie_id_to_index = {
    movie_id: index
    for index, movie_id in enumerate(
        user_movie_matrix.columns
    )
}


def collaborative_scores(user_id, candidates):

    user_index = user_movie_matrix.index.get_loc(
        user_id
    )

    predicted_scores = (
        user_factors[user_index]
        @ movie_factors.T
    )

    predicted_scores = (
        predicted_scores - predicted_scores.min()
    ) / (
        predicted_scores.max()
        - predicted_scores.min()
    )

    scores = []

    for movie_id in candidates["movieId"]:

        if movie_id in movie_id_to_index:

            score = predicted_scores[
                movie_id_to_index[movie_id]
            ]

        else:
            score = 0

        scores.append(score)

    return np.array(scores)

def hybrid_rank(user_id, query, candidates, top_n=20):

    query_embedding = embedding_model.encode([query])

    candidate_indices = candidates.index

    candidate_embeddings = movie_embeddings[
        candidate_indices
    ]

    semantic_scores = cosine_similarity(
        query_embedding,
        candidate_embeddings
    )[0]

    collab_scores = collaborative_scores(
        user_id,
        candidates
    )

    final_scores = (
        0.75 * semantic_scores
        + 0.25 * collab_scores
    )

    results = candidates.copy()

    results["semantic_score"] = semantic_scores
    results["collaborative_score"] = collab_scores
    results["final_score"] = final_scores

    return (
        results
        .sort_values(
            "final_score",
            ascending=False
        )
        .head(top_n)
    )

movie_id_to_index = {
    movie_id: index
    for index, movie_id in enumerate(
        user_movie_matrix.columns
    )
}


def collaborative_scores(user_id, candidates):

    user_index = user_movie_matrix.index.get_loc(
        user_id
    )

    predicted_scores = (
        user_factors[user_index]
        @ movie_factors.T
    )

    predicted_scores = (
        predicted_scores - predicted_scores.min()
    ) / (
        predicted_scores.max()
        - predicted_scores.min()
    )

    scores = []

    for movie_id in candidates["movieId"]:

        if movie_id in movie_id_to_index:

            score = predicted_scores[
                movie_id_to_index[movie_id]
            ]

        else:
            score = 0

        scores.append(score)

    return np.array(scores)

def hybrid_rank(user_id, query, candidates, top_n=20):

    query_embedding = embedding_model.encode([query])

    candidate_indices = candidates.index

    candidate_embeddings = movie_embeddings[
        candidate_indices
    ]

    semantic_scores = cosine_similarity(
        query_embedding,
        candidate_embeddings
    )[0]

    collab_scores = collaborative_scores(
        user_id,
        candidates
    )

    final_scores = (
        0.75 * semantic_scores
        + 0.25 * collab_scores
    )

    results = candidates.copy()

    results["semantic_score"] = semantic_scores
    results["collaborative_score"] = collab_scores
    results["final_score"] = final_scores

    return (
        results
        .sort_values(
            "final_score",
            ascending=False
        )
        .head(top_n)
    )

def gemini_rerank(user_query, candidates):

    movie_list = []

    for _, row in candidates.iterrows():

        movie_tags = tags[
            tags["movieId"] == row["movieId"]
        ]["tag"].astype(str).tolist()

        movie_list.append({
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "genres": row["genres"],
            "tags": movie_tags
        })

    prompt = f"""
You are the final ranking system for a movie recommendation assistant.

User request:
"{user_query}"

Rank these candidate movies according to how well they match
the user's request.

Consider:
- requested genres
- mood and tone
- themes
- exclusions
- information contained in the MovieLens genres and tags

Grounding rules:
- Do NOT invent plot details.
- Do NOT claim a movie has a specific setting, character,
  plot twist, or story element unless supported by the
  provided genres or tags.
- Explicit exclusions must be respected.
- Tags are evidence, not requirements.
- A movie without tags can still be a good recommendation.

Return ONLY valid JSON in this exact format:

[
  {{
    "movieId": 123,
    "score": 95,
    "reason": "Short explanation based only on the provided information."
  }}
]

Rules:
- Include every candidate exactly once.
- score must be an integer from 0 to 100.
- Higher score means a better match.
- Keep each reason under 20 words.

Candidates:
{movie_list}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    return json.loads(text)

def recommend(user_id, user_query, top_n=10):

    preferences = get_preferences(user_query)

    candidates = filter_movies(preferences)

    if len(candidates) == 0:
        return None, preferences

    ranked_candidates = hybrid_rank(
        user_id,
        user_query,
        candidates,
        top_n=20
    )

    final_results = gemini_rerank(
        user_query,
        ranked_candidates
    )

    final_results = pd.DataFrame(final_results)

    final_results = final_results.merge(
        movies_text[
            ["movieId", "title", "genres"]
        ],
        on="movieId",
        how="left"
    )

    final_results = (
        final_results
        .sort_values(
            "score",
            ascending=False
        )
        .head(top_n)
    )

    return final_results, preferences

# =========================
# CineMate UI
# =========================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0b0b0f;
    }

    /* Main content width */
    .block-container {
        max-width: 1100px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* Main title */
    .cinemate-title {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: -2px;
        margin-bottom: 0;
    }

    .cinemate-subtitle {
        font-size: 1.2rem;
        color: #a7a7b0;
        margin-top: -5px;
        margin-bottom: 2.5rem;
    }

    /* Search section */
    .search-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .search-description {
        color: #9999a3;
        margin-bottom: 1rem;
    }

    /* Preference box */
    .preference-box {
        background: #15151c;
        border: 1px solid #292933;
        border-radius: 16px;
        padding: 20px 24px;
        margin: 20px 0 30px 0;
    }

    .preference-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .preference-item {
        display: inline-block;
        background: #20202a;
        border-radius: 20px;
        padding: 7px 13px;
        margin: 4px 5px 4px 0;
        color: #d8d8df;
        font-size: 0.9rem;
    }

    /* Movie card */
    .movie-card {
        background: #15151c;
        border: 1px solid #292933;
        border-radius: 18px;
        padding: 24px 26px;
        margin: 16px 0;
        transition: 0.2s ease;
    }

    .movie-card:hover {
        border-color: #555565;
        transform: translateY(-2px);
    }

    .movie-rank {
        color: #777783;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .movie-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .movie-genres {
        color: #92929d;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }

    .movie-score {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .movie-reason {
        color: #c2c2ca;
        line-height: 1.55;
        font-size: 0.95rem;
    }

</style>
""", unsafe_allow_html=True)


# =========================
# Header
# =========================

st.markdown(
    '<div class="cinemate-title">🎬 CineMate</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="cinemate-subtitle">'
    'Your AI-powered movie recommendation assistant.'
    '</div>',
    unsafe_allow_html=True
)


# =========================
# Search
# =========================

st.markdown(
    '<div class="search-title">What are you in the mood for?</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="search-description">'
    'Describe the kind of movie you want. '
    'You can mention genres, mood, themes, or things you want to avoid.'
    '</div>',
    unsafe_allow_html=True
)

user_query = st.chat_input(
    "e.g. A dark psychological thriller with mind games and plot twists..."
)


# =========================
# Recommendation pipeline
# =========================

if user_query:

    with st.spinner("🎬 CineMate is finding your movies..."):

        results, preferences = recommend(
            user_id=1,
            user_query=user_query,
            top_n=10
        )


    # =========================
    # No results
    # =========================

    if results is None:

        st.warning(
            "I couldn't find movies matching all of those requirements. "
            "Try relaxing one of your preferences."
        )


    # =========================
    # Results
    # =========================

    else:

        # Preference summary

        st.markdown(
            '<div class="preference-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="preference-title">'
            '🧠 CineMate understood your request'
            '</div>',
            unsafe_allow_html=True
        )

        for category, values in preferences.items():

            if values:

                label = category.replace(
                    "_",
                    " "
                ).title()

                formatted_values = " · ".join(values)

                st.markdown(
                    f'<span class="preference-item">'
                    f'<strong>{label}:</strong> '
                    f'{formatted_values}'
                    f'</span>',
                    unsafe_allow_html=True
                )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # Recommendation heading

        st.markdown(
            '<div class="search-title">'
            '🍿 Your recommendations'
            '</div>',
            unsafe_allow_html=True
        )


        # Movie cards

        for rank, (_, movie) in enumerate(
            results.iterrows(),
            start=1
        ):

            st.markdown(
                f"""
                <div class="movie-card">

                    <div class="movie-rank">
                        #{rank}
                    </div>

                    <div class="movie-title">
                        {movie['title']}
                    </div>

                    <div class="movie-genres">
                        {movie['genres']}
                    </div>

                    <div class="movie-score">
                        🎯 {movie['score']}/100 Match
                    </div>

                    <div class="movie-reason">
                        {movie['reason']}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )