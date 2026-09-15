# 🎬 CineMate

**AI-Powered Movie Recommendation Assistant**

CineMate is a hybrid movie recommendation system that combines semantic search, collaborative filtering, and Google Gemini to understand natural-language movie preferences and return personalized recommendations.

## 🚀 Live Demo

[Launch CineMate](https://cinemate-bestmovies.streamlit.app)

## 📌 Features

- Natural-language movie search
- Gemini-powered preference extraction
- Semantic movie matching using MPNet embeddings
- Collaborative filtering using SVD
- Hybrid recommendation ranking
- Gemini-powered final reranking
- Grounded recommendation explanations
- Interactive Streamlit interface

## 🧠 Recommendation Pipeline

```text
User Query
    ↓
Gemini Preference Extraction
    ↓
Genre / Exclusion Filtering
    ↓
MPNet Semantic Ranking
    ↓
SVD Collaborative Filtering
    ↓
Hybrid Ranking
    ↓
Gemini Final Reranking
    ↓
Top Movie Recommendations

🛠️ Tech Stack
Python
Pandas
NumPy
Scikit-learn
SciPy
Sentence Transformers
Google Gemini API
Streamlit
Joblib
📊 Dataset

CineMate uses the MovieLens Latest-Small dataset from GroupLens.

The project uses movie metadata, user ratings, and user-generated tags to build the recommendation system.

🔍 Machine Learning
Semantic Recommendation

Movie titles, genres, and tags are converted into text representations and embedded using:

sentence-transformers/all-mpnet-base-v2

Cosine similarity is then used to find movies semantically related to the user's request.

Collaborative Filtering

A user-movie rating matrix is decomposed using:

TruncatedSVD

The resulting user and movie factors provide collaborative preference signals.

Hybrid Ranking

Semantic and collaborative scores are combined:

75% semantic + 25% collaborative

The resulting candidates are passed to Gemini for final natural-language-aware reranking.

🤖 Generative AI

Gemini is used for two tasks:

Preference extraction — converts natural-language requests into structured preferences such as genres, moods, themes, and exclusions.
Final reranking — evaluates the retrieved candidates against the user's request and produces concise, grounded explanations.
▶️ Run Locally

Clone the repository:

git clone https://github.com/mohammedmustafa678/CineMate.git
cd CineMate

Create and activate a virtual environment:

python -m venv .venv

Windows:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Set your Gemini API key:

$env:GEMINI_API_KEY="YOUR_API_KEY"

Run the application:

streamlit run app.py
📁 Project Structure
CineMate/
├── app.py
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── links.csv
├── models/
│   ├── movie_embeddings.pkl
│   ├── movie_factors.pkl
│   ├── user_factors.pkl
│   ├── user_movie_matrix.pkl
│   └── movies_text.pkl
├── notebooks/
│   └── cinemate.ipynb
├── requirements.txt
└── README.md
👨‍💻 Project

Built as a portfolio project demonstrating practical application of:

Machine Learning + NLP + Recommender Systems + Generative AI + Streamlit

⭐ If you found the project interesting, feel free to explore the code and notebook.
