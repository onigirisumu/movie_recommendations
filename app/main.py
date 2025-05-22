from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

app = FastAPI(title="Multi-label Movie Genre Recommender")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class MovieOverview(BaseModel):
    overview: str

# Load models - adjust paths based on your actual file structure
try:
    model = joblib.load("multi_label_model.pkl")
    mlb = joblib.load("multi_label_binarizer.pkl")
    movies_df = joblib.load("movies_data.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Load SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def recommend_by_genres(predicted_genres, movies_df, top_k=5):
    mask = movies_df['genres'].apply(lambda genres: any(g in genres for g in predicted_genres))
    filtered = movies_df[mask]
    return filtered.head(top_k)[['overview', 'genres']].to_dict(orient='records')

@app.post("/predict/")
async def predict_genres(movie: MovieOverview):
    overview_text = movie.overview

    # Embed input
    embedding = embedding_model.encode([overview_text])

    # Predict multi-label probabilities
    pred_prob = model.predict_proba(embedding)

    # Threshold to get binary labels
    threshold = 0.3
    pred_binary = (pred_prob >= threshold).astype(int)

    # Decode genre labels
    genres = mlb.inverse_transform(pred_binary)[0]

    # Get recommended movies by those genres
    recommendations = recommend_by_genres(genres, movies_df)

    return {
        "predicted_genres": list(genres),
        "recommendations": recommendations
    }

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
