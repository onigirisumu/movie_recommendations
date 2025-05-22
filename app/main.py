from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer
import os

app = FastAPI()

# Serve frontend files from the correct location
app.mount("/frontend", StaticFiles(directory="../frontend"), name="frontend")

# Load models (using absolute paths for reliability)
model = joblib.load(os.path.join(os.path.dirname(__file__), "multi_label_model.pkl"))
mlb = joblib.load(os.path.join(os.path.dirname(__file__), "multi_label_binarizer.pkl"))
movies_df = joblib.load(os.path.join(os.path.dirname(__file__), "movies_data.pkl"))

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class UserRequest(BaseModel):
    overview: str

@app.post("/predict")
async def predict(request: UserRequest):
    embedding = embedding_model.encode([request.overview])
    pred = model.predict_proba(embedding)
    genres = mlb.inverse_transform((pred >= 0.3).astype(int))[0]
    
    # Get recommendations
    mask = movies_df['genres'].apply(lambda x: any(g in x for g in genres))
    recommendations = movies_df[mask].head(5)[['overview', 'genres']]
    
    return {
        "genres": list(genres),
        "movies": recommendations.to_dict(orient='records')
    }

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    # Read frontend file directly
    with open("../frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read())
