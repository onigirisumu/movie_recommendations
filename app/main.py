from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()  # ← THIS is what Render is looking for!

# Load models
classifier = joblib.load("app/model_files/multi_label_model.pkl")
mlb = joblib.load("app/model_files/multi_label_binarizer.pkl")
model = joblib.load("app/model_files/bert_model.pkl")
df = joblib.load("app/model_files/movies_data.pkl")

class UserInput(BaseModel):
    text: str

def recommend_by_genres(predicted_genres, movies_df, top_k=5):
    mask = movies_df['genres'].apply(lambda genres: any(g in genres for g in predicted_genres))
    filtered = movies_df[mask]
    return filtered.head(top_k)

@app.post("/recommend")
def predict_genres(user_input: UserInput):
    user_embedding = model.encode([user_input.text])
    pred = classifier.predict(user_embedding)
    predicted_genres = mlb.inverse_transform(pred)[0]

    recommendations = recommend_by_genres(predicted_genres, df)
    result = {
        "predicted_genres": predicted_genres,
        "recommendations": [
            {
                "overview": row["overview"],
                "genres": row["genres"]
            }
            for _, row in recommendations.iterrows()
        ]
    }
    return result
