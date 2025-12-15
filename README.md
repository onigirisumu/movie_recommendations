🎬 Movie Recommendation System (FastAPI + NLP)

This project is a movie recommendation web application that predicts movie genres from a user’s text description and recommends similar movies based on those genres. It uses FastAPI for the backend and a simple HTML/JavaScript frontend.

⸻

🚀 Features
	•	User enters a text description of what they want to watch
	•	NLP model converts text into embeddings using Sentence Transformers
	•	A trained multi-label classification model predicts movie genres
	•	Recommends top 5 movies matching the predicted genres
	•	Simple web interface for interaction

⸻

🧠 Technologies Used
	•	FastAPI - backend API
	•	Uvicorn -ASGI server
	•	Sentence-Transformers - text embeddings
	•	Scikit-learn - multi-label classification
	•	Pandas & NumPy - data handling
	•	Joblib - model loading
	•	HTML / JavaScript - frontend UI

⸻

📁 Project Structure

project/

backend/
main.py
multi_label_model.pkl
multi_label_binarizer.pkl
movies_data.pkl


frontend/
index.html

requirements.txt

README.md

⸻

⚙️ Installation
	1.	Clone the repository
	2.	Create and activate a virtual environment (optional but recommended)
	3.	Install dependencies:

pip install -r requirements.txt


⸻

▶️ Running the Application

From the backend directory, run:

uvicorn main:app --reload

Then open your browser and go to:

http://127.0.0.1:8000


⸻

📡 API Endpoint

POST /predict

Request body (JSON):

{
  "overview": "I want a dark sci-fi movie with space and action"
}

Response:
	•	Predicted genres
	•	List of recommended movies with overview and genres

⸻

🖥 Frontend
	•	Accessible from the root URL /
	•	Allows users to describe a movie they want to watch
	•	Displays recommended movies and detected genres dynamically

⸻

📌 Notes
	•	Ensure all .pkl model files are placed in the backend directory
	•	The genre prediction threshold is set to 0.3
	•	Recommendations are filtered based on overlapping genres

⸻

📄 License

This project is for educational and demonstration purposes.
