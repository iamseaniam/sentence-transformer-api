# API we're working with
from fastapi import FastAPI

# imports for generating embeddings
from ML_python_scripts.generate_post_embeddings import update_post_embeddings
from ML_python_scripts.fetch_recommendations_V2 import test_jsonify, get_recommendations, PostRecommendation
from typing import List
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()


@app.post("/api/py/helloFastApi")
def hello_fast_api():
    return {"message": "Hello from FastAPI!"}


@app.post("/api/py/embed")
def update_embeddings():
    update_post_embeddings()
    return {"message": "Congrats! The call made it through!"}


@app.post("/api/py/recommendations/{user_id}", response_model=List[PostRecommendation])
def fetch_recommendations(user_id):
    try:
        print("API call made it through for user ID:")
        print(user_id)
        recommended_posts = get_recommendations(user_id)

        return recommended_posts
    except:
        print("Issue with server")


@app.post("/api/py/test_json")
def testing_json():
    test_results = test_jsonify()
    return test_results
