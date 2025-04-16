# API we're working with
from fastapi import APIRouter

# imports for generating embeddings
from ml_routes.generate_post_embeddings import update_post_embeddings
from ml_routes.fetch_recommendations_V2 import test_jsonify

# imports for fetching the recommended posts
# from ML_python_scripts.fetch_recommendations_V2 import get_recommendations, UserProfile, PostRecommendation
# from typing import List

# setting up the app.
router = APIRouter()


@router.post("/api/py/helloFastApi")
def hello_fast_api():
    return {"message": "Hello from FastAPI!"}

# Surely the generating embeddings shouldn't be that hard, right?
# What's the worst that could happen, I screw up the database?
#   Now's not the time to make jokes about that.


@router.post("/api/py/embed")
def update_embeddings():
    update_post_embeddings()
    return {"message": "Congrats! The call made it through!"}


# I genuinely have no idea what I'm doing but I think it is that simple.
# @app.get("/recommendations/{user_id}", response_model=List[PostRecommendation])
# def fetch_recommendations(user_id):
    # recommended_posts = get_recommendations(user_id)

    # return recommended_posts

# Still no idea what I'm doing but we're gonna give it our best.
@router.post("/api/py/test_json")
def testing_json():
    test_results = test_jsonify()
    return test_results
