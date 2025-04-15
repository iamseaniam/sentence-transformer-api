import os
import re
import numpy as np
from fastapi import FastAPI
from huggingface_hub import login
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg2
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
try:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN enviroment variable not found/set")

    login(hf_token)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error during startup: {e}")
    model = None


def pull_tags(text):
    return re.findall(r'#\w+', text)


def clean_text(text):
    return re.sub(r'#\w+', '', text)


def camel_case_split(s):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', s)


def preprocess_tags(tags):
    if isinstance(tags, str):
        tags = camel_case_split(tags)
        return [tag.strip() for tag in tags.split(',') if tag.strip()]
    elif isinstance(tags, list):
        return [camel_case_split(tag) for tag in tags]
    return []


def generate_weighted_embeddings(text, goods):
    tags = pull_tags(text)
    tags = preprocess_tags(tags)
    text_clean = clean_text(text)

    tag_embedding = model.encode(" ".join(tags)) if tags else np.zeros(384)
    goods_embedding = model.encode(goods) * 2
    text_embedding = model.encode(text_clean) if text_clean else np.zeros(384)

    combined = np.concatenate([
        tag_embedding,
        goods_embedding,
        text_embedding * 0.3
    ])

    return combined


# API route

@app.post("/api/embed")
def update_post_embeddings():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cursor = conn.cursor()

    post_query = """
        SELECT posts.id, posts.text, goods.name 
        FROM posts 
        JOIN goods on goods.id = posts.good_id
        WHERE embedding IS NULL;
    """

    cursor.execute(post_query)
    posts = cursor.fetchall()

    if not posts:
        return {"message": "All posts have embeddings!"}

    update_count = 0
    for post_id, text, goods_name in posts:
        embedding = generate_weighted_embeddings(text, goods_name)
        cursor.execute("""
            UPDATE posts
            SET embedding = %s
            WHERE id = %s;
        """, (embedding.tolist(), post_id))
        update_count += 1

    conn.commit()
    conn.close()

    return {"message": f"{update_count} posts updated successfully"}
