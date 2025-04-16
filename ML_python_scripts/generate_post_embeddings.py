#!/usr/bin/env python3

# I have no idea what a postgres implimentation looks like
# So I don't know if we need a shebang
# Anyway here's some imports

# import psycopg2.pool
from sentence_transformers import SentenceTransformer
# import json

# I imagine that since this'll be running on cody's machine
# I won't need to have this downloaded.
# Jury's out though
#   I am in fact downloading this as I am testing it on my own machine
#   Also, past Ace, it wasn't that big of a deal.
#   Don't be a baby next time
import psycopg2
import re
import numpy as np
from huggingface_hub import login
import os
from dotenv.main import load_dotenv
from fastapi import APIRouter
from sklearn.decomposition import PCA
load_dotenv()

router = APIRouter()

os.getenv("HF_TOKEN")

model = SentenceTransformer('paraphrase-albert-small-v2')


def pull_tags(text):
    return re.findall(r'#\w+', text)


def clean_text(text):
    return re.sub(r'#\w+', '', text)


def camel_case_split(str):
    return re.sub(r'([a-z])([A-Z])',
                  r'\1 \2', str)


def preprocess_tags(tags):
    if isinstance(tags, str):
        tags = camel_case_split(tags)
        return [tag.strip() for tag in tags.split(',') if tag.strip()]
    elif isinstance(tags, list):
        return [camel_case_split(tag) for tag in tags]
    return []


def generate_weighted_embeddings(post):
    print(f"generating embeddings for post id {post['id']}")

    tags = pull_tags(post['text'])
    tags = preprocess_tags(tags)
    print(f"tags pulled: {tags}")

    text = clean_text(post['text'])
    print(f"text pulled: {text}")

    goods = post['goods']
    print(f"goods pulled: {goods}")

    tag_embedding = model.encode(" ".join(tags)) if tags else np.zeros(384)
    goods_embedding = model.encode(goods) * 2
    # *2 so we put more weight on the goods that the user wants
    # This can be changed later, hyperparameter tuning is vibes based
    text_embedding = model.encode(text) if text else np.zeros(384)

    combined = np.concatenate([
        tag_embedding,
        goods_embedding,
        text_embedding * 0.3
        # more vibes based hyperparameter tuning
    ])

    return combined


@router.get("/api/py/embed")
# Do I need these if the API calls happen somewhere else???
# Is it that easy???????
def update_post_embeddings():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT posts.id, text, goods.name
        FROM posts JOIN goods on goods.id = posts.good_id
        WHERE embedding IS NULL;
    """)
    posts = cursor.fetchall()

    if not posts:
        print("No new posts to update.")
        return

    embeddings = []
    ids = []

    for post in posts:
        post_dict = {
            "id": post[0],
            "text": post[1],
            "goods": post[2]
        }
        emb = generate_weighted_embeddings(post_dict)
        embeddings.append(emb)
        ids.append(post_dict["id"])

    embeddings = np.array(embeddings)

    # Apply PCA once to all embeddings
    pca = PCA(n_components=384)
    reduced = pca.fit_transform(embeddings)

    for emb, post_id in zip(reduced, ids):
        cursor.execute("""
            UPDATE posts SET embedding = %s WHERE id = %s;
        """, (emb.tolist(), post_id))

    conn.commit()
    conn.close()
    print(f"{len(ids)} posts updated with embeddings.")


if __name__ == "__main__":
    update_post_embeddings()
