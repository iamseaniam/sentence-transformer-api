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
    goods_embedding = model.encode(str(post['goods'])) * 2
    text_embedding = model.encode(text) if text else np.zeros(384)

    print(f"Tag embedding shape: {tag_embedding.shape}")
    print(f"Goods embedding shape: {goods_embedding.shape}")
    print(f"Text embedding shape: {text_embedding.shape}")

    combined = np.concatenate([
        tag_embedding,
        goods_embedding,
        text_embedding * 0.3
    ])

    print(f"Combined embedding shape: {combined.shape}")
    if len(combined) != 1920:
        return None
    else:
        return combined


@router.get("/api/py/embed")
def update_post_embeddings():
    post_query = """SELECT posts.id, text, goods.name as goods
                    FROM posts JOIN goods on goods.id = posts.good_id
                    WHERE embedding IS NULL;"""

    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cursor = conn.cursor()

    cursor.execute(post_query)
    posts = cursor.fetchall()

    if not posts:
        print("All posts have embeddings")
        return

    post_ids = []
    raw_embeddings = []

    for post in posts:
        post_dict = {
            "id": post[0],
            "text": post[1],
            "goods": post[2]
        }

        embedding = generate_weighted_embeddings(post_dict)
        if embedding is None:
            print(f"Post {post_dict['id']} has an invalid embedding.")
            continue

        raw_embeddings.append(embedding)
        post_ids.append(post_dict["id"])

    print("Fitting PCA on collected embeddings...")
    pca = PCA(n_components=512)
    reduced_embeddings = pca.fit_transform(raw_embeddings)

    print("Updating database with PCA-reduced embeddings...")
    for i, reduced in enumerate(reduced_embeddings):
        cursor.execute("""
            UPDATE posts
            SET embedding = %s
            WHERE id = %s;
        """, (reduced.tolist(), post_ids[i]))

    conn.commit()
    conn.close()
    print(f"{len(reduced_embeddings)} posts updated successfully with PCA-reduced embeddings.")


if __name__ == "__main__":
    update_post_embeddings()
