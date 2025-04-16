#!/usr/bin/env python3

from pydantic import BaseModel
import psycopg2

import os
from datetime import datetime

# import cosine_similarity as cos_sim

# Connecting to database


def get_db_connection():
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    return conn


class UserProfile(BaseModel):
    user_id: int
    name: str
    avatar: str
    # avatar data type will more than likely be changed


class PostRecommendation(BaseModel):
    post_id: int
    text: str
    goods: str
    image: str  # Once again this will more than likely be changed
    created_at: datetime
    user: UserProfile


def aggrigate_and_JSONify(post_ids, conn):
    cursor = conn.cursor()

    cursor.execute("""
        -- This entire query will need to be adjusted
        SELECT p.id, p.text, p.timestamp, p.image,
            goods.name as goods, u.name, u.image
        FROM posts p
        JOIN goods on goods.id = p.good_id
        JOIN users u ON p.user_id = u.id
        WHERE p.id = ANY(%s::uuid[]);
    """, (post_ids,))

    post_details = cursor.fetchall()

    populated_post_recs = []
    for row in post_details:
        populated_post_recs.append({
            "post_id": row[0],
            "text": row[1],
            "timestamp": row[2],
            "image": row[3],
            "goods": row[4],
            "user": {
                "name": row[5],
                "avatar": row[6]
            },
        })

    return populated_post_recs


def get_recommendations(user_id):
    conn = get_db_connection()

    post_id_recs = []

    cos_sim_recs = cos_sim.recommend_cosine_sim(user_id, conn)

    post_id_recs.append(cos_sim_recs)

    # Here's where we would call Sean's other functions
    # and append them to a list

    populated_post_recs = aggrigate_and_JSONify(post_id_recs, conn)

    conn.close()

    return populated_post_recs


def test_jsonify():

    conn = get_db_connection()
    cursor = conn.cursor()

    test_posts = []

    cursor.execute("""
        SELECT id FROM posts;
                   """)

    test_posts = cursor.fetchall()

    populated_test_posts = aggrigate_and_JSONify(test_posts, conn)

    conn.close()

    return populated_test_posts
