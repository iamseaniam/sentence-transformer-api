#!/usr/bin/env python3

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2

import os
from dotenv import main
from typing import Optional, List
from uuid import UUID
from datetime import datetime
import random

from .cosine_similarity import recommend_cosine_sim
from .data_fetchers import *


def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError(f"Not da db. Da db: {db_url}")
    print(f"Attempting to connect to database url")

    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        print(f"failed to connect to database: {e}")
        raise


class UserProfile(BaseModel):
    user_id: int
    name: str
    avatar: str


class PostRecommendation(BaseModel):
    id: UUID
    text: str
    timestamp: datetime
    image: Optional[str]
    type: str
    goods: str
    user: UserProfile


def aggrigate_and_JSONify(post_ids, conn):
    cursor = conn.cursor()

    cursor.execute("""
        -- This entire query will need to be adjusted
        SELECT p.id, p.text, p.timestamp, p.image, p.type
            goods.name as goods, u.id as user_id
            u.name, u.image as user_avatar
        FROM posts p
        JOIN goods on goods.id = p.good_id
        JOIN users u ON p.user_id = u.id
        WHERE p.id = ANY(%s::uuid[]);
    """, (post_ids,))

    post_details = cursor.fetchall()

    populated_post_recs = []
    for row in post_details:
        populated_post_recs.append({
            "id": row[0],
            "text": row[1],
            "timestamp": row[2],
            "image": row[3] if row[3] else None,
            "type": row[4],
            "goods": row[5],
            "user": {
                "user_id": row[6],
                "name": row[7],
                "avatar": row[8]
            },
        })

    return populated_post_recs


def get_recommendations(user_id):

    print("Found function")

    conn = get_db_connection()

    post_id_recs = []

    cos_sim_recs = recommend_cosine_sim(user_id, conn)
    post_id_recs.append(cos_sim_recs)

    zipcode_recs = get_recent_posts_by_zipcode(user_id=user_id, conn=conn)
    post_id_recs.append(zipcode_recs)

    recently_liked_recs = get_recent_posts_by_liked_profiles(
        user_id=user_id, conn=conn)
    post_id_recs.append(recently_liked_recs)

    recent_posts = get_recent_posts_by_week(user_id=user_id, conn=conn)
    post_id_recs.append(recent_posts)

    flattened_posts_recs = [
        item for sublist in post_id_recs for item in sublist]

    post_id_recs = list(dict.fromkeys(flattened_posts_recs))
    populated_post_recs = aggrigate_and_JSONify(post_id_recs, conn)

    random.shuffle(populated_post_recs)

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
