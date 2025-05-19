#!/usr/bin/env python3
"""Recommend posts based on cosine similarity of embeddings"""
import psycopg2
import os


def recommend_cosine_sim(user_id, conn):
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.id, 1 - (p.embedding <=> u_avg.avg_embedding) AS similarity -- selecting post IDs and Embeddings
        FROM posts p
        CROSS JOIN (
            SELECT AVG(embedding) AS avg_embedding
            FROM posts
            WHERE id IN (
                SELECT post_id 
                FROM posts_likes 
                WHERE user_id = %s::uuid
            )
            AND embedding IS NOT NULL
        ) u_avg -- Average of the user embeddings (average of the things they like)
        WHERE p.id NOT IN (
            SELECT post_id 
            FROM posts_likes 
            WHERE user_id = %s::uuid
        ) -- getting posts the user hasn't already liked
        AND p.embedding IS NOT NULL
        -- this is where I'd include datetime limits I think. Only recommend recent posts, you know?
        ORDER BY similarity DESC
        LIMIT 10
    """, (user_id, user_id))
    recommended_posts = cursor.fetchall()

    post_id_list = [post[0] for post in recommended_posts]

    print("posts from cosine sim:")
    for post_id in post_id_list:
        print(post_id)

    return post_id_list
