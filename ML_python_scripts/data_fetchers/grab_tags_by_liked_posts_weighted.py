#!/user/bin/env ptython3
import psycopg2
from typing import List, Optional
from numpy import dot
from numpy.linalg import norm


def get_weighted_tags_from_liked_posts(user: int, conn: psycopg2.extensions.connection, limit: int = 5) -> Optional[List[int]]:
    """
    Recommend posts based on tags from the user's liked posts.
    Returns a list of post_ids.
    """
