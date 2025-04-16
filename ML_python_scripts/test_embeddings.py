#!/usr/bin/env python3
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from sklearn.decomposition import PCA
import json
import os
from pathlib import Path

# Initialize model
model = SentenceTransformer('paraphrase-albert-small-v2')

# Mock data generator
def generate_mock_posts(num_posts=20):
    goods = ["Laptop", "Textbook", "Coffee", "Bike", "Phone"]
    tags = ["#NeedHelp", "#Urgent", "#StudyGroup", "#ForSale", "#Wanted"]
    mock_posts = []
    
    for i in range(num_posts):
        text = f"Looking for a {goods[i%len(goods)]} {tags[i%len(tags)]} "
        text += " ".join([f"extra{i+j}" for j in range(3)])
        
        mock_posts.append({
            "id": i+1,
            "text": text,
            "goods": goods[i%len(goods)]
        })
    
    return mock_posts

# Your existing processing functions (slightly modified for local use)
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

def generate_weighted_embeddings(post):
    print(f"Generating embeddings for mock post {post['id']}")
    
    tags = pull_tags(post['text'])
    tags = preprocess_tags(tags)
    text = clean_text(post['text'])
    goods = post['goods']

    tag_embedding = model.encode(" ".join(tags)) if tags else np.zeros(384)
    goods_embedding = model.encode(goods) * 2
    text_embedding = model.encode(text) * 0.3 if text else np.zeros(384)

    combined = np.concatenate([
        tag_embedding,
        goods_embedding,
        text_embedding
    ])
    
    return combined

def test_pca_workflow():
    # Create test directory
    test_dir = Path("embedding_tests")
    test_dir.mkdir(exist_ok=True)
    
    # Generate mock data
    mock_posts = generate_mock_posts()
    with open(test_dir/"mock_posts.json", "w") as f:
        json.dump(mock_posts, f, indent=2)
    
    # Generate embeddings
    embeddings = []
    for post in mock_posts:
        emb = generate_weighted_embeddings(post)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    np.save(test_dir/"raw_embeddings.npy", embeddings)
    print(f"Original embedding shape: {embeddings.shape}")
    
    # Apply PCA
    pca = PCA(n_components=384)
    reduced = pca.fit_transform(embeddings)
    np.save(test_dir/"pca_reduced.npy", reduced)
    
    # Save PCA info
    pca_info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
        "components_shape": pca.components_.shape
    }
    with open(test_dir/"pca_info.json", "w") as f:
        json.dump(pca_info, f, indent=2)
    
    print(f"Reduced embedding shape: {reduced.shape}")
    print(f"Total variance explained: {pca_info['total_variance_explained']:.2%}")
    print(f"Results saved to {test_dir.absolute()}")

if __name__ == "__main__":
    test_pca_workflow()