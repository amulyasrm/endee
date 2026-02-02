import requests
from sentence_transformers import SentenceTransformer
import os
import json

# Initialize the embedding model (local, no API key needed)
# This will be cached after the first run
model = SentenceTransformer("all-MiniLM-L6-v2")

ENDEE_URL = "http://localhost:8080/api/v1"

def create_index(index_name):
    """Creates an index in Endee vector database."""
    print(f"Creating index: {index_name}...")
    try:
        # Based on server/src/main.cpp:
        # Requires: index_name, dim, space_type
        payload = {
            "index_name": index_name,
            "dim": 384,  # all-MiniLM-L6-v2 dimension
            "space_type": "cosine" # valid options usually 'l2', 'ip', 'cosine'
        }
        response = requests.post(f"{ENDEE_URL}/index/create", json=payload)
        print(f"Response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Index creation failed: {e}")

def ingest_text(index_name, text_id, text_content, metadata=None):
    """Embeds text and inserts it into Endee."""
    print(f"Ingesting: {text_id}...")
    vector = model.encode(text_content).tolist()
    
    # Based on server/src/main.cpp:
    # Route: /api/v1/index/<index_name>/vector/insert
    # Payload: {"id": "...", "vector": [...]}
    payload = {
        "id": text_id,
        "vector": vector
    }
    
    response = requests.post(
        f"{ENDEE_URL}/index/{index_name}/vector/insert", 
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    if response.status_code == 200:
        print(f"Successfully ingested {text_id}")
    else:
        print(f"Failed to ingest {text_id}: {response.status_code} - {response.text}")
    return response.status_code

if __name__ == "__main__":
    # Example documents
    docs = [
        {"id": "doc1", "text": "Endee is a high-performance vector database built for SIMD optimization."},
        {"id": "doc2", "text": "Semantic search understands the meaning of words, not just exact matches."},
        {"id": "doc3", "text": "Retrieval Augmented Generation (RAG) uses a vector database to provide context to LLMs."},
        {"id": "doc4", "text": "Apple Silicon M-series chips support NEON instructions for fast vector math."}
    ]
    
    idx_name = "research_index"
    create_index(idx_name)
    
    for doc in docs:
        ingest_text(idx_name, doc["id"], doc["text"])
