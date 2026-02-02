import requests
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
ENDEE_URL = "http://localhost:8080/api/v1"

def semantic_search(index_name, query, top_k=3):
    """Searches for the most similar documents in Endee."""
    query_vector = model.encode(query).tolist()
    
    # Based on server/src/main.cpp:
    # Route: /api/v1/index/<index_name>/search
    # Required: vector, k
    payload = {
        "vector": query_vector,
        "k": top_k
    }
    
    response = requests.post(f"{ENDEE_URL}/index/{index_name}/search", json=payload)
    
    # Note: Search results might be returned in MessagePack if requested, 
    # but let's see what the default JSON response is (if any).
    # Wait, the server code says: 
    # resp.add_header("Content-Type", "application/msgpack");
    # It seems to ONLY return msgpack for search results. 
    # I'll need to install msgpack and handle it.
    
    import msgpack
    if response.status_code == 200:
        try:
            results = msgpack.unpackb(response.content)
            return results
        except Exception as e:
            print(f"Failed to unpack search results: {e}")
            return []
    else:
        print(f"Search failed: {response.status_code} - {response.text}")
        return []

if __name__ == "__main__":
    query = "How does Apple Silicon help vector math?"
    results = semantic_search("research_index", query)
    
    print(f"\nQuery: {query}")
    print("Results:")
    if results:
        # Results is likely a list of matches (id, score)
        # Based on ResultSet serialization in src/core/ndd.hpp
        for match in results:
            print(f"- {match}")
    else:
        print("No results found.")
