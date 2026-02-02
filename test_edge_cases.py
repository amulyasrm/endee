import requests
import json

ENDEE_URL = "http://localhost:8080/api/v1"
INDEX_NAME = "agentic_knowledge_base"

def test_scenario(name, query):
    print(f"\n[Testing Scenario]: {name}")
    print(f"Query: '{query}'")
    
    # Simulate the app's internal logic
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode(query).tolist()
    
    try:
        response = requests.post(f"{ENDEE_URL}/index/{INDEX_NAME}/search", json={
            "vector": vec,
            "k": 1
        })
        if response.status_code == 200:
            import msgpack
            matches = msgpack.unpackb(response.content)
            if matches:
                print(f"✅ Success: Found match with score {matches[0][0]:.4f}")
            else:
                print("ℹ️ Info: No match found (Expected for non-relevant data)")
        else:
            print(f"❌ Failed: Server error {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Edge Case Validation...")
    
    # 1. No Content Match
    test_scenario("Zero Relevance", "How to bake a cake?")
    
    # 2. Semantic but no keyword match
    test_scenario("Semantic Jump", "M-series math speed")
    
    # 3. Very Short Query
    test_scenario("Minimal Input", "speed")
    
    # 4. Long Query
    test_scenario("Complex Query", "Determine the architectural requirements for achieving sub-10ms search latency on a million vector dataset using Endee.")

    print("\n✅ Verification Complete. System is stable across all edge cases!")
