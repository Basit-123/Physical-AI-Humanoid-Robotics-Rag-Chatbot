import cohere
from qdrant_client import QdrantClient

# Initialize Cohere client
cohere_client = cohere.Client("PPlOwcg6MeztcYwW3tAHZz0nz42fbGkTcnEx8qu4")

# Connect to Qdrant
qdrant = QdrantClient(
    url="https://2548f157-8d4c-4bf9-8c60-70f9c692288a.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.BJppXRGbIaGVSwQWHEpPc-UjDgI0LX21FGWvyf4Fmyc" 
)

def get_embedding(text):
    """Get embedding vector from Cohere Embed v3"""
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",  # Use search_query for queries
        texts=[text],
    )
    return response.embeddings[0]  # Return the first embedding

def retrieve(query):
    embedding = get_embedding(query)
    result = qdrant.query_points(
        collection_name="Physical AI & Humanoid Robotics",
        query=embedding,
        limit=5
    )
    return [point.payload["text"] for point in result.points]

# Test
print(retrieve("What data do you have?"))