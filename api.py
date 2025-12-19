from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cohere
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
import google.generativeai as genai  # Top pe import karo

# Load environment variables
load_dotenv()

app = FastAPI(title="Physical AI & Humanoid Robotics Chatbot")

# Load API keys from environment variables
cohere_api_key = os.getenv("COHERE_API_KEY", "PPlOwcg6MeztcYwW3tAHZz0nz42fbGkTcnEx8qu4")  # Fallback to hardcoded key
qdrant_url = os.getenv("QDRANT_URL", "https://2548f157-8d4c-4bf9-8c60-70f9c692288a.europe-west3-0.gcp.cloud.qdrant.io:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.BJppXRGbIaGVSwQWHEpPc-UjDgI0LX21FGWvyf4Fmyc")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Cohere aur Qdrant clients
cohere_client = cohere.Client(cohere_api_key)

qdrant = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key
)

# Gemini configure (env se key load)
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
else:
    raise ValueError("GEMINI_API_KEY environment variable not set")

class Query(BaseModel):
    message: str

def get_embedding(text):
    try:
        response = cohere_client.embed(
            model="embed-english-v3.0",
            input_type="search_query",
            texts=[text]
        )
        return response.embeddings[0]
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting embedding: {str(e)}")

def retrieve_context(query: str):
    embedding = get_embedding(query)
    results = qdrant.query_points(
        collection_name="Physical AI & Humanoid Robotics",
        query=embedding,
        limit=5
    )
    return "\n\n".join([point.payload["text"] for point in results.points])

@app.post("/chat")
async def chat(query: Query):
    context = retrieve_context(query.message)

    # Latest stable fast model use kar rahe hain
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
    You are an expert tutor for the book "Physical AI & Humanoid Robotics".
    Answer the user's question using ONLY the following context.
    If the answer is not in the context, say "I don't have information about that in the book."

    Context:
    {context}

    Question: {query.message}
    Answer:
    """

    try:
        response = model.generate_content(prompt)
        # Safely text extract karo
        if hasattr(response, 'text'):
            answer = response.text
        elif response.parts:
            answer = "".join(part.text for part in response.parts)
        else:
            answer = "Sorry, no response generated."
    except Exception as e:
        answer = f"Error: {str(e)} (Check GEMINI_API_KEY or model name)"

    return {"response": answer}

@app.get("/")
async def root():
    return {"message": "Chatbot API is live! POST to /chat"}