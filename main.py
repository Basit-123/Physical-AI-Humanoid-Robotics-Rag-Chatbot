import requests
import xml.etree.ElementTree as ET
import trafilatura
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import cohere
import time

# -------------------------------------
# CONFIG
# -------------------------------------
SITEMAP_URL = "https://basit-123.github.io/humanoid-robotics-book/sitemap.xml"
COLLECTION_NAME = "Physical AI & Humanoid Robotics"

cohere_client = cohere.Client("PPlOwcg6MeztcYwW3tAHZz0nz42fbGkTcnEx8qu4")
EMBED_MODEL = "embed-english-v3.0"
EMBEDDING_SIZE = 1024  # Cohere v3 dimension

# Connect to Qdrant Cloud
qdrant = QdrantClient(
    url="https://2548f157-8d4c-4bf9-8c60-70f9c692288a.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.BJppXRGbIaGVSwQWHEpPc-UjDgI0LX21FGWvyf4Fmyc"
)

# -------------------------------------
# Step 1 — Extract URLs from sitemap
# -------------------------------------
def get_all_urls(sitemap_url):
    print("Fetching sitemap...")
    response = requests.get(sitemap_url)
    response.raise_for_status()
    xml = response.text
    root = ET.fromstring(xml)

    urls = []
    namespace = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    for child in root.findall(f".//{namespace}loc"):
        urls.append(child.text)

    print(f"\nFOUND {len(urls)} URLs:")
    for u in urls:
        print(" -", u)

    return urls


# -------------------------------------
# Step 2 — Download page + extract text
# -------------------------------------
def extract_text_from_url(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True, include_formatting=True)
        if not text or len(text.strip()) < 100:
            print(f"[WARNING] Very little or no text extracted from: {url}")
            return None
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Failed to extract {url}: {e}")
        return None


# -------------------------------------
# Step 3 — Chunk the text (FIXED & IMPROVED)
# -------------------------------------
def chunk_text(text, max_chars=1500):
    chunks = []
    current_chunk = ""
    
    # Split by double newlines (paragraphs), but keep non-empty
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    for para in paragraphs:
        # If adding this para keeps us under limit
        if len(current_chunk) + len(para) + 2 <= max_chars:  # +2 for \n\n
            current_chunk += ("\n\n" + para if current_chunk else para)
        else:
            # Save current chunk if exists
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If single para is too long, split it safely
            if len(para) > max_chars:
                while len(para) > max_chars:
                    # Try to split at sentence end
                    split_pos = para[:max_chars].rfind(". ")
                    if split_pos == -1:
                        split_pos = para[:max_chars].rfind(" ")  # Fall back to word
                    if split_pos == -1:
                        split_pos = max_chars  # Last resort
                    
                    split_pos += 1  # Include the . or space
                    chunks.append(para[:split_pos].strip())
                    para = para[split_pos:].strip()
            
            current_chunk = para
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out any empty chunks (just in case)
    chunks = [c for c in chunks if len(c) > 50]  # Ignore tiny chunks
    
    return chunks


# -------------------------------------
# Step 4 — Create embedding
# -------------------------------------
def embed(texts):
    response = cohere_client.embed(
        model=EMBED_MODEL,
        input_type="search_document",
        texts=texts,
    )
    return response.embeddings


# -------------------------------------
# Step 5 — Collection management
# -------------------------------------
def ensure_collection_exists():
    if qdrant.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
        return
    
    print(f"Creating collection '{COLLECTION_NAME}'...")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE
        )
    )
    print("Collection created successfully!")


# -------------------------------------
# Step 6 — Batch upsert
# -------------------------------------
def save_chunks_batch(chunks, start_id, url):
    if not chunks:
        print("No chunks to save.")
        return start_id
    
    print(f"Embedding {len(chunks)} chunks...")
    vectors = embed(chunks)
    
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point_id = start_id + i
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "url": url,
                    "text": chunk,
                    "chunk_id": point_id
                }
            )
        )
    
    print(f"Upserting {len(points)} points...")
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    return start_id + len(chunks)


# -------------------------------------
# MAIN INGESTION PIPELINE
# -------------------------------------
def ingest_book():
    ensure_collection_exists()
    
    urls = get_all_urls(SITEMAP_URL)
    
    # Ab academic-integrity ko skip mat kar – fixed chunking se chal jaega
    # Agar phir bhi nahi chahiye to ye line rakh sakta hai:
    # urls = [u for u in urls if "academic-integrity" not in u]
    
    global_id = 1
    
    for url in urls:
        print(f"\n{'='*60}")
        print(f"Processing: {url}")
        text = extract_text_from_url(url)
        
        if not text:
            continue
        
        chunks = chunk_text(text)
        print(f"Split into {len(chunks)} chunks")
        
        if chunks:
            global_id = save_chunks_batch(chunks, global_id, url)
        else:
            print("No valid chunks created.")
        
        time.sleep(0.5)  # Cohere rate limit safe
    
    print("\n" + "="*80)
    print("✔️ INGESTION COMPLETED SUCCESSFULLY!")
    print(f"Total points stored: {global_id - 1}")
    print(f"Ab Qdrant dashboard khol ke dekho – '{COLLECTION_NAME}' mein points hone chahiye!")


if __name__ == "__main__":
    ingest_book()