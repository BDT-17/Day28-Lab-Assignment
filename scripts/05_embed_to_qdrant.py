# scripts/05_embed_to_qdrant.py
import os

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_dotenv()

EMBED_URL = os.environ["EMBED_NGROK_URL"]
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}
qdrant = QdrantClient(host="localhost", port=6333)

# Tạo collection
qdrant.recreate_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

def embed_and_store(records: list[dict]):
    # Gọi Kaggle embedding service
    response = requests.post(
        f"{EMBED_URL}/embed",
        json={"texts": [r["text"] for r in records]},
        headers=NGROK_HEADERS,
    )
    response.raise_for_status()
    payload = response.json()
    if "embeddings" not in payload:
        raise ValueError(f"Unexpected response from embed service: {str(payload)[:300]}")
    embeddings = payload["embeddings"]

    points = [
        PointStruct(id=i, vector=emb, payload=rec)
        for i, (emb, rec) in enumerate(zip(embeddings, records))
    ]
    qdrant.upsert(collection_name="documents", points=points)
    print(f"Integration 5 OK: {len(points)} vectors stored in Qdrant")

# Test với sample data
embed_and_store([
    {"id": "doc_001", "text": "AI platform integration test"},
    {"id": "doc_002", "text": "Kafka to Airflow pipeline"},
])
