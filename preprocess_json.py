import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib


if os.path.exists("embeddings.joblib"):
    print("embeddings.joblib already exists.")
    print("Delete it if you want to regenerate embeddings.")
    exit()

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    data = r.json()

    if "embeddings" not in data:
        print("Error:", data)
        return None

    return data["embeddings"]


jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

BATCH_SIZE = 200

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)

    print(f"Creating Embeddings for {json_file}")

    chunks = content["chunks"]
    embeddings = []

    # Generate embeddings in batches
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]

        batch_embeddings = create_embedding([c["text"] for c in batch])

        if batch_embeddings is None:
            raise Exception("Embedding generation failed.")

        embeddings.extend(batch_embeddings)

    # Same logic as your original code
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)

# Save dataframe
joblib.dump(df, "embeddings.joblib")

print("Embeddings saved successfully.")