# Indie Music Recommender with MERT + ChromaDB

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor
from transformers.models.mert.modeling_mert import MERTModel
import chromadb

# Connect to ChromaDB
client = chromadb.CloudClient(
    api_key=os.getenv("CHROMADB_API_KEY"),
    tenant=os.getenv("CHROMADB_TENANT"),
    database="indie-songs"
)
collection_name = "indie_songs"
try:
    collection = client.create_collection(name=collection_name)
except:
    collection = client.get_collection(name=collection_name)

# Load MERT
processor = AutoProcessor.from_pretrained("facebook/mert-base")
model = MERTModel.from_pretrained("facebook/mert-base").eval()

# Extract embedding
def get_mert_embedding_from_file(npy_path):
    waveform = np.load(npy_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0)
    waveform_tensor = torch.tensor(waveform).unsqueeze(0)
    with torch.no_grad():
        embedding = model(waveform_tensor).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Process dataset
def process_waveforms(waveform_folder, metadata_csv):
    df = pd.read_csv(metadata_csv)
    counter = 0
    for filename in os.listdir(waveform_folder):
        if filename.endswith('.npy'):
            npy_path = os.path.join(waveform_folder, filename)
            try:
                artist = df['Artist'][counter]
                name = df['Name'][counter]
            except:
                name, artist = filename, "Unknown"
            try:
                embedding = get_mert_embedding_from_file(npy_path)
                collection.add(
                    documents=[name],
                    metadatas=[{"Artist": artist}],
                    ids=[str(counter)],
                    embeddings=[embedding]
                )
                counter += 1
                if counter % 100 == 0:
                    print(f"Processed {counter} tracks")
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    print(f"Finished processing {counter} tracks")

# Recommend songs
def recommend_songs(chroma_client, collection_name, query_embedding, n_results=5):
    collection = chroma_client.get_collection(name=collection_name)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    recs = []
    for ids, meta, dist in zip(results["ids"][0], results["metadatas"][0], results["distances"][0]):
        recs.append({
            "id": ids,
            "title": meta.get("title", "Unknown"),
            "artist": meta.get("artist", "Unknown"),
            "genre": meta.get("genre", "Unknown"),
            "distance": dist
        })
    return recs
