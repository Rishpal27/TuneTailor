# -*- coding: utf-8 -*-


!pip install datasets soundfile torchcodec nnAudio transformers torch torchaudio chromadb

from google.colab import drive
import os

mount_point = '/content/drive/MyDrive/Colab_Notebooks/waveforms'

# Create the directory if it doesn't exist
if not os.path.exists(mount_point):
    os.makedirs(mount_point)

drive.mount(mount_point)

import os
import numpy as np
import torch
from transformers import AutoProcessor
from transformers.models.mert.modeling_mert import MERTModel
import chromadb

# Connect to your ChromaDB collection
client = chromadb.CloudClient(
  api_key='YOUR API KEY',
  tenant='',
  database='indie-songs'
)
collection_name = "indie_songs"
# Create if doesn't exist, otherwise get
try:
    collection = client.create_collection(name=collection_name)
except:
    collection = client.get_collection(name=collection_name)
print(f"Connected to collection '{collection_name}'")

# Load the MERT processor and model
processor = AutoProcessor.from_pretrained("facebook/mert-base")
model = MERTModel.from_pretrained("facebook/mert-base")
model.eval()

# Define the function to get MERT embeddings
def get_mert_embedding_from_file(npy_path):
    waveform = np.load(npy_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0)
    waveform_tensor = torch.tensor(waveform).unsqueeze(0)
    with torch.no_grad():
        embedding = model(waveform_tensor).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

# Process the audio files and add embeddings to the collection
counter = 0
df=pd.read_csv("/content/data4.csv")
for filename in os.listdir(waveform_folder):
    if filename.endswith('.npy'):
        npy_path = os.path.join(waveform_folder, filename)
        try:
            artist=df['Artist'][counter]
            name = df['Name'][counter]
        except:
            name = filename
            artist = "Unknown"

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
counter = 0
df=pd.read_csv("/content/data4.csv")
for filename in os.listdir(waveform_folder):
    if filename.endswith('.npy'):
        npy_path = os.path.join(waveform_folder, filename)
        try:
            artist=df['Artist'][counter]
            name = df['Name'][counter]
        except:
            name = filename
            artist = "Unknown"

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
            print(f" Skipping {filename}: {e}")

print(f"Finished processing {counter} tracks")

//RECOMMENDATION SEARCHING
def recommend_songs(chroma_client, collection_name, query_embedding, n_results=5):
    collection = chroma_client.get_collection(name=collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results)
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
