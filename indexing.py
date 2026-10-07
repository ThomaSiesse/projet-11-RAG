import pandas as pd
import numpy as np
import faiss
from mistralai.client import Mistral
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

print(f"\n--- début de l'indexation ---")

# Charger les chunks
print("\n--- Chargement des chunks---")
df_chunks = pd.read_json("chunks_marseille_cache.json")

# Extraire embeddings en numpy array float32
embeddings_array = np.array(df_chunks["embedding"].tolist(), dtype="float32")

# Créer index FAISS
print("\n---Création de l'index FAISS ---")
dimension = embeddings_array.shape[1]  # 1024
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# Sauvegarder
faiss.write_index(index, "faiss_chunks.bin")
print(f"Index créé : {index.ntotal} chunks")
print(f"Index sauvegardé : faiss_chunks.bin")

# Test de recherche
print("\n--- Test de recherche sémantique ---\n")

query = "je veux voir un spectacle"
print(f"🔍 Query: {query}\n")

# Créer embedding de la question
response = client.embeddings.create(model="mistral-embed", inputs=[query])
query_embedding = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)

# Rechercher
k = 5
distances, indices = index.search(query_embedding, k)

print(f"📊 Top {k} résultats :\n")
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
    idx = int(idx)
    row = df_chunks.iloc[idx]

    title = row["title"]
    location = row["location"]
    date = row["date"]
    description = row["description"]

    print(f"{rank}. (distance: {dist:.4f})")
    print(f"   📌 {title}")
    print(f"   📍 {location}")
    print(f"   📅 {date}")
    print(f"   💬 {description[:100]}...\n")
