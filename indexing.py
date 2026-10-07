import os
from dotenv import load_dotenv
import faiss
import numpy as np
import pandas as pd
from mistralai.client import Mistral
from datetime import datetime

# Initialisation du client avec votre clé API
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

import pandas as pd
import numpy as np
import faiss

# Charger les données
df = pd.read_json("embeddings_marseille_cache.json")
# Filtrer les événements à venir
today = datetime.now().strftime("%Y-%m-%d")
df_future = df[df["lastdate_begin"] >= today].reset_index(drop=True)
print(f"✓ {len(df_future)} événements à venir")
# Préparer embeddings avec normalisation L2
embeddings_array = np.array(df_future["embedding"].tolist(), dtype="float32")
faiss.normalize_L2(embeddings_array)  # ← Normaliser pour cosine

# Créer index avec Inner Product (= Cosine distance)
dimension = embeddings_array.shape[1]
index = faiss.IndexFlatIP(dimension)  # ✅ Cosine au lieu de L2
index.add(embeddings_array)

faiss.write_index(index, "faiss_marseille.bin")
print(f"✓ Index créé : {index.ntotal} documents (distance COSINE)")

# Recherche
query = "évenements ce week end"  # ← Plus spécifique
print(f"\n🔍 Recherche : {query}\n")

response = client.embeddings.create(model="mistral-embed", inputs=[query])
query_embedding = np.array(response.data[0].embedding, dtype="float32")
faiss.normalize_L2(query_embedding.reshape(1, -1))  # Normaliser aussi
query_embedding = query_embedding.reshape(1, -1)

k = 5
distances, indices = index.search(query_embedding, k)

print(f"📊 Top {k} résultats :\n")
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
    doc = df_future.iloc[int(idx)]
    print(f"{rank}. (score: {dist:.4f})")
    print(f"   {doc['title_fr']}")
    print(f"   📅 {doc['lastdate_begin']} | 📍 {doc['location_name']}\n")
