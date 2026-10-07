import os
from dotenv import load_dotenv
import faiss
import numpy as np
import pandas as pd
from mistralai.client import Mistral
from datetime import datetime
from mistralai.models.chat_completion import ChatMessage

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

# ========== RETRIEVAL ==========
print("📥 Chargement des données...")
df = pd.read_json("embeddings_marseille_cache.json")
index = faiss.read_index("faiss_marseille.bin")

query = "Que faire ce week-end à Marseille ?"
print(f"\n🔍 Recherche : {query}\n")

# Créer embedding de la question
response = client.embeddings.create(model="mistral-embed", inputs=[query])
query_embedding = np.array(response.data[0].embedding, dtype="float32")
query_embedding = query_embedding / np.linalg.norm(query_embedding)
query_embedding = query_embedding.reshape(1, -1)

# Chercher dans FAISS
k = 5
distances, indices = index.search(query_embedding, k)

# Récupérer les documents pertinents
documents_pertinents = []
for idx in indices[0]:
    doc = df.iloc[int(idx)]
    documents_pertinents.append(
        f"- {doc['title_fr']} ({doc['lastdate_begin']}) à {doc['location_name']}"
    )

print("📚 Documents trouvés:")
for doc in documents_pertinents:
    print(doc)

# ========== GENERATION ==========
print("\n🤖 Génération de la réponse avec Mistral...\n")

# Construire le contexte
contexte = "\n".join(documents_pertinents)

prompt = f"""Tu es un assistant pour les événements à Marseille.

Basé sur les événements suivants :
{contexte}

Réponds à cette question : {query}

Sois concis et utile."""

# Appeler Mistral pour générer la réponse
response = client.ChatMessage.create(
    model="mistral-small",  # ou "mistral-medium"
    messages=[{"role": "user", "content": prompt}],
)

reponse = response.content[0].text
print("=" * 60)
print(f"✅ RÉPONSE:\n{reponse}")
print("=" * 60)
