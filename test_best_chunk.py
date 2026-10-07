import pandas as pd
import numpy as np
import re
import json
import os
from mistralai.client import Mistral
from datetime import datetime, timedelta
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from dotenv import load_dotenv

load_dotenv()
from dotenv import load_dotenv

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)
model = "mistral-embed"

# Date limite
date_un_an = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")


# Nettoyage HTML
def nettoyage_html(texte):
    if pd.isna(texte):
        return ""
    texte = re.sub(r"<[^>]+>", "", str(texte))
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


# ========== CONFIG À TESTER ==========
CONFIGS = [
    {"chunk_size": 500, "chunk_overlap": 50},
    {"chunk_size": 1000, "chunk_overlap": 20},
    {"chunk_size": 1000, "chunk_overlap": 100},
    {"chunk_size": 1000, "chunk_overlap": 200},
    {"chunk_size": 1500, "chunk_overlap": 100},
]

QUERIES = [
    "Que faire ce week-end à Marseille ?",
    "Concerts musique jazz",
    "Événements culturels bibliothèque",
    "Sports loisirs outdoor",
]

# ========== CHARGER & PRÉPARER DONNÉES ==========
print("📥 Chargement des données...")
df = pd.read_json("evenements-publics-openagenda.json")

# Filtres
df["lastdate_begin"] = df["lastdate_begin"].astype(str).str[:10]
df = df[df["lastdate_begin"] >= date_un_an]
df = df[df["location_city"] == "Marseille"]
df = df.dropna(subset=["title_fr", "description_fr", "lastdate_begin"])

df["contenu"] = df["longdescription_fr"].fillna(df["description_fr"])
df["contenu"] = df["contenu"].apply(nettoyage_html)

print(f"✓ {len(df)} événements chargés\n")

# ========== BENCHMARKING ==========
results = []

for config_idx, config in enumerate(CONFIGS, 1):
    chunk_size = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]

    print(f"\n{'=' * 60}")
    print(f"CONFIG {config_idx} : chunk_size={chunk_size}, overlap={chunk_overlap}")
    print(f"{'=' * 60}\n")

    # Créer chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    segments = text_splitter.split_text(df["contenu"].str.cat(sep=" "))
    print(f"📊 Chunks créés : {len(segments)}")

    # Créer embeddings
    embeddings = [None] * len(segments)
    batch_size = 100

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]

        try:
            response = client.embeddings.create(model=model, inputs=batch)
            for j, data in enumerate(response.data):
                embeddings[i + j] = data.embedding
        except Exception as e:
            print(f"  ❌ Erreur batch {(i // batch_size) + 1}")

    # Créer FAISS index
    embeddings_array = np.array(
        [e for e in embeddings if e is not None], dtype="float32"
    )
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)

    print(f"✓ Index FAISS : {index.ntotal} vecteurs\n")

    # Tester avec les queries
    config_result = {
        "config": f"size={chunk_size}, overlap={chunk_overlap}",
        "num_chunks": len(segments),
        "queries": {},
    }

    for query in QUERIES:
        # Embedding de la query
        response = client.embeddings.create(model=model, inputs=[query])
        query_embedding = np.array(response.data[0].embedding, dtype="float32").reshape(
            1, -1
        )

        # Rechercher
        k = 3
        distances, indices = index.search(query_embedding, k)

        avg_distance = np.mean(distances[0])
        min_distance = np.min(distances[0])

        config_result["queries"][query] = {
            "avg_distance": float(avg_distance),
            "min_distance": float(min_distance),
            "distances": distances[0].tolist(),
        }

        print(f"  Query: {query[:50]}...")
        print(f"    Min distance: {min_distance:.4f} | Avg: {avg_distance:.4f}")

    results.append(config_result)

# ========== RAPPORT FINAL ==========
print("\n" + "=" * 60)
print("📊 RAPPORT COMPARATIF")
print("=" * 60 + "\n")

for result in results:
    print(f"CONFIG: {result['config']}")
    print(f"Nombre de chunks: {result['num_chunks']}")

    avg_min_dist = np.mean([q["min_distance"] for q in result["queries"].values()])
    avg_avg_dist = np.mean([q["avg_distance"] for q in result["queries"].values()])

    print(f"  Min distance moyen: {avg_min_dist:.4f}")
    print(f"  Avg distance moyen: {avg_avg_dist:.4f}")
    print()

# ========== RECOMMANDATION ==========
print("\n" + "=" * 60)
print("🎯 RECOMMANDATION")
print("=" * 60)

best_config = min(
    results, key=lambda x: np.mean([q["min_distance"] for q in x["queries"].values()])
)
print(f"\nMeilleure config : {best_config['config']}")
print(f"Chunks créés : {best_config['num_chunks']}")
print("\n✅ Utilise cette config pour ton RAG final !")

# Sauvegarder résultats
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n📄 Résultats sauvegardés dans benchmark_results.json")
