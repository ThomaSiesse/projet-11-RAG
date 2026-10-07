import pandas as pd
import re
from mistralai.client import Mistral
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Lecture de la clé API
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)
model = "mistral-embed"

# Date limite de filtration
date_un_an = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")


# Fonction de nettoyage du HTML
def nettoyage_html(texte):
    if pd.isna(texte):
        return ""
    texte = re.sub(r"<[^>]+>", "", str(texte))
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


# Lecture des données
print(f"\n--- Lecture des données ---")
df = pd.read_json("evenements-publics-openagenda.json")

# Filtre date
print(f"\n--- Filtre de la date : moins d'un an ---")
# Garde la date locale (AAAA-MM-JJ) : "2019-07-29T01:00:00+02:00" -> "2019-07-29"
df["firstdate_begin"] = df["firstdate_begin"].astype(str).str[:10]  # début 1re séance
df["lastdate_end"] = df["lastdate_end"].astype(str).str[:10]  # fin dernière séance
print(f"Avant filtre date : {len(df)} lignes")
# Filtre sur la fin : un événement commencé il y a longtemps mais encore en cours est gardé
df = df[df["lastdate_end"] >= date_un_an]
print(f"Après filtre date : {len(df)} lignes")


# Nettoyage données
print(f"\n--- Nettoyage des données ---")
df = df.dropna(subset=["title_fr", "description_fr", "firstdate_begin", "lastdate_end"])
df["contenu"] = df["longdescription_fr"].fillna(df["description_fr"])
df["contenu"] = df["contenu"].apply(nettoyage_html)
print(f"Après nettoyage : {len(df)} lignes")
print(f"\n--- Nettoyage terminé ---")

# Chunking
print(f"\n--- Début de la phase de chunck ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100,
)

segments = text_splitter.split_text(df["contenu"].str.cat(sep=" "))

print(f"Nombre de segments créés : {len(segments)}")
print(f"Exemple de segment : {segments[0][:200]}...")
print(f"\n--- Fin de la phase de chunck ---")

chunks_data = []
for idx, row in df.iterrows():
    # Chunker cette description
    chunks = text_splitter.split_text(row["contenu"])

    for chunk_text in chunks:
        chunks_data.append(
            {
                "chunk_text": chunk_text,
                "embedding": None,  # À remplir après
                "original_id": idx,
                "title": row["title_fr"],
                "location": row["location_name"],
                "date_debut": row["firstdate_begin"],
                "date_fin": row["lastdate_end"],
                "description": row["description_fr"],
            }
        )

df_chunks = pd.DataFrame(chunks_data)
print(f"✓ {len(df_chunks)} chunks créés avec métadonnées")
# Embeddings
embeddings = [None] * len(df_chunks)  # ← Pré-initialiser avec NaN
batch_size = 100

print(f"\n--- Début de l'embedding ---")

for i in range(0, len(df_chunks), batch_size):
    batch_indices = range(i, min(i + batch_size, len(df_chunks)))
    batch = df_chunks["chunk_text"].iloc[i : i + batch_size].tolist()

    try:
        response = client.embeddings.create(model=model, inputs=batch)

        for j, data in enumerate(response.data):
            embeddings[i + j] = data.embedding  # ← Remplir directement l'index

        print(f"  ✓ Batch {(i // batch_size) + 1} : {len(batch)} chunks traités")

    except Exception as e:
        print(f"  ❌ Erreur batch {(i // batch_size) + 1} : {e}")

df_chunks["embedding"] = embeddings
# Vérifications
print(f"\n--- Vérifications ---")
assert len(df_chunks["embedding"]) == len(df_chunks), "Pas tous les embeddings"
assert df_chunks["embedding"].isnull().sum() == 0, "Des NULL dedans"
assert len(df_chunks["embedding"].iloc[0]) == 1024, "Mauvaise taille"
print("✓ Embeddings OK !")

# Sauvegarder
df_chunks.to_json("chunks_marseille_cache.json", orient="records", force_ascii=False)
print(f"\n--- {len(df_chunks)} chunks sauvegardés ---")
