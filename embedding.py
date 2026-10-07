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
df["lastdate_begin"] = df["lastdate_begin"].astype(str).str[:10]
print(f"Avant filtre date : {len(df)} lignes")
df = df[df["lastdate_begin"] >= date_un_an]
print(f"Après filtre date : {len(df)} lignes")

# Filtre Marseille
print(f"\n--- Filtre Marseille ---")
df = df[df["location_city"] == "Marseille"]
print(f"Après filtre Marseille : {len(df)} lignes")

# Nettoyage données
print(f"\n--- Nettoyage des données ---")
df = df.dropna(subset=["title_fr", "description_fr", "lastdate_begin"])
df["contenu"] = df["longdescription_fr"].fillna(df["description_fr"])
df["contenu"] = df["contenu"].apply(nettoyage_html)
print(f"Après nettoyage : {len(df)} lignes")
print(f"\n--- Nettoyage terminé ---")

# Chunking
print(f"\n--- Début de la phase de chunck ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
)

segments = text_splitter.split_text(df["contenu"].str.cat(sep=" "))

print(f"Nombre de segments créés : {len(segments)}")
print(f"Exemple de segment : {segments[0][:200]}...")
print(f"\n--- Fin de la phase de chunck ---")

# Embeddings
print(f"\n--- Début de la phase d'embedding ---")
batch_size = 100
embeddings = []

print(f"\n--- Début de l'embedding ---")

# ✅ Boucler sur len(segments), pas len(df)
for i in range(0, len(segments), batch_size):
    batch = segments[i : i + batch_size]

    try:
        # ✅ Utiliser .create() et inputs (pas input)
        response = client.embeddings.create(model=model, inputs=batch)

        for data in response.data:
            embeddings.append(data.embedding)
        print(f"  ✓ Batch {(i // batch_size) + 1} : {len(batch)} chunks traités")

    except Exception as e:
        print(f"  ❌ Erreur batch {(i // batch_size) + 1} : {e}")
        embeddings.extend([None] * len(batch))

# ✅ Créer un dataframe des chunks (pas assigner à df)
df_chunks = pd.DataFrame({"chunk_text": segments, "embedding": embeddings})

# Vérifications
print(f"\n--- Vérifications ---")
assert len(df_chunks["embedding"]) == len(df_chunks), "Pas tous les embeddings"
assert df_chunks["embedding"].isnull().sum() == 0, "Des NULL dedans"
assert len(df_chunks["embedding"].iloc[0]) == 1024, "Mauvaise taille"
print("✓ Embeddings OK !")

# Sauvegarder
df_chunks.to_json("chunks_marseille_cache.json", orient="records", force_ascii=False)
print(f"\n--- {len(df_chunks)} chunks sauvegardés ---")
