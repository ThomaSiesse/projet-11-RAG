import pandas as pd
import re
from mistralai.client import Mistral  # ✅ BON IMPORT 0.4.2
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

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

# Filtre date: moins d'un an
print(f"\n--- Filtre de la date : moins d'un an ---")
df["lastdate_begin"] = df["lastdate_begin"].astype(str).str[:10]
print(f"Avant filtre date : {len(df)} lignes")
df = df[df["lastdate_begin"] >= date_un_an]
print(f"Après filtre date : {len(df)} lignes")

# Filtre Marseille
print(f"\n--- Filtre Marseille ---")
print(f"Avant filtre ville : {len(df)} lignes")
df = df[df["location_city"] == "Marseille"]
print(f"Après filtre Marseille : {len(df)} lignes")

# Nettoyage données
print(f"\n--- Nettoyage des données ---")
df = df.dropna(subset=["title_fr", "description_fr", "lastdate_begin"])
df["contenu"] = df["longdescription_fr"].fillna(df["description_fr"])
df["contenu"] = df["contenu"].apply(nettoyage_html)
df["contenu_limité"] = df["contenu"].str[:2000]  # Limiter à 2000 chars
print(f"Après nettoyage : {len(df)} lignes")
print(f"\n--- Nettoyage terminé ---")

print(f"\n--- Début de la phase d'embedding ---")

batch_size = 100
embeddings = []

print(f"\n--- Début de l'embedding ---")
# Obtention des embeddings avec Mistral
for i in range(0, len(df), batch_size):
    batch = df["contenu_limité"].iloc[i : i + batch_size].tolist()

    try:
        response = client.embeddings.create(model="mistral-embed", inputs=batch)

        for data in response.data:
            embeddings.append(data.embedding)  # ✅ .embedding (pas .embeddings)

        print(f"  ✓ Batch {(i // batch_size) + 1} : {len(batch)} textes traités")

    except Exception as e:
        print(f"  ❌ Erreur batch {(i // batch_size) + 1} : {e}")
        embeddings.extend([None] * len(batch))

df["embedding"] = embeddings  # ✅ "embedding" pas "embeddings"

# Vérifications
print(f"\n--- Vérifications ---")
assert len(df["embedding"]) == len(df), "Pas tous les embeddings"
print(f"✓ {len(df['embedding'])} embeddings créés")

assert df["embedding"].isnull().sum() == 0, "Des NULL dedans"
print(f"Aucun NULL détecté")

assert len(df["embedding"].iloc[0]) == 1024, "Mauvaise taille"
print(f" Taille correcte (1024 dimensions)")

print(" Tous les embeddings OK !")
print(f"\n--- Embedding terminé ---")

# Colonnes finales
colonnes_finales = [
    "title_fr",
    "description_fr",
    "contenu",
    "embedding",  # ✅ "embedding"
    "lastdate_begin",
    "location_name",
    "location_city",
]

df = df[colonnes_finales]

# Sauvegarder
df.to_json("embeddings_marseille_cache.json", orient="records", force_ascii=False)
print(f"\n---Fichier: embeddings_marseille_cache.json ---")
