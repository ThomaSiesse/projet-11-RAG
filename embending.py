import pandas as pd
import re
from mistralai.client import MistralClient
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta


load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = MistralClient(api_key=api_key)
date_un_an = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")


def nettoyer_html(texte):
    if pd.isna(texte):
        return ""
    texte = re.sub(r"<[^>]+>", "", str(texte))
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


# VÉRIFIER SI LES EMBEDDINGS MARSEILLE EXISTENT DÉJÀ
if os.path.exists("embeddings_marseille_cache.json"):
    print("✅ Embeddings Marseille trouvés en cache !")
    df = pd.read_json("evenements-publics-openagenda.json")
else:
    print("🔄 Créant les embeddings pour Marseille...")

    df = pd.read_json("evenements-publics-openagenda.json")
    df["lastdate_begin"] = df["lastdate_begin"].astype(str).str[:10]
    # Filtre date: moins d'un an
    print(f"Avant filtre date : {len(df)} lignes")
    df = df[df["lastdate_begin"] >= date_un_an]
    print(f"Après filtre date : {len(df)} lignes")

    # Nettoyage données
    df = df.dropna(subset=["title_fr", "description_fr", "lastdate_begin"])
    df["contenu"] = df["longdescription_fr"].fillna(df["description_fr"])
    df["contenu"] = df["contenu"].apply(nettoyer_html)

    # Limiter à 2000 caractères
    df["contenu_limité"] = df["contenu"].str[:2000]

    # Batches de 100
    batch_size = 100
    embeddings = []

    print(f"\n🔢 Création des embeddings par batches...")
    print(f"Total : {len(df)} événements en {(len(df) // batch_size) + 1} appels API\n")

    for i in range(0, len(df), batch_size):
        batch = df["contenu_limité"].iloc[i : i + batch_size].tolist()

        try:
            response = client.embeddings(model="mistral-embed", input=batch)

            for data in response.data:
                embeddings.append(data.embedding)

            print(f"  ✓ Batch {(i // batch_size) + 1} : {len(batch)} textes traités")

        except Exception as e:
            print(f"  ❌ Erreur batch {(i // batch_size) + 1} : {e}")
            embeddings.extend([None] * len(batch))

    df["embedding"] = embeddings

    # Colonnes finales
    colonnes_finales = [
        "title_fr",
        "description_fr",
        "contenu",
        "embedding",
        "lastdate_begin",
        "location_name",
        "location_city",
    ]

    df = df[colonnes_finales]

    # Sauvegarder MARSEILLE UNIQUEMENT
    df.to_json("embeddings_marseille_cache.json", orient="records", force_ascii=False)
    print(f"\n✅ Embeddings Marseille créés et sauvegardés !")

print(f"\n📊 {len(df)} événements Marseille prêts pour FAISS")
if df["embedding"].isna().sum() > 0:
    print(f"⚠️ ATTENTION : {df['embedding'].isna().sum()} événements sans embedding !")
else:
    print("✅ Tous les embeddings sont créés !")

# Voir un exemple
print(f"\nExemple :")
print(f"Ville : {df['location_city'].iloc[0]}")
print(f"Titre : {df['title_fr'].iloc[0]}")
print(f"Taille du vecteur : {len(df['embedding'].iloc[0])} dimensions")
