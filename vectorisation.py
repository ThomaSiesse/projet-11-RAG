"""
Vectorisation des chunks et gestion de l'index FAISS pour le RAG Puls-Events.

Étape 2 du pipeline :
    chunks_marseille.json  ->  chunks_marseille_cache.json  (chunks + vecteurs)
                           ->  faiss_chunks.bin             (index FAISS)

1. calcul des embeddings mistral-embed (1 024 dimensions) par lots de 100 ;
   les vecteurs déjà présents dans le cache sont réutilisés : seuls les chunks
   nouveaux ou modifiés sont envoyés à l'API (reconstruction quasi gratuite) ;
2. création d'un index FAISS exact (IndexFlatL2) ;
3. sauvegarde de l'index ; la ligne i de l'index correspond à la ligne i du cache,
   alignement que chatbot.py vérifie au démarrage.

Usage :
    python vectorisation.py                 # réutilise les vecteurs existants
    python vectorisation.py --recalculer    # recalcule tous les vecteurs (≈ 53 appels API)
    python vectorisation.py --test "concert de jazz"   # + une recherche de contrôle
"""

import argparse
import os
import time

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mistralai.client import Mistral

CHUNKS_FILE = "chunks_marseille.json"  # produit par preprocessing.py
CACHE_FILE = "chunks_marseille_cache.json"  # chunks + vecteurs (lu par chatbot.py)
INDEX_FILE = "faiss_chunks.bin"

EMBED_MODEL = "mistral-embed"  # DOIT rester le modèle utilisé par chatbot.py
DIMENSION = 1024  # taille des vecteurs mistral-embed
TAILLE_LOT = 100  # chunks par appel API
TENTATIVES = 5  # reprises d'un lot en erreur (quota, réseau)


def creer_client():
    """Crée le client Mistral à partir de la clé MISTRAL_API_KEY du fichier .env."""
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY absente : ajoutez-la dans le fichier .env.")
    return Mistral(api_key=api_key)


def charger_vecteurs_existants(chemin=CACHE_FILE):
    """
    Renvoie {texte du chunk: vecteur} à partir d'un cache précédent, ou {} s'il n'existe pas.

    La clé est le texte exact : un chunk dont le texte a changé sera recalculé.
    """
    if not os.path.exists(chemin):
        return {}
    cache = pd.read_json(chemin)
    if "embedding" not in cache.columns:
        return {}
    return dict(zip(cache["chunk_text"], cache["embedding"]))


def embed_lot(client, textes):
    """
    Calcule les vecteurs d'un lot de textes, avec reprise en cas d'erreur.

    L'attente double à chaque échec (2 s, 4 s, 8 s...) pour respecter les limites de
    débit de l'API ; après TENTATIVES échecs, l'erreur est levée et rien n'est sauvegardé.
    """
    for tentative in range(1, TENTATIVES + 1):
        try:
            reponse = client.embeddings.create(model=EMBED_MODEL, inputs=textes)
            return [d.embedding for d in reponse.data]
        except Exception as e:  # noqa: BLE001 - toute erreur API est retentée
            if tentative == TENTATIVES:
                raise
            attente = 2**tentative
            print(f"  ⚠️ Erreur API ({e}) : nouvel essai dans {attente} s")
            time.sleep(attente)


def calculer_embeddings(textes, client=None, existants=None):
    """
    Renvoie un vecteur par texte, en n'appelant l'API que pour les textes absents de `existants`.

    Le client n'est créé que si au moins un texte est à calculer : une reconstruction
    sans nouveau chunk fonctionne sans clé API.
    """
    existants = existants or {}
    a_calculer = [t for t in dict.fromkeys(textes) if t not in existants]
    print(f"{len(textes) - len(a_calculer)} vecteurs réutilisés, {len(a_calculer)} à calculer")

    nouveaux = {}
    if a_calculer:
        client = client or creer_client()
        nb_lots = (len(a_calculer) + TAILLE_LOT - 1) // TAILLE_LOT
        for n, i in enumerate(range(0, len(a_calculer), TAILLE_LOT), 1):
            lot = a_calculer[i : i + TAILLE_LOT]
            nouveaux.update(zip(lot, embed_lot(client, lot)))
            print(f"  ✓ lot {n}/{nb_lots}")

    return [existants.get(t) or nouveaux[t] for t in textes]


def creer_index(vecteurs):
    """
    Crée un index FAISS exact (IndexFlatL2) à partir d'une liste de vecteurs.

    Recherche exacte : avec quelques milliers de vecteurs, comparer la question à tous
    les chunks est instantané. Les vecteurs mistral-embed étant normalisés, la distance
    L2 au carré se convertit en similarité cosinus : cos = 1 - d / 2.
    """
    matrice = np.array(vecteurs, dtype="float32")
    if matrice.ndim != 2 or matrice.shape[1] != DIMENSION:
        raise ValueError(f"Vecteurs de forme {matrice.shape}, attendu (n, {DIMENSION})")
    index = faiss.IndexFlatL2(DIMENSION)
    index.add(matrice)
    return index


def sauvegarder_index(index, chemin=INDEX_FILE):
    """Écrit l'index FAISS sur disque."""
    faiss.write_index(index, chemin)


def charger_index(chemin=INDEX_FILE):
    """Relit un index FAISS sauvegardé par sauvegarder_index."""
    return faiss.read_index(chemin)


def rechercher(index, chunks, question, client=None, k=5):
    """
    Recherche de contrôle : affiche les k chunks les plus proches de `question`.

    Coûte un appel API (vectorisation de la question).
    """
    client = client or creer_client()
    vecteur = np.array([embed_lot(client, [question])[0]], dtype="float32")
    distances, indices = index.search(vecteur, k)
    print(f"\n🔍 {question}")
    for rang, (d, i) in enumerate(zip(distances[0], indices[0]), 1):
        row = chunks.iloc[int(i)]
        print(f"{rang}. {row['title']} | {row['date_debut']} → {row['date_fin']} "
              f"| {row['location']} (similarité {1 - d / 2:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Vectorisation des chunks + index FAISS")
    parser.add_argument("--chunks", default=CHUNKS_FILE)
    parser.add_argument("--recalculer", action="store_true", help="ignore les vecteurs existants")
    parser.add_argument("--test", metavar="QUESTION", help="recherche de contrôle après indexation")
    args = parser.parse_args()

    if not os.path.exists(args.chunks):
        raise FileNotFoundError(f"{args.chunks} introuvable : exécutez d'abord preprocessing.py.")
    chunks = pd.read_json(args.chunks)
    print(f"--- {len(chunks)} chunks à vectoriser ---")

    existants = {} if args.recalculer else charger_vecteurs_existants()
    chunks["embedding"] = calculer_embeddings(chunks["chunk_text"].tolist(), existants=existants)

    index = creer_index(chunks["embedding"].tolist())
    # Écriture seulement une fois tous les vecteurs obtenus : en cas d'échec, l'ancienne base reste intacte
    chunks.to_json(CACHE_FILE, orient="records", force_ascii=False)
    sauvegarder_index(index)
    print(f"✓ {CACHE_FILE} et {INDEX_FILE} : {index.ntotal} vecteurs de dimension {index.d}")

    if args.test:
        rechercher(index, chunks, args.test)


if __name__ == "__main__":
    main()
