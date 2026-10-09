"""
Pré-processing des événements OpenAgenda pour le RAG Puls-Events.

Étape 1 du pipeline (aucun appel API) :
    evenements-publics-openagenda.json  ->  chunks_marseille.json

1. lecture de l'export OpenAgenda ;
2. filtre géographique : ville = Marseille ET coordonnées GPS dans l'emprise de la ville
   (le code postal n'est pas fiable : « 10003 » pour la Friche, événement de Troyes
   déclaré à Marseille...) ;
3. filtre temporel : événements terminés il y a moins d'un an (ou à venir) ;
4. nettoyage : champs obligatoires présents, HTML retiré des descriptions ;
5. découpage (chunking) des descriptions avec chevauchement, chaque chunk
   conservant les métadonnées de son événement (titre, lieu, ville, dates...).

Les vecteurs sont calculés ensuite par vectorisation.py.

Usage :
    python preprocessing.py
"""

import argparse
import re
from datetime import date, timedelta

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

SOURCE_FILE = "evenements-publics-openagenda.json"
CHUNKS_FILE = "chunks_marseille.json"  # chunks + métadonnées, sans vecteurs

VILLE = "Marseille"  # périmètre géographique du POC
# Emprise approximative de la commune de Marseille (latitude, longitude)
LAT_MIN, LAT_MAX = 43.15, 43.40
LON_MIN, LON_MAX = 5.20, 5.60
DUREE_MAX_JOURS = 365  # « événements de moins d'un an »

# 1500 / 100 : meilleur compromis des benchmarks de découpage (voir le rapport technique)
TAILLE_CHUNK = 1500
CHEVAUCHEMENT = 100

CHAMPS_OBLIGATOIRES = ["title_fr", "description_fr", "firstdate_begin", "lastdate_end"]


def date_limite(aujourd_hui=None):
    """
    Renvoie la date (AAAA-MM-JJ) avant laquelle un événement terminé est jugé trop ancien.

    >>> date_limite(date(2026, 10, 9))
    '2025-10-09'
    """
    aujourd_hui = aujourd_hui or date.today()
    return (aujourd_hui - timedelta(days=DUREE_MAX_JOURS)).isoformat()


def nettoyer_html(texte):
    """
    Retire les balises HTML et les espaces multiples d'un texte.

    Une valeur manquante (None, NaN) donne une chaîne vide.

    >>> nettoyer_html("<p>Concert   <b>gratuit</b></p>")
    'Concert gratuit'

    Les balises sont supprimées sans espace : le texte des chunks doit rester identique
    d'une exécution à l'autre pour que vectorisation.py réutilise les vecteurs déjà calculés.
    """
    if pd.isna(texte):
        return ""
    texte = re.sub(r"<[^>]+>", "", str(texte))
    return re.sub(r"\s+", " ", texte).strip()


def charger_evenements(chemin=SOURCE_FILE):
    """
    Lit l'export OpenAgenda et ramène les dates de séance au format AAAA-MM-JJ.

    firstdate_begin (début de la 1re séance) et lastdate_end (fin de la dernière séance)
    sont fournis avec l'heure et le fuseau ("2026-10-10T20:00:00+02:00") : seule la date
    locale est conservée, ce qui permet de comparer les dates comme des chaînes.
    """
    df = pd.read_json(chemin)
    for col in ["firstdate_begin", "lastdate_end"]:
        df[col] = df[col].astype(str).str[:10]
    return df


def dans_marseille(coordonnees):
    """
    Indique si des coordonnées OpenAgenda ({"lat": ..., "lon": ...}) sont dans l'emprise de Marseille.

    >>> dans_marseille({"lat": 43.31, "lon": 5.39})   # Friche la Belle de Mai
    True
    >>> dans_marseille({"lat": 48.28, "lon": 4.02})   # Troyes
    False
    """
    if not isinstance(coordonnees, dict):
        return False
    lat, lon = coordonnees.get("lat"), coordonnees.get("lon")
    if lat is None or lon is None:
        return False
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def filtrer_ville(df, ville=VILLE):
    """
    Garde les événements déclarés dans `ville` (location_city, sans tenir compte de la casse)
    ET dont les coordonnées GPS sont dans l'emprise de Marseille.

    La double condition écarte les fiches incohérentes, par exemple une adresse marseillaise
    avec des coordonnées et une région situées à Troyes.
    """
    meme_ville = df["location_city"].fillna("").str.strip().str.lower() == ville.lower()
    return df[meme_ville & df["location_coordinates"].apply(dans_marseille)]


def filtrer_recents(df, aujourd_hui=None):
    """
    Garde les événements terminés il y a moins d'un an, ou pas encore terminés.

    Le filtre porte sur la date de FIN : une exposition commencée il y a plus d'un an
    mais toujours ouverte est conservée.
    """
    return df[df["lastdate_end"] >= date_limite(aujourd_hui)]


def nettoyer(df):
    """
    Écarte les événements sans titre, description ou dates, puis construit le champ
    `contenu` : description longue nettoyée du HTML, ou à défaut la description courte.
    """
    df = df.dropna(subset=CHAMPS_OBLIGATOIRES).copy()
    df["contenu"] = df["longdescription_fr"].fillna(df["description_fr"]).apply(nettoyer_html)
    return df[df["contenu"] != ""]


def decouper_en_chunks(df, taille=TAILLE_CHUNK, chevauchement=CHEVAUCHEMENT):
    """
    Découpe le contenu de chaque événement en chunks et y attache ses métadonnées.

    Le découpage récursif coupe d'abord sur les paragraphes, puis les phrases, puis les
    mots ; le chevauchement évite de perdre une information coupée entre deux chunks.

    Renvoie un DataFrame avec une ligne par chunk :
    chunk_text, original_id (index de l'événement dans l'export), title, location,
    ville, latitude, longitude, date_debut, date_fin, description.
    """
    decoupeur = RecursiveCharacterTextSplitter(chunk_size=taille, chunk_overlap=chevauchement)
    lignes = []
    for idx, row in df.iterrows():
        for chunk_text in decoupeur.split_text(row["contenu"]):
            lignes.append(
                {
                    "chunk_text": chunk_text,
                    "original_id": idx,
                    "title": row["title_fr"],
                    "location": row["location_name"],
                    "ville": row["location_city"],
                    "latitude": row["location_coordinates"]["lat"],
                    "longitude": row["location_coordinates"]["lon"],
                    "date_debut": row["firstdate_begin"],
                    "date_fin": row["lastdate_end"],
                    "description": row["description_fr"],
                }
            )
    return pd.DataFrame(lignes)


def main():
    parser = argparse.ArgumentParser(description="Pré-processing OpenAgenda -> chunks")
    parser.add_argument("--source", default=SOURCE_FILE)
    parser.add_argument("--sortie", default=CHUNKS_FILE)
    args = parser.parse_args()

    print(f"--- Lecture : {args.source} ---")
    df = charger_evenements(args.source)
    print(f"{len(df)} événements dans l'export")

    df = filtrer_ville(df)
    print(f"{len(df)} événements à {VILLE} (ville et coordonnées GPS)")

    df = filtrer_recents(df)
    print(f"{len(df)} événements terminés après le {date_limite()} (moins d'un an)")

    df = nettoyer(df)
    print(f"{len(df)} événements avec titre, description et dates")

    chunks = decouper_en_chunks(df)
    chunks.to_json(args.sortie, orient="records", force_ascii=False)
    print(f"✓ {len(chunks)} chunks sauvegardés : {args.sortie}")


if __name__ == "__main__":
    main()
