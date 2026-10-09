"""
Tests de la base vectorielle : les données indexées respectent le périmètre du POC.

- événements de moins d'un an (date de fin >= aujourd'hui - 365 jours) ;
- situés à Marseille (ville déclarée et coordonnées GPS) ;
- métadonnées complètes et cohérentes ;
- index FAISS aligné sur les chunks.

Ces tests lisent les fichiers produits par le pipeline et n'appellent pas l'API Mistral.
Un échec sur l'ancienneté signifie que la base doit être reconstruite : python pipeline.py
"""

from datetime import date
from pathlib import Path

import faiss
import pandas as pd
import pytest

from preprocessing import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, VILLE, date_limite
from vectorisation import DIMENSION

RACINE = Path(__file__).resolve().parent.parent
CACHE = RACINE / "chunks_marseille_cache.json"
INDEX = RACINE / "faiss_chunks.bin"


@pytest.fixture(scope="module")
def chunks():
    """Chunks indexés (lus une seule fois pour tout le module)."""
    if not CACHE.exists():
        pytest.fail(f"{CACHE.name} introuvable : exécutez python pipeline.py")
    return pd.read_json(CACHE, convert_dates=False)


@pytest.fixture(scope="module")
def index():
    if not INDEX.exists():
        pytest.fail(f"{INDEX.name} introuvable : exécutez python pipeline.py")
    return faiss.read_index(str(INDEX))


def evenements_en_echec(chunks, masque):
    """Liste lisible (titre, valeur) des chunks qui ne respectent pas une condition."""
    return chunks.loc[~masque, ["title", "date_fin", "ville"]].drop_duplicates().head(5).to_dict("records")


# --- Périmètre temporel ---
def test_evenements_de_moins_d_un_an(chunks):
    limite = date_limite(date.today())
    recents = chunks["date_fin"] >= limite
    assert recents.all(), (
        f"{(~recents).sum()} chunks terminés avant le {limite} : reconstruisez la base. "
        f"Exemples : {evenements_en_echec(chunks, recents)}"
    )


def test_dates_au_format_iso(chunks):
    for col in ["date_debut", "date_fin"]:
        valides = chunks[col].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}")
        assert valides.all(), f"{col} mal formée : {evenements_en_echec(chunks, valides)}"


def test_debut_avant_fin(chunks):
    coherent = chunks["date_debut"] <= chunks["date_fin"]
    assert coherent.all(), f"Début après la fin : {evenements_en_echec(chunks, coherent)}"


# --- Périmètre géographique ---
def test_ville_marseille(chunks):
    a_marseille = chunks["ville"].str.strip().str.lower() == VILLE.lower()
    assert a_marseille.all(), f"Hors {VILLE} : {evenements_en_echec(chunks, a_marseille)}"


def test_coordonnees_dans_marseille(chunks):
    dedans = chunks["latitude"].between(LAT_MIN, LAT_MAX) & chunks["longitude"].between(LON_MIN, LON_MAX)
    assert dedans.all(), f"Coordonnées hors de Marseille : {evenements_en_echec(chunks, dedans)}"


# --- Qualité des métadonnées ---
@pytest.mark.parametrize("champ", ["chunk_text", "title", "location", "description"])
def test_champs_obligatoires_remplis(chunks, champ):
    rempli = chunks[champ].fillna("").astype(str).str.strip() != ""
    assert rempli.all(), f"{(~rempli).sum()} chunks sans {champ}"


# --- Index vectoriel ---
def test_index_aligne_sur_les_chunks(chunks, index):
    assert index.ntotal == len(chunks), (
        f"Index ({index.ntotal} vecteurs) et chunks ({len(chunks)}) désalignés : "
        "relancez python vectorisation.py"
    )


def test_dimension_des_vecteurs(chunks, index):
    assert index.d == DIMENSION
    assert chunks["embedding"].map(len).eq(DIMENSION).all()


def test_index_retrouve_ses_propres_vecteurs(chunks, index):
    """
    Le vecteur du chunk i doit retrouver, à distance nulle, un chunk de même texte
    (le chunk i lui-même, ou un doublon exact : un événement publié plusieurs fois).
    """
    import numpy as np

    lignes = [0, len(chunks) // 2, len(chunks) - 1]
    vecteurs = np.array(chunks["embedding"].iloc[lignes].tolist(), dtype="float32")
    distances, voisins = index.search(vecteurs, 1)
    assert np.allclose(distances[:, 0], 0, atol=1e-4)
    textes = chunks["chunk_text"]
    assert [textes.iloc[v] for v in voisins[:, 0]] == [textes.iloc[i] for i in lignes]
