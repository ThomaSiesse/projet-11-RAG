"""
Tests unitaires des fonctions de pré-processing et de la logique du chatbot.

Données fictives uniquement : rapides, sans fichier de données ni appel API.
"""

import doctest
from datetime import date

import pandas as pd

import preprocessing
from chatbot import (
    AnalyseQuestion,
    calendrier_reference,
    creer_filtre,
    est_generique,
    sans_ville,
    valider_periode,
)
from preprocessing import (
    dans_marseille,
    date_limite,
    decouper_en_chunks,
    filtrer_recents,
    filtrer_ville,
    nettoyer,
    nettoyer_html,
)

MERCREDI = date(2026, 10, 7)
FRICHE = {"lat": 43.31, "lon": 5.39}


def evenements(**colonnes):
    """DataFrame d'événements fictifs au format de l'export OpenAgenda."""
    base = {
        "title_fr": "Concert",
        "description_fr": "Un concert",
        "longdescription_fr": "<p>Un <b>concert</b> de jazz</p>",
        "firstdate_begin": "2026-10-10",
        "lastdate_end": "2026-10-10",
        "location_name": "Le Makeda",
        "location_city": "Marseille",
        "location_coordinates": FRICHE,
    }
    base.update(colonnes)
    n = max(len(v) if isinstance(v, list) else 1 for v in base.values())
    return pd.DataFrame({k: v if isinstance(v, list) else [v] * n for k, v in base.items()})


# --- Pré-processing ---
def test_doctests_preprocessing():
    assert doctest.testmod(preprocessing).failed == 0


def test_nettoyer_html_valeur_manquante():
    assert nettoyer_html(None) == ""
    assert nettoyer_html(float("nan")) == ""


def test_date_limite_un_an():
    assert date_limite(MERCREDI) == "2025-10-07"


def test_filtrer_recents_garde_les_evenements_en_cours():
    df = evenements(
        firstdate_begin=["2024-01-01", "2024-01-01", "2026-12-01"],
        lastdate_end=["2025-10-06", "2027-01-01", "2026-12-01"],  # trop ancien / en cours / à venir
    )
    assert filtrer_recents(df, MERCREDI)["lastdate_end"].tolist() == ["2027-01-01", "2026-12-01"]


def test_filtrer_recents_limite_incluse():
    df = evenements(lastdate_end=["2025-10-07"])
    assert len(filtrer_recents(df, MERCREDI)) == 1


def test_filtrer_ville_exige_ville_et_coordonnees():
    df = evenements(
        location_city=["Marseille", "marseille ", "Aix-en-Provence", "Marseille"],
        location_coordinates=[FRICHE, FRICHE, FRICHE, {"lat": 48.28, "lon": 4.02}],
    )
    assert filtrer_ville(df).index.tolist() == [0, 1]


def test_dans_marseille_coordonnees_absentes():
    assert not dans_marseille(None)
    assert not dans_marseille({"lat": None, "lon": 5.39})


def test_nettoyer_ecarte_les_evenements_incomplets():
    df = evenements(title_fr=["Concert", None], longdescription_fr=["<p>Jazz</p>", None])
    resultat = nettoyer(df)
    assert len(resultat) == 1
    assert resultat["contenu"].iloc[0] == "Jazz"


def test_nettoyer_utilise_la_description_courte_a_defaut():
    resultat = nettoyer(evenements(longdescription_fr=[None]))
    assert resultat["contenu"].iloc[0] == "Un concert"


def test_decouper_conserve_les_metadonnees_et_le_chevauchement():
    texte = " ".join(f"mot{i}" for i in range(400))  # ≈ 2 700 caractères -> plusieurs chunks
    df = nettoyer(evenements(longdescription_fr=[texte]))
    chunks = decouper_en_chunks(df, taille=1500, chevauchement=100)
    assert len(chunks) >= 2
    assert (chunks["chunk_text"].str.len() <= 1500).all()
    assert (chunks["title"] == "Concert").all() and (chunks["ville"] == "Marseille").all()
    assert (chunks["latitude"] == FRICHE["lat"]).all()
    fin_premier = chunks["chunk_text"].iloc[0].split()[-1]
    assert fin_premier in chunks["chunk_text"].iloc[1]  # chevauchement entre chunks consécutifs


# --- Logique du chatbot ---
def test_sans_ville_retire_marseille():
    assert sans_ville("festival Primed à Marseille") == "festival Primed"
    assert sans_ville("exposition dans Marseille") == "exposition"
    assert sans_ville("concert de jazz") == "concert de jazz"


def test_est_generique():
    assert est_generique("événements pour la semaine prochaine")
    assert not est_generique("exposition la semaine prochaine")


def test_calendrier_un_mercredi():
    cal = calendrier_reference(MERCREDI)
    assert "2026-10-10" in cal["we_debut"] and "2026-10-11" in cal["we_fin"]
    assert "2026-10-12" in cal["sp_debut"] and "2026-10-18" in cal["sp_fin"]
    assert "2026-10-31" in cal["mois_fin"]


def test_calendrier_un_dimanche_limite_le_week_end_au_jour_meme():
    cal = calendrier_reference(date(2026, 10, 11))
    assert "2026-10-11" in cal["we_debut"] and "2026-10-11" in cal["we_fin"]


def test_valider_periode_remet_les_dates_dans_l_ordre():
    analyse = AnalyseQuestion(requete="concert", date_debut="2026-12-31", date_fin="2026-12-01")
    assert valider_periode(analyse, MERCREDI) == ("2026-12-01", "2026-12-31")


def test_valider_periode_ignore_une_date_mal_formee():
    analyse = AnalyseQuestion(requete="concert", date_debut="31/12/2026", date_fin=None)
    assert valider_periode(analyse, MERCREDI) == (None, None)


def test_valider_periode_ne_cherche_pas_avant_aujourd_hui():
    analyse = AnalyseQuestion(requete="concert", date_debut="2026-01-01", date_fin="2026-12-31")
    assert valider_periode(analyse, MERCREDI) == ("2026-10-07", "2026-12-31")


def test_creer_filtre_chevauchement_de_periodes():
    filtre = creer_filtre("2026-10-10", "2026-10-11", inclure_passes=True)  # un week-end
    assert filtre({"date_debut": "2026-09-30", "date_fin": "2026-10-16"})  # exposition en cours
    assert filtre({"date_debut": "2026-10-11", "date_fin": "2026-10-11"})  # le dimanche
    assert not filtre({"date_debut": "2026-10-14", "date_fin": "2026-10-14"})  # mercredi suivant
    assert not filtre({"date_debut": "2026-09-01", "date_fin": "2026-10-09"})  # terminé avant


def test_creer_filtre_exclut_les_evenements_termines():
    filtre = creer_filtre()
    assert not filtre({"date_debut": "2020-01-01", "date_fin": "2020-01-02"})
    assert filtre({"date_debut": "2020-01-01", "date_fin": "2099-01-01"})
