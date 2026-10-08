"""
Évaluation du chatbot avec RAGAS sur le jeu de test annoté (evaluation/jeu_test.json).

Pour chaque question : le chatbot répond (conversation neuve), puis RAGAS compare
la réponse, les événements retrouvés et la réponse de référence.

Métriques RAGAS (0 à 1, plus haut = mieux, notées par un LLM juge) :
- faithfulness       : la réponse ne contient que des faits présents dans les événements retrouvés
- answer_relevancy   : la réponse répond à la question posée
- context_precision  : les événements retrouvés utiles sont bien classés en tête
- context_recall     : les événements retrouvés couvrent la réponse de référence
  (le juge est strict : un événement n'est « utile » que s'il justifie toute la référence)

Mesures exactes (sans LLM, à partir des titres annotés dans evenements_attendus) :
- rappel_recherche          : part des événements attendus retrouvés par FAISS (5 candidats max)
- precision_recommandations : part des événements recommandés qui sont attendus

Usage (depuis la racine du projet) :
    python evaluation/evaluer_ragas.py               # tout le jeu de test
    python evaluation/evaluer_ragas.py --limite 2    # essai rapide sur 2 questions

Coût : environ 3 appels API par question pour le chatbot, plus une dizaine pour RAGAS.
"""

import argparse
import json
import math
import os
import sys

# mistralai >= 2 a déplacé la classe Mistral dans mistralai.client, mais instructor
# (dépendance de RAGAS) l'importe encore depuis mistralai : on l'expose avant l'import.
import mistralai
from mistralai.client import Mistral

mistralai.Mistral = Mistral

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
from chatbot import EMBED_MODEL, creer_chatbot, formater_contexte, normaliser  # noqa: E402

DOSSIER = os.path.join(RACINE, "evaluation")
JEU_TEST = os.path.join(DOSSIER, "jeu_test.json")
MODELE_JUGE = "mistral-small-latest"


def mesures_exactes(attendus, candidats, recommandes):
    """
    Mesures sans LLM, à partir des titres annotés (evenements_attendus) :
    - rappel_recherche : part des événements attendus présents parmi les candidats FAISS
    - precision_recommandations : part des événements recommandés qui sont attendus
      (sans événement attendu : 1 si le chatbot ne recommande rien, 0 sinon)
    """
    attendus = {normaliser(t) for t in attendus}
    candidats = {normaliser(d.metadata["titre"]) for d in candidats}
    recommandes = [normaliser(d.metadata["titre"]) for d in recommandes]
    if not attendus:
        return math.nan, 0.0 if recommandes else 1.0
    rappel = len(attendus & candidats) / len(attendus)
    precision = sum(t in attendus for t in recommandes) / len(recommandes) if recommandes else 0.0
    return rappel, precision


def interroger_chatbot(questions):
    """Pose chaque question au chatbot et collecte réponse + contextes retrouvés."""
    # Chemins absolus : le script peut être lancé depuis n'importe quel dossier
    bot = creer_chatbot(
        chunks_path=os.path.join(RACINE, "chunks_marseille_cache.json"),
        index_path=os.path.join(RACINE, "faiss_chunks.bin"),
    )
    echantillons, exactes = [], []
    for q in questions:
        print(f"\n[{q['id']}] {q['question']}")
        bot.reinitialiser()  # chaque question est évaluée sans historique
        resultat = bot.demander(q["question"])
        # Un contexte par événement transmis au LLM, au même format que dans le prompt
        contextes = [formater_contexte([doc]) for doc in resultat["evenements"]]
        print(f"    {len(contextes)} événements retrouvés, {len(resultat['recommandes'])} recommandés")
        exactes.append(
            mesures_exactes(q["evenements_attendus"], resultat["evenements"], resultat["recommandes"])
        )
        echantillons.append(
            {
                "user_input": q["question"],
                "retrieved_contexts": contextes or ["(aucun événement pertinent trouvé)"],
                "response": resultat["reponse"],
                "reference": q["reference"],
            }
        )
    return echantillons, exactes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--jeu", default=JEU_TEST, help="fichier JSON du jeu de test")
    parser.add_argument("--limite", type=int, help="n'évalue que les N premières questions")
    parser.add_argument("--sortie", default=os.path.join(DOSSIER, "resultats_ragas.csv"))
    args = parser.parse_args()

    load_dotenv(os.path.join(RACINE, ".env"))
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY absente : ajoutez-la dans le fichier .env.")

    with open(args.jeu, encoding="utf-8") as f:
        questions = json.load(f)["questions"][: args.limite]

    print(f"--- 1. Réponses du chatbot ({len(questions)} questions) ---")
    echantillons, exactes = interroger_chatbot(questions)

    print("\n--- 2. Évaluation RAGAS ---")
    # Le juge est un LLM : température 0 pour des notes aussi stables que possible
    juge = LangchainLLMWrapper(
        ChatMistralAI(model=MODELE_JUGE, temperature=0, api_key=api_key, max_retries=5)
    )
    embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings(model=EMBED_MODEL, api_key=api_key))
    metriques = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),  # 1 seule question générée : moins d'appels API
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    resultats = evaluate(
        dataset=EvaluationDataset.from_list(echantillons),
        metrics=metriques,
        llm=juge,
        embeddings=embeddings,
        # Peu d'appels en parallèle : l'API Mistral limite le débit
        run_config=RunConfig(max_workers=2, max_retries=8, max_wait=60, timeout=180),
        show_progress=True,
    )

    df = resultats.to_pandas()
    df.insert(0, "id", [q["id"] for q in questions])
    df["rappel_recherche"] = [r for r, _ in exactes]
    df["precision_recommandations"] = [p for _, p in exactes]
    df.to_csv(args.sortie, index=False)

    noms = [c for c in df.columns if c not in ("id", "user_input", "retrieved_contexts", "response", "reference")]
    print("\n--- 3. Résultats ---")
    print(df[["id"] + noms].round(2).to_string(index=False))
    print("\nMoyennes :")
    for nom in noms:
        moyenne = df[nom].mean()
        print(f"  {nom:<45} {'n/a' if math.isnan(moyenne) else f'{moyenne:.2f}'}")
    print(f"\nDétail (questions, réponses, contextes) : {args.sortie}")


if __name__ == "__main__":
    main()
