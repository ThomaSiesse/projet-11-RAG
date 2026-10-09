"""
Pipeline de construction de la base vectorielle du RAG Puls-Events.

Enchaîne les trois étapes et s'arrête à la première qui échoue :

    1. preprocessing.py   export OpenAgenda -> chunks (filtre Marseille, < 1 an, nettoyage)
    2. vectorisation.py   chunks -> embeddings Mistral + index FAISS
    3. pytest             contrôle de la base (périmètre, métadonnées, index) et des fonctions

La base n'est déclarée prête que si tous les tests passent.

Usage :
    python pipeline.py               # reconstruction complète + tests
    python pipeline.py --tests       # tests seuls, sur la base existante (aucun appel API)
    python pipeline.py --recalculer  # recalcule tous les embeddings (≈ 53 appels API)
"""

import argparse
import subprocess
import sys
import time


def etape(numero, titre, commande):
    """Exécute une commande avec le Python de l'environnement courant ; arrête le pipeline si elle échoue."""
    print(f"\n{'=' * 70}\n[{numero}/3] {titre}\n{'=' * 70}", flush=True)
    debut = time.time()
    resultat = subprocess.run([sys.executable, *commande])
    if resultat.returncode != 0:
        sys.exit(f"\n❌ Échec de l'étape {numero} ({titre}) : pipeline arrêté.")
    print(f"✓ {titre} ({time.time() - debut:.0f} s)")


def main():
    parser = argparse.ArgumentParser(description="Construit et contrôle la base vectorielle")
    parser.add_argument("--tests", action="store_true", help="lance seulement les tests")
    parser.add_argument("--recalculer", action="store_true", help="recalcule tous les embeddings")
    args = parser.parse_args()

    if not args.tests:
        etape(1, "Pré-processing", ["preprocessing.py"])
        etape(2, "Vectorisation et index FAISS",
              ["vectorisation.py"] + (["--recalculer"] if args.recalculer else []))
    etape(3, "Tests", ["-m", "pytest", "-v", "-p", "no:warnings"])

    print("\n✅ Base prête : lancez python chatbot.py")


if __name__ == "__main__":
    main()
