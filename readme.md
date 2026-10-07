# **Assistant RAG pour Puls-Events**
-------------------------
Ce projet implémente un assistant virtuel basé sur le modèle Mistral, utilisant la technique de Retrieval-Augmented Generation (RAG) pour fournir des réponses précises et contextuelles à partir d'une base de connaissances personnalisée.

## **Fonctionnalités**
---------------------------
- 🔍 Recherche sémantique avec FAISS pour trouver les événements pertinents
- 💬 Chatbot conversationnel (LangChain + Mistral) qui recommande des événements à venir à Marseille
- 📅 Compréhension des périodes (« ce week-end », « la semaine prochaine », « en décembre ») : seuls les événements qui ont lieu pendant la période demandée sont proposés
- 🧠 Mémoire de conversation : les questions de suivi (« et la semaine prochaine ? ») gardent le thème
- 🛡️ Titres, dates et lieux affichés depuis la base (jamais rédigés par le modèle) : pas d'événement inventé, refus des demandes hors sujet

## **Prérequis**
 - ***Python***: 3.12 et supérieur
 - ***Clé API Mistral*** (obtenue sur console.mistral.ai)

## Installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/ThomaSiesse/projet-11-RAG.git
cd projet-11-RAG
```

2. **Créer un environnement virtuel**

```bash
# Création de l'environnement virtuel
python -m venv venv

# Activation de l'environnement virtuel
# Sur Windows
venv\Scripts\activate
# Sur macOS/Linux
source venv/bin/activate
```
3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Configurer la clé API**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```
MISTRAL_API_KEY=votre_clé_api_mistral
```
## Structure du projet

```
.
├── embedding.py       # Nettoyage, chunking et embeddings des événements
├── indexing.py        # Création de l'index FAISS
├── chatbot.py         # Chatbot RAG (LangChain : FAISS <-> Mistral)
├── CHATBOT.md         # Analyse détaillée du chatbot + diagrammes
├── requirements.txt
├── chunks_marseille_cache.json (créé par embedding.py)
├── faiss_chunks.bin   (créé par indexing.py)
```

## Utilisation

Le pipeline s'exécute dans l'ordre suivant :

```bash
python embedding.py    # 1. chunks + embeddings  -> fichier de chunks (JSON)
python indexing.py     # 2. index FAISS          -> faiss_chunks.bin
python chatbot.py      # 3. conversation
```

> Chaque chunk conserve la période de l'événement (`date_debut` = 1re séance, `date_fin` = fin de la dernière séance) :
> le chatbot ne propose que les événements pas encore terminés, y compris ceux déjà en cours.
> Après toute modification de `embedding.py`, relancez aussi `indexing.py` (le chatbot vérifie que l'index et les chunks sont alignés).

### Lancer le chatbot

```bash
python chatbot.py                                   # mode conversation
python chatbot.py -q "un atelier pour enfants ?"    # une seule question
python chatbot.py --details                         # affiche la requête reformulée et tous les candidats
python chatbot.py --modele mistral-large-latest -k 8
python chatbot.py --inclure-passes                  # inclut les événements déjà passés
```

Dans la conversation : `/reset` démarre une nouvelle conversation, `/quit` quitte.

Exemple :

```
Vous > Un concert en décembre ?

Assistant > Voici une sélection de concerts en décembre 2026 à Marseille :

1. **LUTHER - Marseille - La Plateforme**
   Le vendredi 18 décembre 2026 – La Plateforme_ - Entrée Nord
   LUTHER se produit en concert dans le cadre du SUBLIME TOUR...

2. **PACK DUO - Luther + Rounhaa - Marseille - La Plateforme**
   Du jeudi 17 décembre 2026 au vendredi 18 décembre 2026 – La Plateforme_ - Entrée Nord
   ...
   Sources :
   ✓ LUTHER - Marseille - La Plateforme | 2026-12-18 | La Plateforme_ - Entrée Nord (similarité 0.70)
   ✓ PACK DUO - Luther + Rounhaa - Marseille - La Plateforme | 2026-12-17 → 2026-12-18 | ... (similarité 0.73)

Vous > Et une exposition la semaine prochaine ?
```

### Utiliser le chatbot depuis un autre script (ex. Streamlit)

```python
from chatbot import creer_chatbot

bot = creer_chatbot()
resultat = bot.demander("un concert ce week-end ?")
print(resultat["reponse"])             # texte de la réponse
print(resultat["recommandes"])         # Documents réellement recommandés (titre, dates, lieu en metadata)
print(resultat["question_recherche"])  # requête utilisée pour FAISS, ex. "concert"
print(resultat["periode"])             # ex. "du samedi 10 octobre 2026 au dimanche 11 octobre 2026"
print(resultat["evenements"])          # les 5 candidats transmis au modèle
```

Le fonctionnement interne (chaîne LangChain, filtres, choix techniques, limites) est détaillé dans [CHATBOT.md](CHATBOT.md).
