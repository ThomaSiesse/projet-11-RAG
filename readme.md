# Assistant RAG pour Puls-Events

Ce projet implémente un assistant virtuel de recommandation d'événements culturels à Marseille, basé sur les modèles Mistral et la technique de Retrieval-Augmented Generation (RAG). Il répond aux demandes en langage naturel à partir des événements publics OpenAgenda, sans jamais inventer d'événement.

## Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS pour trouver les événements pertinents
- 📅 **Compréhension des périodes** (« ce week-end », « la semaine prochaine », « en décembre ») : seuls les événements qui ont lieu pendant la période demandée sont proposés
- 🧠 **Mémoire de conversation** : les questions de suivi (« et la semaine prochaine ? ») gardent le thème
- 🛡️ **Recommandations vérifiables** : titres, dates et lieux affichés depuis la base, jamais rédigés par le modèle ; refus des demandes hors sujet
- ⚙️ **Pipeline reproductible** : pré-processing, vectorisation et tests enchaînés en une commande
- 📊 **Évaluation** de la qualité des réponses avec RAGAS sur un jeu de test annoté

## Prérequis

- Python 3.12+
- Clé API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))

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
├── preprocessing.py            # 1. Pré-processing : filtres, nettoyage, chunking
├── vectorisation.py            # 2. Embeddings Mistral + index FAISS
├── pipeline.py                 # Enchaîne pré-processing, vectorisation et tests
├── chatbot.py                  # Chatbot RAG (LangChain : FAISS <-> Mistral)
├── CHATBOT.md                  # Analyse détaillée du chatbot + diagrammes
├── tests/                      # Tests unitaires (pytest)
│   ├── test_donnees.py         # Contrôle de la base : < 1 an, Marseille, index aligné
│   └── test_fonctions.py       # Fonctions de pré-processing et logique du chatbot
├── evaluation/                 # Évaluation de la qualité des réponses
│   ├── jeu_test.json           # 12 questions annotées (référence + événements attendus)
│   ├── evaluer_ragas.py        # Évaluation RAGAS + mesures exactes
│   └── resultats_ragas.csv     # Résultats de la dernière évaluation
├── evenements-publics-openagenda.json   # Export source (à télécharger)
├── chunks_marseille.json       # Chunks + métadonnées (créé par preprocessing.py)
├── chunks_marseille_cache.json # Chunks + vecteurs (créé par vectorisation.py)
├── faiss_chunks.bin            # Index FAISS (créé par vectorisation.py)
├── pytest.ini
└── requirements.txt
```

## Utilisation

### 1. Récupérer les données

Téléchargez l'export JSON du jeu de données [Événements publics OpenAgenda](https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/exports/json/?lang=fr&refine=location_city%3AMarseille&timezone=Europe%2FParis) filtré sur la ville de Marseille, et enregistrez-le à la racine sous le nom `evenements-publics-openagenda.json`.https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/exports/json/?lang=fr&refine=location_city%3AMarseille&timezone=Europe%2FParis

> La base déjà construite (`chunks_marseille_cache.json`, `faiss_chunks.bin`) est versionnée : sans l'export, vous pouvez directement lancer les tests et le chatbot.

### 2. Construire la base vectorielle

Exécutez le pipeline pour traiter les données, créer l'index FAISS et le contrôler :

```bash
python pipeline.py               # pré-processing -> vectorisation -> tests
python pipeline.py --tests       # tests seuls sur la base existante (aucun appel API)
python pipeline.py --recalculer  # recalcule tous les embeddings
```

Ce pipeline va :
1. Charger l'export OpenAgenda et garder les événements situés à Marseille (ville et coordonnées GPS)
2. Garder les événements terminés il y a moins d'un an ou à venir
3. Nettoyer les descriptions (HTML) et les découper en chunks de 1 500 caractères (chevauchement de 100)
4. Générer les embeddings avec Mistral (`mistral-embed`, 1 024 dimensions)
5. Créer l'index FAISS et le sauvegarder avec les chunks
6. Lancer les tests : la base n'est déclarée prête que s'ils passent tous

Le pipeline s'arrête à la première étape en échec.

### 3. Lancer le chatbot

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

### 4. Évaluer le chatbot

```bash
python evaluation/evaluer_ragas.py               # les 12 questions du jeu de test
python evaluation/evaluer_ragas.py --limite 2    # essai rapide
```

Le script pose chaque question au chatbot et enregistre les scores dans `evaluation/resultats_ragas.csv`. Coût : environ 150 appels API pour le jeu complet.

> Le jeu de test est valable jusqu'à la date indiquée dans `jeu_test.json` (`valide_jusqu_au`) : les événements terminés sortent de la recherche, il faut ensuite mettre les références à jour.

## Fonctionnalités principales

### Compréhension de la demande

Chaque question passe d'abord par une étape d'analyse (Mistral, sortie structurée) qui en extrait le thème et la période. Le calendrier (« demain », « ce week-end », « la semaine prochaine ») est calculé en Python et fourni au modèle, puis les dates renvoyées sont contrôlées. Le thème est recherché par similarité dans FAISS, la période est appliquée comme filtre sur les dates des événements.

### Recommandations sans hallucination

Le modèle ne rédige pas les fiches d'événement : il choisit des numéros dans la liste des candidats et justifie son choix en une phrase. Le titre, les dates et le lieu sont ensuite écrits par Python à partir des métadonnées. Un numéro inventé est ignoré, et le chatbot répond honnêtement quand rien ne correspond.

### Reconstruction à moindre coût

`vectorisation.py` réutilise les vecteurs déjà calculés (clé : texte du chunk) : seuls les chunks nouveaux ou modifiés sont envoyés à l'API Mistral. Une reconstruction sans nouvel événement ne fait aucun appel.

### Tests et évaluation

Les tests (pytest, aucun appel API) vérifient que la base ne contient que des événements de moins d'un an situés à Marseille, que les métadonnées sont complètes et que l'index est aligné sur les chunks. Le test « moins d'un an » compare les dates au jour de l'exécution : s'il échoue, la base est périmée et doit être reconstruite. L'évaluation RAGAS mesure en plus la qualité des réponses (fidélité, pertinence, précision et rappel du contexte).

## Modules principaux

### `preprocessing.py`

Prépare les données sources, sans appel API :
- Lecture de l'export OpenAgenda et conversion des dates
- Filtre géographique (ville et coordonnées GPS) et filtre « moins d'un an »
- Nettoyage HTML et découpage en chunks avec métadonnées

### `vectorisation.py`

Gère les embeddings et l'index vectoriel FAISS :
- Génération des embeddings avec Mistral, par lots, avec reprise en cas d'erreur
- Réutilisation des vecteurs déjà calculés
- Création, sauvegarde et chargement de l'index FAISS

### `chatbot.py`

Relie l'index FAISS au modèle de chat Mistral via LangChain :
- Analyse de la question (thème + période)
- Recherche filtrée par période, avec dédoublonnage des événements
- Génération de la réponse et mémoire de conversation

Il peut aussi être utilisé depuis un autre script (ex. Streamlit) :

```python
from chatbot import creer_chatbot

bot = creer_chatbot()
resultat = bot.demander("un concert ce week-end ?")
print(resultat["reponse"])  # texte de la réponse
print(
    resultat["recommandes"]
)  # Documents réellement recommandés (titre, dates, lieu en metadata)
print(resultat["question_recherche"])  # requête utilisée pour FAISS, ex. "concert"
print(
    resultat["periode"]
)  # ex. "du samedi 10 octobre 2026 au dimanche 11 octobre 2026"
print(resultat["evenements"])  # les 5 candidats transmis au modèle
```

Son fonctionnement interne (chaîne LangChain, filtres, choix techniques, limites) est détaillé dans [CHATBOT.md](CHATBOT.md).

### `pipeline.py`

Enchaîne la construction et le contrôle de la base :
- Pré-processing puis vectorisation
- Lancement des tests
- Arrêt à la première étape en échec

## Personnalisation

Vous pouvez personnaliser le projet en modifiant les constantes en tête de fichier :

- **`preprocessing.py`** : ville (`VILLE`) et emprise GPS, durée maximale des événements (`DUREE_MAX_JOURS`), taille des chunks et chevauchement (`TAILLE_CHUNK`, `CHEVAUCHEMENT`)
- **`vectorisation.py`** : modèle d'embedding (`EMBED_MODEL`), taille des lots (`TAILLE_LOT`), nombre de reprises (`TENTATIVES`)
- **`chatbot.py`** : modèle de chat (`CHAT_MODEL`), température (`TEMPERATURE`), nombre d'événements transmis au modèle (`NB_EVENEMENTS`), seuil de similarité (`SIMILARITE_MIN`), taille de la mémoire (`MAX_TOURS_HISTORIQUE`)

> Le modèle d'embedding doit être le même dans `vectorisation.py` et `chatbot.py`. Après un changement de modèle, de ville ou de découpage, relancez `python pipeline.py --recalculer`.
