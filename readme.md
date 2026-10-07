# **Assistant RAG pour Puls-Events**
-------------------------
Ce projet implémente un assistant virtuel basé sur le modèle Mistral, utilisant la technique de Retrieval-Augmented Generation (RAG) pour fournir des réponses précises et contextuelles à partir d'une base de connaissances personnalisée.

## **Fonctionnalités**
---------------------------
🔍 Recherche sémantique avec FAISS pour trouver les évènementspertinents

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
├── embeddings.py    # Créer les vecteur
├── indexing.py      # créer l'index 
├── requirements.txt
├── embeddings_marseille_cache.json
├── faiss_marseille.bin   (créé par indexing.py)
├── metadata_marseille.json (créé par indexing.py)
```
