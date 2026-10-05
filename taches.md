# Projet_11 ** Concevoir et deployer un système RAG**
## 25/09/2026
# Cours
    BDD vectoriel = FAISS
    Framework = LongChain
    chunking = découpage des données

## 29/09/2026
### Tâches:
    - préparation de l'environnement:
        readme avec présentation, objectifs instruction pour la reproduction

## 30/09/2026
### Tâches
    -reqiurements.txt 
    - étape 1 ok
    
## 05/10/2026
### Tâches:
    embending ok
    code indexing.py

Mettre en place les branches
1. Créer la branche develop
bash
# Depuis main
git checkout main

# Créer une nouvelle branche
git checkout -b develop

# Pousser la branche
git push -u origin develop
2. Travailler sur develop
bash
# Aller sur develop
git checkout develop

# Faire tes modifications, commits
git add .
git commit -m "Modification X"

# Pousser vers develop (pas main !)
git push origin develop
3. Quand c'est prêt pour la prod
bash
# Aller sur main
git checkout main

# Fusionner develop dedans
git merge develop

# Pousser vers main (production)
git push origin main

Maintenant tu peux travailler !
Workflow quotidien :
bash
# 1. Tu es sur develop (où tu travailles)
git checkout develop

# 2. Faire des modifications
# ... éditer des fichiers ...

# 3. Vérifier l'état
git status

# 4. Ajouter et commiter
git add .
git commit -m "Ajout conversion documents"

# 5. Pousser sur develop (pas main !)
git push origin develop
Quand c'est prêt pour la production :
bash
# 1. Aller sur main
git checkout main

# 2. Fusionner develop
git merge develop

# 3. Pousser vers main
git push origin main

# 4. Revenir à develop pour continuer
git checkout develop