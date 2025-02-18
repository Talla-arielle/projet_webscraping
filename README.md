# Projet de Web Scraping et Analyse des Tendances des Réseaux Sociaux

Ce projet vise à analyser les tendances des réseaux sociaux en extrayant des commentaires de plateformes comme YouTube, en les prétraitant, en effectuant une analyse des sentiments, et en modélisant les thèmes à l'aide de techniques de traitement du langage naturel (NLP).

## Fonctionnalités

1. **Scraping des Commentaires** :
   - Extraction des commentaires YouTube à l'aide de Selenium.
   - Sauvegarde des commentaires dans un fichier CSV.

2. **Prétraitement des Données** :
   - Nettoyage des commentaires (suppression de la ponctuation, des chiffres, etc.).
   - Génération de N-grams (Bigrams, Trigrams).
   - Création de nuages de mots pour visualiser les mots les plus fréquents.

3. **Analyse des Sentiments** :
   - Utilisation d'un modèle BERT pour classer les commentaires en émotions (joie, colère, tristesse, etc.).
   - Ajout de colonnes pour le sentiment (positif/négatif) et le classement (douteux, acceptable, bon).

4. **Modélisation des Thèmes** :
   - Utilisation de BERTopic pour identifier les thèmes principaux dans les commentaires.
   - Génération de titres pour les thèmes à l'aide de Google Gemini.

## Comment Utiliser le Projet

1. **Installation des Dépendances** :
   - Installez les dépendances nécessaires en exécutant :
     ```bash
     pip install -r requirements.txt
     ```

2. **Exécution de l'Application** :
   - Lancez l'application Streamlit avec la commande :
     ```bash
     streamlit run app.py
     ```

3. **Navigation dans l'Application** :
   - **Accueil** : Présentation du projet et des contributeurs.
   - **Scraping** : Extraction des commentaires depuis YouTube.
   - **Prétraitement** : Nettoyage des données et génération de visualisations.
   - **Analyse des Sentiments** : Classification des commentaires en émotions.
   - **Topic Modeling** : Identification des thèmes principaux dans les commentaires.

## Structure du Projet

- **`app.py`** : Script principal contenant l'application Streamlit.
- **`requirements.txt`** : Liste des dépendances nécessaires.
- **`README.md`** : Documentation du projet.

## Auteurs

- Bognare Kale Evariste
- Dzogoung Talla Arielle Lareine
- Managa Previs

## Superviseur

- Mr Serge
