import time
import threading
import pandas as pd
import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import streamlit as st
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
import numpy as np
from time import sleep
import threading
import nltk
import csv
import io
import spacy
import re
import num2words
from nltk import ngrams
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
import google.generativeai as genai
from bertopic import BERTopic
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
#from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
# Télécharger les ressources nécessaires pour NLTK
nltk.download('wordnet')
nltk.download('punkt')

# Chargement du modèle de langue français
nlp = spacy.load("fr_core_news_sm")

# Variable globale pour contrôler le scraping
scraping_active = False

# Fonction pour générer des N-grams

def scrape_youtube_comments(url, file_name, chromedriver_path):
    # Configuration de Selenium pour Chrome
    options = Options()
    options.add_argument("--headless")  # Exécuter sans fenêtre du navigateur
    service = Service(chromedriver_path)

    # Lancer le navigateur avec chromedriver
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Accéder à la vidéo YouTube
        driver.get(url)
        time.sleep(2)  # Attendre que la page se charge

        # Faire défiler la page pour charger les commentaires
        body = driver.find_element(By.TAG_NAME, 'body')
        for _ in range(10):  # Faire défiler 10 fois pour charger plus de commentaires
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(1)

        # Extraire les commentaires
        comments = []
        comment_elements = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer #content-text")

        for comment_element in comment_elements:
            comments.append(comment_element.text)

        # Sauvegarder les commentaires dans un fichier CSV
        df = pd.DataFrame(comments, columns=["Commentaire"])
        df.to_csv(f"{file_name}.csv", index=False)

        st.success(f"Scraping terminé. {len(comments)} commentaires extraits.")

    except Exception as e:
        st.error(f"Une erreur s'est produite : {e}")

    finally:
        driver.quit()  # Fermer le navigateur

# Fonction principale de l'interface Streamlit
def generate_ngrams(text, n=2):
    tokens = text.split()
    ngrams_list = list(ngrams(tokens, n))
    ngram_freq = Counter(ngrams_list)
    return ngram_freq

# Visualiser les résultats des N-grams
def plot_ngrams(ngram_freq, title="N-grams Frequency", top_n=10):
    most_common = ngram_freq.most_common(top_n)
    ngrams_labels = [" ".join(ngram) for ngram, _ in most_common]
    frequencies = [freq for _, freq in most_common]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(ngrams_labels, frequencies, color='skyblue')

    # Set x-ticks and their labels
    ax.set_xticks(range(len(ngrams_labels)))  # Set the tick positions
    ax.set_xticklabels(ngrams_labels, rotation=45, ha='right')  # Set the tick labels

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("N-grams", fontsize=12)
    ax.set_ylabel("Fréquence", fontsize=12)
    
    st.pyplot(fig)

# Fonction pour générer un nuage de mots global
def generate_global_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    return wordcloud

# Nuage de mots pour les N-grams
def plot_wordcloud(ngram_freq, title="WordCloud - N-grams"):
    wordcloud_data = {" ".join(ngram): freq for ngram, freq in ngram_freq.items()}
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(wordcloud_data)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16)
    
    st.pyplot(fig)

# Fonction pour remplacer les chiffres par des mots en français
def replace_numbers_with_words(text):
    return re.sub(r"\d+", lambda x: num2words.num2words(x.group(), lang="fr"), text)

# Fonction de traitement de texte
def text_processing(text, lemmatize=False):
    text = replace_numbers_with_words(text)
    text = text.lower()
    
    # Remove unwanted characters (punctuation, special characters)
    text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation and special characters
    text = re.sub(r'\d+', '', text)  # Optionally remove numbers if not required
    
    doc = nlp(text)
    tokens = [token.text for token in doc]

    # Liste de mots à ignorer, incluant cameroun et camerounais
    stop_Wd = [
        "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et", "eux", "il", "je",
        "la", "le", "leur", "lui", "ma", "mais", "me", "même", "mes", "moi", "mon", "ne", "nos", "notre",
        "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur",
        "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos", "votre", "vous", "c", "d", "j", "l",
        "à", "m", "n", "s", "t", "y", "été", "étée", "étées", "étés", "étant", "étante", "étants", "étantes",
        "suis", "es", "est", "sommes", "êtes", "sont", "serai", "seras", "sera", "serons", "serez", "seront",
        "serais", "serait", "serions", "seriez", "seraient", "étais", "était", "étions", "étiez", "étaient",
        "fus", "fut", "fûmes", "fûtes", "furent", "sois", "soit", "soyons", "soyez", "soient", "fusse",
        "fusses", "fût", "fussions", "fussiez", "fussent",
        "le", "mr", "qu", "est", "c'est", "juste", "master", "class", "toujours", "vraiment", "si", "quand",
        "jai", "j'ai", "beaucoup", "déjà", "deja", "nest", "n'est", "franchement", "cet", "vois", "ceux",
        "ici", "mme", "quil", "qu'il", "trs", "cette", "très", "trop", "cest", "c'est", "comme", "tout",
        "plus", "bien", "faire", "aussi", "fait", "peut", "tre", "quel", "sans", "autre", "donc", "tous",
        "faut", "peu", "dit", "avoir", "non", "fois", "ans", "alors", "sont", "peu", "peux", "peut",
        "cher", "chers", "mesdames", "messieurs", "chaque", "ensemble", "ici", "aujourd'hui", "citoyens",
        "nigriens", "nigerien", "vingt", "cinq", "quatre", "année", "cameroun", "camerounais", "naja", "tv"
    ]
    
    stop_words = spacy.lang.fr.stop_words.STOP_WORDS
    stop_words.update(stop_Wd)

    tokens = [token for token in tokens if token not in stop_words and token.strip() != '']
    
    if lemmatize:
        tokens = [nlp(token)[0].lemma_ for token in tokens]

    text_propre = " ".join(tokens)
    return text_propre

# Page d'accueil
def home():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Projet de Webscraping</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #333;'>Thème : Analyse des tendances des réseaux sociaux</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #007BFF;'>Concepteurs :</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #555;'>Bognare Kale Evariste</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #555;'>Dzogoung Talla Arielle Lareine</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #555;'>Managa Previs</h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #888;'>Sous la supervision de Mr Serge</h5>", unsafe_allow_html=True)

# Fonction pour scraper les commentaires YouTube

# Page de scraping
def scraping():
    global scraping_active
    st.title("📥 Scraping des Commentaires")
    st.write("Cette page vous permet de scraper des commentaires depuis YouTube ou Facebook.")

    # Entrée de l'URL et de la plateforme
    url = st.text_input("Entrez l'URL de la vidéo YouTube ou du post Facebook:")
    platform = st.selectbox("Choisissez la plateforme:", ["YouTube", "Facebook"])
    file_name = st.text_input("Nom du fichier (sans extension):", "comments")

    if platform == "YouTube":
        chromedriver_path = st.text_input("Entrez le chemin complet vers chromedriver.exe:")

    if st.button("Scraper"):
        scraping_active = True
        if platform == "YouTube":
            if chromedriver_path and os.path.exists(chromedriver_path):
                # Lancer le scraping dans un thread séparé
                threading.Thread(target=scrape_youtube_comments, args=(url, file_name, chromedriver_path)).start()
                st.success("Scraping en cours...")
            else:
                st.error("Le chemin vers chromedriver.exe est invalide ou le fichier n'existe pas.")
        elif platform == "Facebook":
            st.warning("Le scraping de Facebook nécessite une méthode spécifique et n'est pas implémenté.")

    if st.button("Arrêter"):
        scraping_active = False
        st.success("Scraping arrêté.")

    if st.button("Télécharger le fichier"):
        if os.path.exists(f"{file_name}.csv"):
            with open(f"{file_name}.csv", "rb") as f:
                st.download_button("Télécharger", f, file_name=f"{file_name}.csv")
        else:
            st.warning("Aucun fichier à télécharger.")

# Page de prétraitement
def preprocessing():
    st.title("🔄 Prétraitement")
    st.write("Cette page est dédiée au prétraitement des données.")

    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        # Charger les données une seule fois
        if 'commentaire' not in st.session_state:
            file_content = uploaded_file.getvalue().decode('utf-8')
            file_like = io.StringIO(file_content)

            liste_comments = []
            lecteur_csv = csv.reader(file_like)
            
            for ligne in lecteur_csv:
                if any(ligne):
                    liste_comments.append(ligne)

            liste_simple = [element[0] for element in liste_comments]
            liste_filtrée = [element for element in liste_simple if element != "0"]

            st.session_state.commentaire = pd.DataFrame(liste_filtrée, columns=['Commentaire'])
        
        st.write("Aperçu des données chargées :")
        st.dataframe(st.session_state.commentaire.head())

        lemmatize = st.radio("Voulez-vous effectuer la lemmatisation des commentaires ?", ('Oui', 'Non'))

        if st.button("Nettoyer les données"):
            st.write("Traitement en cours...")
            st.session_state.commentaire['Commentaire'] = st.session_state.commentaire['Commentaire'].apply(
                lambda x: text_processing(x, lemmatize=lemmatize == 'Oui')
            )
            st.write("Données nettoyées :")
            st.dataframe(st.session_state.commentaire.head())

            # Stocker les commentaires nettoyés dans session_state
            st.session_state.comments_list = st.session_state.commentaire["Commentaire"].tolist()

        # Afficher les options pour les N-grams et les nuages de mots
        if 'comments_list' in st.session_state:
            n_choice = st.radio("Choisissez le type de N-grams :", ('Bigrams', 'Trigrams'))

            # Définir la valeur de n en fonction du choix de l'utilisateur
            n_value = 2 if n_choice == 'Bigrams' else 3

            if st.button("Générer N-grams"):
                ngram_freq = generate_ngrams(" ".join(st.session_state.comments_list), n=n_value)

                # Afficher les N-grams sous forme de graphique
                plot_ngrams(ngram_freq, title=f"Top {n_value}-grams Frequency")

                # Générer et afficher le nuage de mots pour les N-grams
                plot_wordcloud(ngram_freq, title=f"Nuage de mots - {n_value}-grams")

            # Générer et afficher le nuage de mots global
            if st.button("Générer le nuage de mots global"):
                global_wordcloud = generate_global_wordcloud(" ".join(st.session_state.comments_list))
                st.subheader("Nuage de mots global")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(global_wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

            # Téléchargement du fichier nettoyé
            csv_output = st.session_state.commentaire.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger le fichier CSV nettoyé",
                data=csv_output,
                file_name='commentaires_nettoyes.csv',
                mime='text/csv',
            )


# Ajouter les colonnes sentiment et classement
def ajouter_colonnes_sentiment_classement(df):
    """
    Ajoute deux colonnes au DataFrame :
    - 'sentiment' : 'positif' si le label est 'joy' ou 'love', sinon 'négatif'.
    - 'classement' : 'douteux' si score < 0.5, 'acceptable' si 0.5 <= score < 0.7, 'bon' si score >= 0.7.
    """
    # Colonne 'sentiment'
    df['sentiment'] = df['label'].apply(lambda x: 'positif' if x in ['joy', 'love'] else 'négatif')
    
    # Colonne 'classement'
    df['classement'] = df['score'].apply(
        lambda x: 'douteux' if x < 0.5 else ('acceptable' if x < 0.7 else 'bon')
    )
    
    return df

# Page d'analyse des sentiments

def sentiment_analysis():
    st.title("🔍 Analyse des Sentiments")
    
    
    # Demande de chemin pour enregistrer le modèle
    model_path = st.text_input("Entrez le chemin pour enregistrer le modèle (ex: C:/Users/FTAB TECH/tp4_nlp/modele):")

    if st.button("Télécharger et enregistrer le modèle"):
        try:
            # Charger le modèle et le tokenizer
            model = AutoModelForSequenceClassification.from_pretrained('bhadresh-savani/bert-base-uncased-emotion')
            tokenizer = AutoTokenizer.from_pretrained('bhadresh-savani/bert-base-uncased-emotion')

            # Enregistrer le modèle et le tokenizer
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            st.success("Modèle et tokenizer téléchargés et enregistrés avec succès !")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement du modèle : {e}")

    # File uploader for CSV file containing comments
    uploaded_file = st.file_uploader("Téléchargez votre fichier CSV contenant les commentaires pour l'analyse :", type="csv")
    
    if st.button("Charger le modèle et analyser les sentiments"):
        if uploaded_file is not None:
            try:
                # Charger le modèle et le tokenizer depuis le chemin spécifié
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)

                # Créer un pipeline pour l'analyse de texte
                classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

                # Lire le fichier CSV dans un DataFrame
                commentaires = pd.read_csv(uploaded_file)

                if 'Commentaire' in commentaires.columns:
                    # Appliquer le modèle aux commentaires
                    commentaires['Commentaire'] = commentaires['Commentaire'].fillna("")  # Remplacer NaN par des chaînes vides
                    commentaires['Commentaire'] = commentaires['Commentaire'].astype(str)  # S'assurer que la colonne est de type string

                    # Appliquer la fonction à chaque ligne du DataFrame
                    def analyser_emotion_batch(textes):
                        resultats = classifier(textes, truncation=True, max_length=512)
                        return [(res['label'], res['score']) for res in resultats]

                    # Appliquer la fonction à chaque ligne du DataFrame
                    resultats = analyser_emotion_batch(commentaires['Commentaire'].tolist())  # Passer les commentaires en liste

                    # Créer un DataFrame avec les résultats
                    sentiments_df = pd.DataFrame(resultats, columns=['label', 'score'])

                    # Ajouter les résultats au DataFrame original
                    commentaires[['label', 'score']] = sentiments_df[['label', 'score']]

                    # Ajouter les colonnes 'sentiment' et 'classement'
                    commentaires = ajouter_colonnes_sentiment_classement(commentaires)

                    # Stocker les résultats dans st.session_state
                    st.session_state.commentaires = commentaires

                    # Afficher le DataFrame avec les émotions et les scores
                    st.write("Résultats de l'analyse des sentiments :")
                    st.dataframe(commentaires.head())  # Afficher le DataFrame avec les résultats

                else:
                    st.error("Le fichier CSV doit contenir une colonne nommée 'Commentaire'.")
            except Exception as e:
                st.error(f"Erreur lors de l'analyse des sentiments : {e}")
        else:
            st.warning("Veuillez d'abord télécharger votre fichier CSV.")

    # Afficher les visualisations si les résultats sont disponibles
    if 'commentaires' in st.session_state:
        commentaires = st.session_state.commentaires

        if st.button("Afficher la répartition des sentiments"):
            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 5))
            sns.countplot(x=commentaires['sentiment'], palette='viridis')
            plt.title("Répartition des sentiments (positif/négatif)")
            plt.ylabel("Nombre d'occurrences")
            plt.xlabel("Sentiment")
            st.pyplot(plt)

        if st.button("Afficher la répartition des emotions"):
            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 5))
            sns.countplot(x=commentaires['label'], palette='viridis')
            plt.title("Répartition des emotions")
            plt.ylabel("Nombre d'occurrences")
            plt.xlabel("Emotions")
            st.pyplot(plt)

        if st.button("Afficher la répartition des classements"):
            sns.set_style("whitegrid")
            plt.figure(figsize=(10, 5))
            sns.countplot(x=commentaires['classement'], palette='viridis')
            plt.title("Répartition des classements (douteux, acceptable, bon)")
            plt.ylabel("Nombre d'occurrences")
            plt.xlabel("Classement")
            st.pyplot(plt)

    def generer_nuage_mots(text, titre):
        if text.strip():  # Vérifier si le texte n'est pas vide
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(titre)
            plt.axis('off')
            st.pyplot(plt)  # Utiliser st.pyplot() pour afficher dans Streamlit
        else:
            st.warning(f"Aucun mot à afficher pour le nuage de mots de l'émotion '{titre}'.")


        
    
    if 'commentaires' in st.session_state:
        commentaires = st.session_state.commentaires
        
        

        
            # Ajouter les résultats du modèle à chaque commentaire (ici, un modèle fictif est utilisé pour l'exemple)
            # Dans la vraie implémentation, vous appliquez le modèle comme dans votre code précédent
            # Par exemple : commentaires[['label', 'score']] = résultats obtenus avec votre pipeline de classification

            # Exemples d'émotions basées sur des labels fictifs, remplacer par les résultats de votre modèle
            # Simulation des labels
        #commentaires['label'] = ['anger', 'joy', 'love', 'fear', 'sadness'] * (len(commentaires) // 5)  # Exemple

            # Sélectionner les commentaires par émotion
        df_anger = commentaires[commentaires['label'] == 'anger']['Commentaire']
        df_joy = commentaires[commentaires['label'] == 'joy']['Commentaire']
        df_love = commentaires[commentaires['label'] == 'love']['Commentaire']
        df_fear = commentaires[commentaires['label'] == 'fear']['Commentaire']
        df_sadness = commentaires[commentaires['label'] == 'sadness']['Commentaire']

            # Créer des textes à partir des commentaires de chaque émotion
        texte_anger = ' '.join(df_anger)
        texte_joy = ' '.join(df_joy)
        texte_love = ' '.join(df_love)
        texte_fear = ' '.join(df_fear)
        texte_sadness = ' '.join(df_sadness)

            # Liste déroulante pour choisir l'émotion à afficher
        emotion_choisie = st.selectbox("Choisissez une émotion pour afficher le nuage de mots :", 
                                          ["anger", "joy", "love", "fear", "sadness"])

            # Afficher le nuage de mots pour l'émotion sélectionnée
        if st.button("Générer le nuage de mots"):
                if emotion_choisie == "anger":
                    generer_nuage_mots(texte_anger, "Nuage de mots pour les commentaires 'anger'")
                elif emotion_choisie == "joy":
                    generer_nuage_mots(texte_joy, "Nuage de mots pour les commentaires 'joy'")
                elif emotion_choisie == "love":
                    generer_nuage_mots(texte_love, "Nuage de mots pour les commentaires 'love'")
                elif emotion_choisie == "fear":
                    generer_nuage_mots(texte_fear, "Nuage de mots pour les commentaires 'fear'")
                elif emotion_choisie == "sadness":
                    generer_nuage_mots(texte_sadness, "Nuage de mots pour les commentaires 'sadness'")

        
        
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """
    Charge un fichier CSV et prétraite les commentaires.
    """
    try:
        df = pd.read_csv(uploaded_file)
        if 'Commentaire' not in df.columns:
            st.error("Le fichier CSV doit contenir une colonne nommée 'Commentaire'.")
            return None
        comments = df['Commentaire'].dropna().tolist()  # Supprime les valeurs manquantes
        comments = [comment for comment in comments if comment]  # Nettoie chaque commentaire
        return comments
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
        return None

# Fonction pour générer les embeddings
@st.cache_resource
def generate_embeddings(comments):
    """
    Génère des embeddings pour les commentaires à l'aide de SentenceTransformer.
    """
    try:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = model.encode(comments)  # Encode les commentaires en embeddings
        return embeddings
    except Exception as e:
        st.error(f"Erreur lors de la génération des embeddings : {e}")
        return None

# Fonction pour appliquer PCA
@st.cache_resource
def apply_pca(embeddings, n_components=2):
    """
    Applique une réduction de dimensionnalité avec PCA.
    """
    try:
        pca = PCA(n_components=n_components)
        embeddings_pca = pca.fit_transform(embeddings)  # Applique PCA aux embeddings
        return embeddings_pca
    except Exception as e:
        st.error(f"Erreur lors de l'application de PCA : {e}")
        return None

# Fonction pour générer les titres des thèmes avec LangChain

# Fonction pour générer les titres des thèmes avec LangChain
# Fonction pour générer les titres des thèmes avec LangChain

# Fonction pour générer les titres des thèmes avec Google Gemini
def generate_topic_titles(topic_info, num_topics, gemini_api_key):
    """
    Génère des titres pour les thèmes à l'aide de Google Gemini.
    """
    try:
        # Configurer l'API Gemini
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-pro')

        prompt = """
        Tu es un agent qui permet de donner un titre à un ensemble de topics que l'utilisateur te donnera. Donne uniquement le titre comme réponse.

        Topics :
        {topics}
        """
        titles = []
        for i in range(1, num_topics + 1):
            # Vérifiez si la colonne 'Representative_Docs' existe, sinon utilisez une autre colonne
            if 'Representative_Docs' in topic_info.columns:
                topics_text = " ".join(topic_info['Representative_Docs'].iloc[i - 1])
            elif 'Representation' in topic_info.columns:
                topics_text = " ".join(topic_info['Representation'].iloc[i - 1])
            else:
                st.error("Colonne 'Representative_Docs' ou 'Representation' introuvable dans topic_info.")
                return None

            final_prompt = prompt.format(topics=topics_text)
            try:
                response = model.generate_content(final_prompt)  # Utilisation de Gemini
                titles.append(response.text)
            except Exception as e:
                st.error(f"Erreur lors de la génération des titres des thèmes : {e}")
                return None
        return titles
    except Exception as e:
        st.error(f"Erreur lors de la génération des titres des thèmes : {e}")
        return None

# Page de Topic Modeling
def topic_modeling():
    st.title("🧠 Modélisation des Thèmes")

    # Étape 1 : Téléchargement du fichier CSV
    uploaded_file = st.file_uploader("Téléchargez votre fichier CSV contenant les commentaires :", type="csv")
    if uploaded_file is not None:
        if st.button("Valider le fichier CSV"):
            comments = load_and_preprocess_data(uploaded_file)
            if comments:
                st.session_state.comments = comments  # Sauvegarde les commentaires dans session_state
                st.success("Fichier CSV validé et commentaires prétraités avec succès !")

    # Étape 2 : Génération des embeddings
    if 'comments' in st.session_state:
        if st.button("Générer les embeddings"):
            embeddings = generate_embeddings(st.session_state.comments)
            if embeddings is not None:
                st.session_state.embeddings = embeddings  # Sauvegarde les embeddings dans session_state
                st.success("Embeddings générés avec succès !")

    # Étape 3 : Paramétrage et application de PCA
    if 'embeddings' in st.session_state:
        st.sidebar.header("Paramètres de modélisation")
        min_topic_size = st.sidebar.slider("Taille minimale des thèmes", min_value=2, max_value=20, value=5)
        n_components = st.sidebar.slider("Nombre de composantes pour PCA", min_value=2, max_value=10, value=2)
        num_topics = st.sidebar.number_input("Nombre de thèmes à afficher", min_value=1, max_value=50, value=5)

        if st.button("Appliquer PCA et modéliser les thèmes"):
            embeddings_pca = apply_pca(st.session_state.embeddings, n_components=n_components)
            if embeddings_pca is not None:
                st.session_state.embeddings_pca = embeddings_pca  # Sauvegarde les embeddings PCA dans session_state

                # Initialisation de BERTopic
                stopwords_fr = ["le", "la", "les", "un", "une", "des", "du", "de", "dans", "sur", 
                               "pour", "avec", "par", "au", "aux", "ce", "ces", "ça", "elle", 
                               "il", "ils", "elles", "nous", "vous", "on", "en", "et", "ou", "mais"]
                vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords_fr)
                topic_model = BERTopic(language="french", vectorizer_model=vectorizer_model, min_topic_size=min_topic_size)

                # Ajuster le modèle BERTopic
                topics, probs = topic_model.fit_transform(st.session_state.comments, st.session_state.embeddings_pca)
                st.session_state.topic_model = topic_model  # Sauvegarde le modèle dans session_state
                st.success("PCA appliquée et thèmes modélisés avec succès !")

                # Afficher les thèmes générés
                topic_info = topic_model.get_topic_info()
                st.write("Informations sur les thèmes générés :")
                st.dataframe(topic_info)

                # Génération des titres des thèmes avec Google Gemini
                gemini_api_key = st.text_input("Entrez votre clé API Google Gemini :", type="password")
                if gemini_api_key and st.button("Générer les titres des thèmes"):
                    titles = generate_topic_titles(topic_info, num_topics, gemini_api_key)
                    if titles:
                        for i, title in enumerate(titles):
                            st.write(f"**Titre pour le thème {i+1}:** {title}")















    # Étape 4 : Affichage des thèmes et visualisation
    if 'topic_model' in st.session_state:
        topic_info = st.session_state.topic_model.get_topic_info()
        st.write("Informations sur les thèmes générés :")
        st.dataframe(topic_info)

        if len(topic_info) > 1:  # Vérifie si des thèmes ont été générés
            
            num_topics = min(num_topics, len(topic_info) - 1)  # Limite le nombre de thèmes affichés

            # Visualisation des thèmes avec Plotly
            st.subheader("Visualisation des thèmes")
            fig = st.session_state.topic_model.visualize_barchart(topics=range(1,num_topics))
            st.plotly_chart(fig)

            # Génération des titres des thèmes avec LangChain
            openai_api_key = st.text_input("Entrez votre clé API OpenAI :", type="password")
            if openai_api_key and st.button("Générer les titres des thèmes"):
                titles = generate_topic_titles(topic_info, num_topics, openai_api_key)
                if titles:
                    for i, title in enumerate(titles):
                        st.write(f"**Titre pour le thème {i+1}:** {title}")
        else:
            st.warning("Aucun thème significatif n'a été généré. Essayez d'ajuster les paramètres ou de fournir plus de données.")


# Configuration de la page
st.set_page_config(page_title="Projet Webscraping", layout="wide")

# Fonctionnalités dans la barre latérale
st.sidebar.title("Fonctionnalités")

if 'page' not in st.session_state:
    st.session_state.page = "home"

if st.sidebar.button("Accueil"):
    st.session_state.page = "home"
if st.sidebar.button("Scraping"):
    st.session_state.page = "scraping"
if st.sidebar.button("Prétraitement"):
    st.session_state.page = "preprocessing"
if st.sidebar.button("Analyse des Sentiments"):
    st.session_state.page = "sentiment_analysis"
if st.sidebar.button("Topic Modeling"):
    st.session_state.page = "topic_modeling"

# Affichage de la page correspondante
if st.session_state.page == "home":
    home()
elif st.session_state.page == "scraping":
    scraping()
elif st.session_state.page == "preprocessing":
    preprocessing()
elif st.session_state.page == "sentiment_analysis":
    sentiment_analysis()
elif st.session_state.page == "topic_modeling":
    topic_modeling()