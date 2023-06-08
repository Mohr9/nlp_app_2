#importation des packages de base :

import pandas as pd
import numpy as np


#import package nécessaire au prétraitement de texte :

from bs4 import BeautifulSoup

# import nltk 
# from nltk.stem import WordNetLemmatizer, PorterStemmer
# from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
# from nltk.corpus import words, stopwords

#import des packages pour la prédiction :

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

#import modeles : 
from sklearn.linear_model import  SGDClassifier

#import bag of words : 
from sklearn.feature_extraction.text import CountVectorizer 

#import pour charger fichier : 
import pickle

#import package mise en page : 
from PIL import Image
import base64
import streamlit as st
# import os
# os.environ["NLTK_DATA"]="/home/appuser/nltk_data"

# nltk.download("stopwords")
# from nltk_data.corpora import wordnet
# nltk.download("wordnet")
# nltk.download("corpora")
# nltk.download('all')
# nltk.download("wordnet", "https://github.com/Mohr9/nlp_app_2/edit/master/app.py")

#### Chargement des fichiers : 
toptag = pickle.load(open("toptag.pkl","rb"))

eng_words = pickle.load(open("eng_words","rb"))

sgd = pickle.load(open("sgd","rb"))

mlb = pickle.load(open("mlb","rb"))

vect = pickle.load(open("vect","rb"))



    #Fonction pour transformer le texte de l'utilisateur en feature compatible avec notre modèle de ML:
###########################################################

def BoW(text):#le text doit etre une chaine de caractère en entrée
    sentence = vect.transform([text])#ensuite on le met dans une seule liste (et non en split)
    cv_sentence= pd.DataFrame(sentence.toarray(),columns=vect.get_feature_names_out()) #récupération du dataframe du bow
    return cv_sentence


### Fonction qui, à partir du texte rentré par l'utilisateur, va retourner une prédiction de tag :

def applying(text):
#     text = clean_balise(text) #utilisation des fonctions 2 de prétraitement de texte
#     text = cleaning(text)#ici le text devient une chaine de caractère
    text = BoW(text) # transformation du texte en feature compatible avec notre modèle de prédiction
    prediction = sgd.predict(text) #prediction du texte
    tag_pred = mlb.inverse_transform(prediction) #transformation de la target binarizée en target lisible 

    return tag_pred #affichage des tags prédit 
 



######## Mise en page ###################

####Arriere plan
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background.jpg')

### chargement de l'image :
image_url = Image.open('logo.png')
st.image(image_url, use_column_width=True)



# Titre : 
st.title("Keyword prediction tool Stackoverflow ") 


# Données entrées par l'utilisateur :
Title_input = st.text_input("Write the title of your request below")
input_body_utilisateurs = st.text_area("Enter the content of your request below ", height=200)


#Bouton de validation :
if st.button("Valider"):#si l'utilisateur appui sur valider 
    #Réponse de notre modèle : 
    reponse = applying(input_body_utilisateurs)
    st.text(reponse)



