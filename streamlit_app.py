import streamlit as st
import pickle
import gdown
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import spacy
import pandas as pd
import numpy as np
import os

# Download SpaCy model
import spacy.cli
spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')

# Load datasets
data_goa = pd.read_csv('Processed_with_MLP_Sentiment_Goa.csv')
data_manipur = pd.read_csv('Processed_with_MLP_Sentiment_Manipur.csv')
data_punjab = pd.read_csv('Processed_with_MLP_Sentiment.csv')

# Load vectorizers
with open('tfidf_vectorizer_goa.pkl', 'rb') as f:
    vectorizer_goa = pickle.load(f)
with open('tfidf_vectorizer_manipur.pkl', 'rb') as f:
    vectorizer_manipur = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer_punjab = pickle.load(f)

# Google Drive download helper
def download_model_if_needed(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)

# Download and load models
download_model_if_needed("https://drive.google.com/file/d/1MOpRLQcbqXqpE50vnDbnWh2vF7rVrPd9/view?usp=drive_link", "mlp_sentiment_model_goa.keras")
download_model_if_needed("https://drive.google.com/file/d/1J7xVS8pRKjbMpbZqwEnNfBzEIn5c8Mh_/view?usp=drive_link", "mlp_sentiment_model_manipur.keras")
download_model_if_needed("https://drive.google.com/file/d/1ZkVjrtnsZL7XujaLl4SWk5tvvMQumce9/view?usp=drive_link", "mlp_sentiment_model.keras")

mlp_model_goa = load_model("mlp_sentiment_model_goa.keras")
mlp_model_manipur = load_model("mlp_sentiment_model_manipur.keras")
mlp_model_punjab = load_model("mlp_sentiment_model.keras")

# Candidates and Parties
candidates_parties_goa = {
    'Shripad  Naik': 'BJP',
    'Francisco Sardinha': 'INC'
}
candidates_parties_manipur = {
    'N Biren Singh': 'BJP',
    'Yumnam Joykumar Singh': 'NPP',
    'Okram Ibobi Singh': 'INC'
}
candidates_parties_punjab = {
    'Bhagwant Mann': 'AAP',
    'Charanjit Singh Channi': 'INC',
    'Sukhbir Singh Badal': 'SAD'
}
parties = list(set(candidates_parties_goa.values()) |
               set(candidates_parties_manipur.values()) |
               set(candidates_parties_punjab.values()))

# App UI
st.title("🗳️ Election Winner Predictor")
st.markdown("Predict the winning candidate based on public sentiment ")

# Inputs
dataset_choice = st.selectbox("Choose State Dataset", ['Goa', 'Manipur', 'Punjab'])
user_comment = st.text_area("Enter a voter comment:")

if st.button("Predict Winner"):
    if dataset_choice.lower() == 'goa':
        data = data_goa
        vectorizer = vectorizer_goa
        mlp_model = mlp_model_goa
        candidates_parties = candidates_parties_goa
    elif dataset_choice.lower() == 'manipur':
        data = data_manipur
        vectorizer = vectorizer_manipur
        mlp_model = mlp_model_manipur
        candidates_parties = candidates_parties_manipur
    else:
        data = data_punjab
        vectorizer = vectorizer_punjab
        mlp_model = mlp_model_punjab
        candidates_parties = candidates_parties_punjab

    def identify_candidate_or_party(comment):
        doc = nlp(comment)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and ent.text in candidates_parties:
                return ent.text
            elif ent.label_ == "ORG" and ent.text in parties:
                return ent.text
        return None

    entity = identify_candidate_or_party(user_comment)

    user_comment_tfidf = vectorizer.transform([user_comment]).toarray()
    sentiment_prediction = mlp_model.predict(user_comment_tfidf)
    predicted_sentiment = int(np.argmax(sentiment_prediction))

    candidate_sentiments = {
        candidate: data[data['Translated_Comment'].str.contains(candidate)]['mlp_sentiment'].sum()
        for candidate in candidates_parties.keys()
    }
    party_sentiments = {
        party: data[data['Translated_Comment'].str.contains(party)]['mlp_sentiment'].sum()
        for party in parties
    }

    if entity:
        if entity in candidates_parties:
            candidate_sentiments[entity] += predicted_sentiment
            party_sentiments[candidates_parties[entity]] += predicted_sentiment
        elif entity in parties:
            party_sentiments[entity] += predicted_sentiment

    final_sentiments = {
        candidate: candidate_sentiments[candidate] + party_sentiments[candidates_parties[candidate]]
        for candidate in candidates_parties.keys()
    }

    predicted_winner_candidate = max(final_sentiments, key=final_sentiments.get)
    predicted_winner_party = candidates_parties[predicted_winner_candidate]

    st.subheader("Model Prediction")
    st.write(f"Detected sentiment: **{predicted_sentiment}**")
    st.success(f"Predicted Winner: **{predicted_winner_candidate}** ({predicted_winner_party})")
