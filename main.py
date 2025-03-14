import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load models and vectorizer
@st.cache_resource
def load_models():
    autoencoder = tf.keras.models.load_model("autoencoder_model.h5")
    encoder = tf.keras.models.load_model("encoder_model.h5")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    dataset = pd.read_csv("dataset-update.csv")
    tfidf_matrix = vectorizer.transform(dataset['Ingredients'])  
    return autoencoder, encoder, vectorizer, dataset, tfidf_matrix

# Load ML models & data
autoencoder, encoder, vectorizer, dataset, tfidf_matrix = load_models()

# Function to recommend recipes
def recommend_recipes(user_input, encoder, tfidf_matrix, vectorizer, data, top_n=5):
    user_input_vector = vectorizer.transform([user_input])
    user_input_encoded = encoder.predict(user_input_vector.toarray())
    tfidf_encoded = encoder.predict(tfidf_matrix.toarray())

    similarities = cosine_similarity(user_input_encoded, tfidf_encoded)[0]
    top_indices = similarities.argsort()[::-1][:top_n]
    
    recommendations = data.iloc[top_indices][['Title', 'Ingredients', 'Steps', 'URL']]
    return recommendations

# --- STREAMLIT UI ---
st.title("🍽️ Recipe Recommender System")
st.write("Enter ingredients you have, and we'll recommend recipes!")

user_input = st.text_area("Enter ingredients (comma-separated):", "")

if st.button("Find Recipes"):
    if user_input:
        with st.spinner("Finding best matches..."):
            recommended_recipes = recommend_recipes(user_input, encoder, tfidf_matrix, vectorizer, dataset, top_n=5)
            st.write("### 🍽️ Recommended Recipes")
            for _, recipe in recommended_recipes.iterrows():
                st.write(f"**🍜 {recipe['Title']}**")
                st.write(f"**Ingredients:** {recipe['Ingredients']}")
                st.write(f"[View Recipe]({recipe['URL']})")
                st.write("---")
    else:
        st.warning("Please enter at least one ingredient.")

