import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from flask import Flask, request, jsonify

# Load models and vectorizer
autoencoder = load_model('autoencoder_model.h5')
encoder = load_model('encoder_model.h5')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load dataset
stemmed_df = pd.read_csv('dataset-update.csv')
tfidf_matrix = vectorizer.transform(stemmed_df['Ingredients'])

# Function to recommend recipes
def recommend_recipes(user_input, encoder, tfidf_matrix, vectorizer, data, top_n=None):
    # Transform input to TF-IDF vector
    user_input_tfidf = vectorizer.transform([user_input]).toarray()

    # Encode input and dataset
    user_input_encoded = encoder.predict(user_input_tfidf)
    tfidf_encoded = encoder.predict(tfidf_matrix.toarray())

    # Cosine similarity between input and all recipes
    similarities = cosine_similarity(user_input_encoded, tfidf_encoded).flatten()

    # Sort by similarity
    similar_indices = similarities.argsort()[::-1]

    # If top_n is specified, limit the number of results
    if top_n is not None:
        similar_indices = similar_indices[:top_n]

    # Get the corresponding recipes
    recommended_recipes = data.iloc[similar_indices]

    return recommended_recipes

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data['ingredients']
    recommended_recipes = recommend_recipes(user_input, encoder, tfidf_matrix, vectorizer, stemmed_df, top_n=None)
    
    # Convert the recommended recipes to a dictionary format
    recommendations = recommended_recipes[['Title', 'Ingredients', 'Steps', 'URL','unique_id']].to_dict(orient='records')
    
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
