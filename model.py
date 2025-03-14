import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the stemmed dataset
stemmed_df = pd.read_csv('dataset-update.csv')

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform Ingredients
tfidf_matrix = vectorizer.fit_transform(stemmed_df['Ingredients'])

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

input_dim = tfidf_df.shape[1]
encoding_dim = 128

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='softmax')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.summary()

# Fit the autoencoder model
autoencoder.fit(tfidf_df, tfidf_df, epochs=20, batch_size=256, shuffle=True, validation_split=0.2)

# Save the models
autoencoder.save('autoencoder_model.h5')
encoder.save('encoder_model.h5')

# Save the vectorizer
import joblib
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
