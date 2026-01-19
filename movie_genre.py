# ==============================
# MOVIE GENRE CLASSIFICATION
# FINAL GUARANTEED VERSION
# ==============================

import os
import re
import kagglehub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# -------- STEP 1: DOWNLOAD DATASET --------
base_path = kagglehub.dataset_download(
    "hijest/genre-classification-dataset-imdb"
)

data_folder = os.path.join(base_path, "Genre Classification Dataset")
train_file = os.path.join(data_folder, "train_data.txt")

print("Using file:", train_file)

# -------- STEP 2: LOAD DATA (ROBUST PARSING) --------
data = []

with open(train_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # ✅ Robust split: handles any spacing
        parts = re.split(r"\s*:::\s*", line)

        if len(parts) >= 3:
            genre = parts[1]
            plot = parts[2]
            data.append([genre, plot])

df = pd.DataFrame(data, columns=["genre", "plot"])

print("\nLoaded samples:", len(df))
print(df.head())

# -------- STEP 3: TEXT CLEANING --------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["plot"] = df["plot"].apply(clean_text)

# -------- STEP 4: TF-IDF --------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(df["plot"])
y = df["genre"]

# -------- STEP 5: SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- STEP 6: TRAIN MODEL --------
model = MultinomialNB()
model.fit(X_train, y_train)

# -------- STEP 7: EVALUATE --------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# -------- STEP 8: TEST SAMPLE --------
sample_plot = "A detective investigates a brutal murder in a dark city"
sample_plot = clean_text(sample_plot)
sample_vector = vectorizer.transform([sample_plot])
prediction = model.predict(sample_vector)

print("\nSample Plot:", sample_plot)
print("Predicted Genre:", prediction[0])

# -------- STEP 9: SAVE MODEL --------
joblib.dump(model, "movie_genre_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ MODEL SAVED SUCCESSFULLY")
