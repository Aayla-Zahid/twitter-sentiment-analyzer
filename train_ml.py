import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocessing import clean_text

COLS = ["id", "entity", "sentiment", "text"]

print("Loading data...")
train = pd.read_csv("data/twitter_training.csv", names=COLS, header=None)
val = pd.read_csv("data/twitter_validation.csv", names=COLS, header=None)

train = train.dropna(subset=["text"]).reset_index(drop=True)
val = val.dropna(subset=["text"]).reset_index(drop=True)

print("Cleaning text...")
train["clean"] = train["text"].apply(clean_text)
val["clean"] = val["text"].apply(clean_text)

# drop rows that became empty after cleaning
train = train[train["clean"].str.len() > 0].reset_index(drop=True)
val = val[val["clean"].str.len() > 0].reset_index(drop=True)

X_train_text, y_train = train["clean"], train["sentiment"]
X_val_text, y_val = val["clean"], val["sentiment"]

print("Vectorizing with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
print("TF-IDF shape:", X_train.shape)

models = {
    "Logistic Regression": LogisticRegression(max_iter=300, n_jobs=-1, C=5),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(C=1, max_iter=3000),
}

results = {}
reports = {}
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    results[name] = acc
    reports[name] = classification_report(y_val, preds, output_dict=True)
    trained_models[name] = model
    print(f"  {name}: accuracy={acc:.4f}  ({time.time()-t0:.1f}s)")

best_name = max(results, key=results.get)
print(f"\nBest ML model: {best_name} ({results[best_name]:.4f})")

# Save vectorizer + all models + results
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/ml_models.pkl", "wb") as f:
    pickle.dump(trained_models, f)

with open("models/ml_results.json", "w") as f:
    json.dump({"accuracies": results, "best_model": best_name}, f, indent=2)

with open("models/ml_reports.json", "w") as f:
    json.dump(reports, f, indent=2)

print("\nSaved: models/tfidf_vectorizer.pkl, models/ml_models.pkl, models/ml_results.json")
