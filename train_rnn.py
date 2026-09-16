import json
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocessing import clean_text

tf.random.set_seed(42)
np.random.seed(42)

COLS = ["id", "entity", "sentiment", "text"]
MAX_WORDS = 20000
MAX_LEN = 40
EMBED_DIM = 100

print("Loading data...")
train = pd.read_csv("data/twitter_training.csv", names=COLS, header=None)
val = pd.read_csv("data/twitter_validation.csv", names=COLS, header=None)
train = train.dropna(subset=["text"]).reset_index(drop=True)
val = val.dropna(subset=["text"]).reset_index(drop=True)

print("Cleaning text...")
train["clean"] = train["text"].apply(clean_text)
val["clean"] = val["text"].apply(clean_text)
train = train[train["clean"].str.len() > 0].reset_index(drop=True)
val = val[val["clean"].str.len() > 0].reset_index(drop=True)

le = LabelEncoder()
y_train = le.fit_transform(train["sentiment"])
y_val = le.transform(val["sentiment"])
num_classes = len(le.classes_)
print("Classes:", list(le.classes_))

print("Tokenizing...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train["clean"])

X_train_seq = tokenizer.texts_to_sequences(train["clean"])
X_val_seq = tokenizer.texts_to_sequences(val["clean"])

X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_val = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding="post", truncating="post")

print("X_train shape:", X_train.shape)

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM),
    SimpleRNN(128),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

t0 = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=512,
    callbacks=[early_stop],
    verbose=2,
)
print(f"Training took {time.time()-t0:.1f}s")

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nFinal validation accuracy: {val_acc:.4f}")

# Save model, tokenizer, label encoder, history
model.save("models/simplernn_model.keras")

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("models/rnn_history.json", "w") as f:
    json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

with open("models/rnn_results.json", "w") as f:
    json.dump({"accuracy": float(val_acc), "loss": float(val_loss),
               "max_len": MAX_LEN, "max_words": MAX_WORDS}, f, indent=2)

print("Saved model, tokenizer, label encoder, and results.")
