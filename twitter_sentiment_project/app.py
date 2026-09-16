import json
import pickle

import numpy as np
import streamlit as st

st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="🐦", layout="centered")

# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------

@st.cache_resource
def load_ml_assets():
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/ml_models.pkl", "rb") as f:
        ml_models = pickle.load(f)
    with open("models/ml_results.json") as f:
        ml_results = json.load(f)
    return vectorizer, ml_models, ml_results


@st.cache_resource
def load_rnn_assets():
    import tensorflow as tf
    model = tf.keras.models.load_model("models/simplernn_model.keras")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("models/rnn_results.json") as f:
        rnn_results = json.load(f)
    return model, tokenizer, label_encoder, rnn_results


from preprocessing import clean_text

LABEL_COLORS = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral": "#3498db",
    "Irrelevant": "#95a5a6",
}
LABEL_EMOJI = {
    "Positive": "😊",
    "Negative": "😠",
    "Neutral": "😐",
    "Irrelevant": "🤷",
}

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.title("⚙️ Settings")
approach = st.sidebar.radio(
    "Choose the model type:",
    ["Machine Learning", "SimpleRNN (Deep Learning)"],
    help="Pick a classical ML model (TF-IDF based) or the SimpleRNN deep learning model.",
)

ml_model_choice = None
if approach == "Machine Learning":
    vectorizer, ml_models, ml_results = load_ml_assets()
    ml_model_choice = st.sidebar.selectbox("Choose ML algorithm:", list(ml_models.keys()))
    st.sidebar.metric("Validation Accuracy", f"{ml_results['accuracies'][ml_model_choice]*100:.2f}%")
else:
    rnn_model, tokenizer, label_encoder, rnn_results = load_rnn_assets()
    st.sidebar.metric("Validation Accuracy", f"{rnn_results['accuracy']*100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**\n\n"
    "This app classifies tweet sentiment as **Positive**, **Negative**, **Neutral**, "
    "or **Irrelevant**, using either classical Machine Learning models (TF-IDF + "
    "Logistic Regression / Naive Bayes / Linear SVM) or a **SimpleRNN** deep learning model."
)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

st.title("🐦 Twitter Sentiment Analyzer")
st.write(
    "Enter a tweet or any short piece of text below, choose a model in the sidebar, "
    "and see the predicted sentiment."
)

text_input = st.text_area(
    "Enter text to analyze:",
    height=120,
    placeholder="e.g. I absolutely love the new update, it's fantastic!",
)

analyze = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

if analyze:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(text_input)

        if approach == "Machine Learning":
            model = ml_models[ml_model_choice]
            X = vectorizer.transform([cleaned])
            pred = model.predict(X)[0]

            # Probabilities if supported, otherwise decision function
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                classes = model.classes_
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)[0]
                exp = np.exp(scores - np.max(scores))
                proba = exp / exp.sum()
                classes = model.classes_

            label = pred
        else:
            seq = tokenizer.texts_to_sequences([cleaned])
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            padded = pad_sequences(seq, maxlen=rnn_results["max_len"], padding="post", truncating="post")
            proba_arr = rnn_model.predict(padded, verbose=0)[0]
            classes = label_encoder.classes_
            proba = proba_arr
            label = classes[int(np.argmax(proba_arr))]

        color = LABEL_COLORS.get(label, "#333")
        emoji = LABEL_EMOJI.get(label, "")
        st.markdown(
            f"<div style='padding:1.2rem;border-radius:10px;background:{color}22;"
            f"border:2px solid {color};text-align:center;'>"
            f"<span style='font-size:2rem;'>{emoji}</span><br>"
            f"<span style='font-size:1.5rem;font-weight:700;color:{color};'>{label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if proba is not None:
            st.write("")
            st.write("**Confidence by class:**")
            probs_sorted = sorted(zip(classes, proba), key=lambda x: -x[1])
            for cls, p in probs_sorted:
                st.write(f"{LABEL_EMOJI.get(cls,'')} {cls}")
                st.progress(float(p), text=f"{p*100:.1f}%")

        with st.expander("See cleaned text used by the model"):
            st.code(cleaned if cleaned else "(empty after cleaning)")

st.markdown("---")

# ----------------------------------------------------------------------------
# Model comparison section
# ----------------------------------------------------------------------------

with st.expander("📊 Model accuracy comparison (ML vs Deep Learning)"):
    st.image("assets/accuracy_comparison.png", use_container_width=True)
    st.image("assets/rnn_training_curves.png", use_container_width=True)
    st.caption(
        "All models were trained on the Twitter Sentiment Analysis dataset "
        "(74,682 training tweets, 4 classes: Positive / Negative / Neutral / Irrelevant) "
        "and evaluated on a held-out 1,000-tweet validation set."
    )
