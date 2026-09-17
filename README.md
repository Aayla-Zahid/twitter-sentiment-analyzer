# 🐦 Twitter Sentiment Analyzer — ML + SimpleRNN + LSTM + GRU + Streamlit

A sentiment classification project on the **Twitter Sentiment Analysis** dataset
(74,682 training tweets / 1,000 validation tweets, 4 classes: `Positive`,
`Negative`, `Neutral`, `Irrelevant`).

It trains **three classical ML models** and **three deep learning models**
(SimpleRNN, LSTM, and GRU), compares their accuracy, and serves all of them
through a single **Streamlit** app where the user picks a model type from a
sidebar.

## Results

| Model                              | Type | Validation Accuracy |
|-------------------------------------|------|----------------------|
| Logistic Regression (TF-IDF)        | ML   | 96.40% |
| Multinomial Naive Bayes (TF-IDF)    | ML   | 84.30% |
| Linear SVM (TF-IDF)                 | ML   | 96.80% |
| SimpleRNN (Embedding + SimpleRNN)   | DL   | 95.80% |
| LSTM (Bidirectional, 2-layer)       | DL   | **96.00%** |
| GRU (Bidirectional, 2-layer)        | DL   | 95.50% |

All models clear the 80% accuracy target. Charts:
- `assets/accuracy_comparison.png` — bar chart of the ML models + SimpleRNN
- `assets/rnn_training_curves.png` — SimpleRNN accuracy/loss per epoch
- `assets/final_accuracy_comparison.png` — bar chart comparing LSTM vs GRU final accuracy
- `assets/training_curves_comparison.png` — LSTM vs GRU accuracy/loss curves per epoch

## Project structure

```
project/
├── app.py                     # Streamlit app (ML / SimpleRNN / LSTM / GRU toggle)
├── preprocessing.py           # shared text-cleaning function
├── train_ml.py                # trains Logistic Regression / Naive Bayes / Linear SVM
├── train_rnn.py                # trains the SimpleRNN model
├── make_charts.py             # builds the ML/RNN comparison charts
├── Twitter_Sentiment_LSTM_vs_GRU.ipynb   # Colab notebook: trains & evaluates LSTM and GRU
├── requirements.txt
├── data/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
├── models/                    # saved, ready-to-use model artifacts (already trained)
│   ├── tfidf_vectorizer.pkl
│   ├── ml_models.pkl
│   ├── ml_results.json
│   ├── simplernn_model.keras
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   ├── rnn_results.json
│   ├── lstm_model.keras
│   ├── gru_model.keras
│   ├── lstm_gru_tokenizer.pkl
│   ├── lstm_gru_label_encoder.pkl
│   ├── lstm_results.json
│   └── gru_results.json
└── assets/
    ├── accuracy_comparison.png
    ├── rnn_training_curves.png
    ├── final_accuracy_comparison.png
    └── training_curves_comparison.png
```

Model artifacts are already trained and saved in `models/` — **you don't need to
retrain anything** to run or deploy the app. If you ever want to retrain:
- ML + SimpleRNN: `python train_ml.py` then `python train_rnn.py` then `python make_charts.py`
- LSTM + GRU: run `Twitter_Sentiment_LSTM_vs_GRU.ipynb` in Google Colab (mount
  Drive, point it at `data/twitter_training.csv` and `data/twitter_validation.csv`,
  run all cells).

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploy it for free on Streamlit Community Cloud

1. **Create a GitHub repo** and push this whole `project/` folder to it
   (make sure `app.py`, `requirements.txt`, `preprocessing.py`, the `models/`
   folder, and the `assets/` folder are all included — the `data/` folder is
   optional since the app doesn't need the raw CSVs at runtime).
   ```bash
   git init
   git add .
   git commit -m "Twitter sentiment analyzer: ML + SimpleRNN + LSTM + GRU + Streamlit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, pick your repo/branch, set the main file path to
   `app.py`, and click **Deploy**.
4. Streamlit Cloud will install `requirements.txt` and launch the app. You'll
   get a public URL like `https://<your-app-name>.streamlit.app`.
5. Paste that link into your project write-up / submission.

**Notes for a smooth deploy:**
- Free-tier Streamlit Cloud apps have ~1 GB RAM — this app (TF-IDF models +
  three small recurrent networks, no GPU needed) fits comfortably.
- If GitHub's web upload UI rejects a file over 25 MB (this affects
  `lstm_model.keras` and `gru_model.keras`, both ~30–35 MB), push those files
  via `git` from the command line or from Colab instead — GitHub's real limit
  via git is 100 MB per file, no Git LFS needed at this size.
- First load after a period of inactivity may take ~30–60s while the app
  "wakes up" — that's normal for the free tier.

Alternatives if you'd rather not use Streamlit Cloud: Hugging Face Spaces
(supports Streamlit natively) or Render.com both work with the same
`app.py` + `requirements.txt`.
