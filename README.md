# 🐦 Twitter Sentiment Analyzer — ML + SimpleRNN + Streamlit

A sentiment classification project on the **Twitter Sentiment Analysis** dataset
(74,682 training tweets / 1,000 validation tweets, 4 classes: `Positive`,
`Negative`, `Neutral`, `Irrelevant`).

It trains **three classical ML models** and a **SimpleRNN deep learning model**,
compares their accuracy, and serves both through a single **Streamlit** app where
the user picks "Machine Learning" or "SimpleRNN (Deep Learning)" from a sidebar.

## Results

| Model                        | Type | Validation Accuracy |
|-------------------------------|------|----------------------|
| Logistic Regression (TF-IDF)  | ML   | 96.40% |
| Multinomial Naive Bayes (TF-IDF) | ML | 84.30% |
| Linear SVM (TF-IDF)           | ML   | **96.80%** |
| SimpleRNN (Embedding + SimpleRNN) | DL | **95.80%** |

All four models clear the 80% accuracy target. Charts:
- `assets/accuracy_comparison.png` — bar chart of all four models
- `assets/rnn_training_curves.png` — SimpleRNN accuracy/loss per epoch

## Project structure

```
project/
├── app.py                     # Streamlit app (ML + SimpleRNN toggle)
├── preprocessing.py           # shared text-cleaning function
├── train_ml.py                # trains Logistic Regression / Naive Bayes / Linear SVM
├── train_rnn.py                # trains the SimpleRNN model
├── make_charts.py             # builds the comparison charts
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
│   └── label_encoder.pkl
└── assets/
    ├── accuracy_comparison.png
    └── rnn_training_curves.png
```

Model artifacts are already trained and saved in `models/` — **you don't need to
retrain anything** to run or deploy the app. If you ever want to retrain:
`python train_ml.py` then `python train_rnn.py` then `python make_charts.py`.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploy it for free on Streamlit Community Cloud

I can't create a live public URL myself from this sandbox — Streamlit Cloud
deployment needs to happen through **your own** GitHub + Streamlit accounts.
Here's the exact path (5–10 minutes):

1. **Create a GitHub repo** and push this whole `project/` folder to it
   (make sure `app.py`, `requirements.txt`, `preprocessing.py`, the `models/`
   folder, and the `assets/` folder are all included — the `data/` folder is
   optional since the app doesn't need the raw CSVs at runtime).
   ```bash
   git init
   git add .
   git commit -m "Twitter sentiment analyzer: ML + SimpleRNN + Streamlit"
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
  SimpleRNN, no GPU needed) fits comfortably.
- If GitHub warns about large files, `models/simplernn_model.keras` (~24 MB)
  is the biggest one — it's under GitHub's 100 MB limit so a normal `git push`
  is fine, no Git LFS needed.
- First load after a period of inactivity may take ~30–60s while the app
  "wakes up" — that's normal for the free tier.

Alternatives if you'd rather not use Streamlit Cloud: Hugging Face Spaces
(supports Streamlit natively) or Render.com both work with the same
`app.py` + `requirements.txt`.
