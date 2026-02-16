"""
train.py
- Downloads a real public SMS spam dataset (no CSV files needed from you)
- Optionally adds your app's user feedback (user_data.jsonl)
- Trains TF-IDF + Logistic Regression
- Saves model to model.pkl
"""

import io
import json
import joblib
import requests
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

MODEL_FILE = "model.pkl"
USER_DATA_FILE = "user_data.jsonl"

# Public dataset (TSV): label + text
DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


def load_user_feedback(path: str):

    texts = []
    labels = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip corrupted lines

                text = str(row.get("text", "")).strip()
                label = row.get("label", None)

                if not text:
                    continue
                if label not in (0, 1):
                    continue

                texts.append(text)
                labels.append(int(label))

    except FileNotFoundError:
        pass

    return texts, labels


def main():
    # 1) Download dataset
    print("Downloading dataset from source...")
    resp = requests.get(DATA_URL, timeout=30)
    resp.raise_for_status()

    # Dataset is tab-separated with two columns: label \t text
    raw = resp.text
    df = pd.read_csv(io.StringIO(raw), sep="\t", names=["label", "text"])

    # Convert "ham"/"spam" -> 0/1
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["text"] = df["text"].astype(str)

    # Drop any weird rows
    df = df.dropna(subset=["label", "text"])

    # 2) Load user feedback and append it (optional)
    user_texts, user_labels = load_user_feedback(USER_DATA_FILE)
    if user_texts:
        print(f"Loaded {len(user_texts)} user-labeled examples from {USER_DATA_FILE}")
        df_user = pd.DataFrame({"text": user_texts, "label": user_labels})
        df = pd.concat([df, df_user], ignore_index=True)
    else:
        print("No user feedback found yet (user_data.jsonl not present or empty).")

    # 3) Train/test split
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y
    )

    # 4) Build model pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    # 5) Train
    print("Training model...")
    model.fit(X_train, y_train)

    # 6) Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\nAccuracy:", round(acc, 4))
    print("\nClassification report:\n", classification_report(y_test, preds))

    # 7) Save model
    joblib.dump(model, MODEL_FILE)
    print(f"\nSaved model to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
