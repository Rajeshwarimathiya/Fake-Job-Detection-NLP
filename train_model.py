"""Train and evaluate the fake-job classifier.

Run from the project directory after placing the Kaggle CSV at
``data/fake_job_postings.csv``:

    python train_model.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


REQUIRED_COLUMNS = {"title", "description", "requirements", "fraudulent"}
RANDOM_STATE = 42


def normalize_text(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def build_text(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["title"].fillna("").map(normalize_text)
        + " "
        + frame["description"].fillna("").map(normalize_text)
        + " "
        + frame["requirements"].fillna("").map(normalize_text)
    ).str.strip()


def choose_threshold(y_true: pd.Series, fake_probability: np.ndarray) -> float:
    thresholds = np.arange(0.10, 0.91, 0.01)
    scores = []
    for threshold in thresholds:
        prediction = (fake_probability >= threshold).astype(int)
        scores.append((f1_score(y_true, prediction, zero_division=0), recall_score(y_true, prediction, zero_division=0), threshold))
    return float(max(scores)[2])


def metrics(y_true: pd.Series, prediction: np.ndarray, fake_probability: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, fake_probability)),
        "average_precision": float(average_precision_score(y_true, fake_probability)),
    }


def train(input_path: Path, model_dir: Path) -> None:
    frame = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.drop_duplicates(subset=["title", "description", "requirements"]).copy()
    frame["fraudulent"] = pd.to_numeric(frame["fraudulent"], errors="coerce")
    frame = frame.dropna(subset=["fraudulent"])
    frame = frame[frame["fraudulent"].isin([0, 1])]
    frame["text"] = build_text(frame)
    frame = frame[frame["text"].str.len() > 0]

    text_train, text_temp, y_train, y_temp = train_test_split(
        frame["text"], frame["fraudulent"].astype(int), test_size=0.30,
        stratify=frame["fraudulent"], random_state=RANDOM_STATE,
    )
    text_valid, text_test, y_valid, y_test = train_test_split(
        text_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE,
    )

    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train = vectorizer.fit_transform(text_train)
    x_valid = vectorizer.transform(text_valid)
    x_test = vectorizer.transform(text_test)

    positive = int(y_train.sum())
    negative = len(y_train) - positive
    classifier = XGBClassifier(
        n_estimators=350, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=2,
        reg_alpha=0.1, reg_lambda=2.0, scale_pos_weight=negative / max(positive, 1),
        eval_metric="logloss", tree_method="hist", random_state=RANDOM_STATE,
    )
    classifier.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    valid_probability = classifier.predict_proba(x_valid)[:, list(classifier.classes_).index(1)]
    threshold = choose_threshold(y_valid, valid_probability)
    test_probability = classifier.predict_proba(x_test)[:, list(classifier.classes_).index(1)]
    test_prediction = (test_probability >= threshold).astype(int)

    report = metrics(y_test, test_prediction, test_probability)
    print(json.dumps({"threshold": threshold, "test": report}, indent=2))
    print(classification_report(y_test, test_prediction, target_names=["real", "fake"], zero_division=0))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_dir / "xgb_final_model.joblib", compress=3)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib", compress=3)
    manifest = {
        "sha256": {
            filename: hashlib.sha256((model_dir / filename).read_bytes()).hexdigest()
            for filename in ("xgb_final_model.joblib", "tfidf_vectorizer.joblib")
        }
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (model_dir / "metadata.json").write_text(
        json.dumps({"threshold": threshold, "metrics": report, "random_state": RANDOM_STATE}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/fake_job_postings.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    args = parser.parse_args()
    train(args.input, args.model_dir)
