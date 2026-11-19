# src/baseline_aug.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocess import normalize

train = pd.concat([pd.read_csv("data/processed/train.csv"),
                   pd.read_csv("data/processed/augment.csv")], ignore_index=True)
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(preprocessor=normalize, analyzer="char", ngram_range=(3, 5),
                              min_df=2, sublinear_tf=True, max_features=200_000)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
]).fit(train["Sentence"].astype(str), train["Label"])

import joblib
joblib.dump(pipe, "models/logreg_aug.joblib")
print("Đã huấn luyện với", len(train), "mẫu")