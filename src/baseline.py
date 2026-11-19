import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from preprocess import normalize

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
Xtr, ytr = train["Sentence"].astype(str), train["Label"].values
Xte, yte = test["Sentence"].astype(str), test["Label"].values

REGEX = re.compile(r"union\s+select|or\s+1\s*=\s*1|'\s*or\s*'|--|;\s*drop|sleep\s*\(|benchmark\s*\(|xp_cmdshell")

def metrics(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9); fpr = fp / max(fp + tn, 1)
    return prec, rec, f1, fpr

def boot(y, p, n=500, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); out.append(metrics(y[i], p[i]))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)

def tfidf():
    return TfidfVectorizer(preprocessor=normalize, analyzer="char", ngram_range=(3, 5),
                           min_df=2, sublinear_tf=True, max_features=200_000)

preds = {"Regex": Xte.map(lambda s: int(bool(REGEX.search(normalize(s))))).values}
for name, clf in {
    "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "LinearSVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
}.items():
    pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)]).fit(Xtr, ytr)
    preds[name] = (pipe.predict_proba(Xte)[:, 1] >= 0.5).astype(int)

rows = []
for name, p in preds.items():
    m = metrics(yte, p); lo, hi = boot(yte, p)
    rows.append(dict(model=name, precision=m[0], recall=m[1], f1=m[2], fpr=m[3],
                     f1_ci=f"[{lo[2]:.3f}, {hi[2]:.3f}]", fpr_ci=f"[{lo[3]:.3f}, {hi[3]:.3f}]"))
res = pd.DataFrame(rows).round(4)
print(res.to_string(index=False))
res.to_csv("reports/baseline.csv", index=False)

# FPR riêng trên từng nguồn benign
for src in ["kaggle", "synthetic_benign"]:
    mask = ((test["source"] == src) & (test["Label"] == 0)).values
    print(f"\nFPR trên {src}:", {k: round(float(v[mask].mean()), 4) for k, v in preds.items()})