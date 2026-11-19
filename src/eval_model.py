import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocess import normalize


def read_raw(path):
    try:
        ls = [l.rstrip("\r\n") for l in open(path, encoding="utf-8-sig", errors="replace")]
    except FileNotFoundError:
        return pd.Series([], dtype=str)
    if ls and ls[0].strip().lower() == "sentence":
        ls = ls[1:]
    out = []
    for l in ls:
        l = l.strip()
        if len(l) >= 2 and l[0] == '"' and l[-1] == '"':
            l = l[1:-1].replace('""', '"')
        if l:
            out.append(l)
    return pd.Series(out, dtype=str)


base = pd.read_csv("data/processed/train.csv")
aug = pd.read_csv("data/processed/augment.csv")
extra = pd.read_csv("data/processed/extra_train.csv")
test = pd.read_csv("data/processed/test.csv")
ext = read_raw("data/raw/external_sqli.csv")
hand = read_raw("data/raw/handmade_benign.csv")

configs = {
    "A: train gốc": base,
    "B: gốc + augment": pd.concat([base, aug], ignore_index=True),
    "C: gốc + augment + sqliv2": pd.concat([base, aug, extra], ignore_index=True),
}

rows = []
for name, tr in configs.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, analyzer="char", ngram_range=(3, 5),
                                  min_df=2, sublinear_tf=True, max_features=300_000)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ]).fit(tr["Sentence"].astype(str), tr["Label"])

    pt = pipe.predict_proba(test["Sentence"].astype(str))[:, 1] >= 0.5
    y = test["Label"].values
    tp, fp = int((pt & (y == 1)).sum()), int((pt & (y == 0)).sum())
    fn, tn = int((~pt & (y == 1)).sum()), int((~pt & (y == 0)).sum())
    pe = pipe.predict_proba(ext)[:, 1] >= 0.5
    ph = pipe.predict_proba(hand)[:, 1] >= 0.5
    rows.append(dict(config=name, n_train=len(tr),
                     test_recall=round(tp / max(tp + fn, 1), 3),
                     test_fpr=round(fp / max(fp + tn, 1), 4),
                     ext_recall=round(float(pe.mean()), 3),
                     hand_fpr=f"{int(ph.sum())}/{len(ph)}"))
    if name.startswith("C"):
        joblib.dump(pipe, "models/logreg_full.joblib")
        print("\nBị chặn nhầm (handmade), cấu hình C:\n", hand[ph].to_string())
        print("\nBị bỏ sót (external), cấu hình C:\n", ext[~pe].head(20).to_string())

res = pd.DataFrame(rows)
print("\n", res.to_string(index=False))
res.to_csv("reports/ablation_data.csv", index=False)