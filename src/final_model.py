import json
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from preprocess import normalize

FPR_TARGET = 0.003


def read_raw(path):
    ls = [l.rstrip("\r\n") for l in open(path, encoding="utf-8-sig", errors="replace")]
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


def make_pipe():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize, analyzer="char", ngram_range=(3, 5),
                                  min_df=2, sublinear_tf=True, max_features=300_000)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])


tr = pd.concat([pd.read_csv("data/processed/train.csv"),
                pd.read_csv("data/processed/augment.csv")], ignore_index=True)
tr["Sentence"] = tr["Sentence"].astype(str)
tr["group"] = tr["Sentence"].map(lambda s: re.sub(r"\d+", "0", normalize(s)))
test = pd.read_csv("data/processed/test.csv")
ext = read_raw("data/raw/external_sqli.csv")
hand = read_raw("data/raw/handmade_benign.csv")

# 1) chọn ngưỡng trên validation (tách theo nhóm từ train)
fit_i, val_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=1).split(tr, tr.Label, groups=tr.group))
p = make_pipe().fit(tr.iloc[fit_i]["Sentence"], tr.iloc[fit_i]["Label"])
pv = p.predict_proba(tr.iloc[val_i]["Sentence"])[:, 1]
yv = tr.iloc[val_i]["Label"].values
best = None
for th in [x / 100 for x in range(30, 100)]:
    fpr = (pv[yv == 0] >= th).mean()
    rec = (pv[yv == 1] >= th).mean()
    if fpr <= FPR_TARGET and (best is None or rec > best[1]):
        best = (th, rec, fpr)
TH = best[0] if best else 0.5
print(f"Ngưỡng chọn trên validation: {TH:.2f} (val recall {best[1]:.3f}, val FPR {best[2]:.4f})" if best
      else "Không đạt FPR mục tiêu, dùng 0.5")

# 2) huấn luyện lại trên toàn bộ train + augment, đánh giá
pipe = make_pipe().fit(tr["Sentence"], tr["Label"])
y = test["Label"].values
pt = pipe.predict_proba(test["Sentence"].astype(str))[:, 1]
pe = pipe.predict_proba(ext)[:, 1]
ph = pipe.predict_proba(hand)[:, 1]

print("\nngưỡng | test_recall | test_fpr | ext_recall | hand_fpr")
for th in sorted({0.5, 0.6, 0.7, 0.8, 0.9, round(TH, 2)}):
    print(f"{th:5.2f}  | {(pt[y == 1] >= th).mean():.3f}       | {(pt[y == 0] >= th).mean():.4f}  |"
          f" {(pe >= th).mean():.3f}      | {int((ph >= th).sum())}/{len(ph)}")

joblib.dump(pipe, "models/best.joblib")
json.dump({"block": 0.5, "monitor": 0.3}, open("models/threshold.json", "w"))
print("\nĐã lưu models/best.joblib và models/threshold.json")