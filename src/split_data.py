import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from preprocess import normalize

df = pd.read_csv("data/processed/dataset_clean.csv")
df["norm"] = df["Sentence"].map(normalize)
df["group"] = df["norm"].map(lambda s: re.sub(r"\d+", "0", s))   # che chữ số để gom các biến thể

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(df, df["Label"], groups=df["group"]))
train, test = df.iloc[tr_idx], df.iloc[te_idx]

train.drop(columns=["norm", "group"]).to_csv("data/processed/train.csv", index=False)
test.drop(columns=["norm", "group"]).to_csv("data/processed/test.csv", index=False)

print("Train:", len(train), "| Test:", len(test))
print("Giao nhau về nhóm (phải = 0):", len(set(train.group) & set(test.group)))
print("\nTrain:\n", train.groupby(["source", "Label"]).size())
print("\nTest:\n", test.groupby(["source", "Label"]).size())