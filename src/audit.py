import pandas as pd
from preprocess import normalize

df = pd.read_csv("data/raw/sqli_utf8.csv").dropna(subset=["Sentence"])
df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
df = df.dropna(subset=["Label"]).astype({"Label": int})
df["norm"] = df["Sentence"].map(normalize)

SQL_STRONG = r"\b(select|union|insert|drop|exec|sleep|benchmark|waitfor|xp_cmdshell)\b|--"
SQL_ANY = SQL_STRONG + r"|'|\"|;|=|\bor\b|\band\b"

sus1 = df[(df.Label == 1) & ~df["norm"].str.contains(SQL_ANY, regex=True)]
sus0 = df[(df.Label == 0) & df["norm"].str.contains(SQL_STRONG, regex=True)]

sus1.to_csv("reports/suspect_label1.csv", index=False)
sus0.to_csv("reports/suspect_label0.csv", index=False)
print("Nhãn 1 không có dấu hiệu SQL:", len(sus1))
print("Nhãn 0 có từ khóa SQL mạnh:", len(sus0))
pd.set_option("display.max_colwidth", 100)
print(sus1["Sentence"].head(15).to_string())
print(sus0["Sentence"].head(15).to_string())