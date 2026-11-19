import pandas as pd

df = pd.read_csv("data/raw/sqli.csv", encoding="utf-16", on_bad_lines="skip")
df.columns = [c.strip() for c in df.columns]
df.to_csv("data/raw/sqli_utf8.csv", index=False, encoding="utf-8")
print(df.shape)