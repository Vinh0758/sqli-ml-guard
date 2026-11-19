import pandas as pd

df = pd.read_csv("data/raw/sqli.csv", encoding="utf-16", on_bad_lines="skip")
df.columns = [c.strip() for c in df.columns]

print("Kích thước:", df.shape)
print("Tên cột:", df.columns.tolist())
print("\nGiá trị thiếu:\n", df.isna().sum())

text_col, label_col = "Sentence", "Label"
df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
bad = df[df[label_col].isna()]
print("\nDòng có nhãn không hợp lệ:", len(bad))
print(bad.head(10).to_string())
df = df.dropna(subset=[label_col, text_col])
df[label_col] = df[label_col].astype(int)

print("\nPhân bố nhãn:\n", df[label_col].value_counts())
print("\nSố dòng trùng hoàn toàn:", df.duplicated().sum())
print("Số văn bản trùng:", df[text_col].duplicated().sum())
lens = df[text_col].astype(str).str.len()
print("\nĐộ dài (ký tự):\n", lens.describe())

pd.set_option("display.max_colwidth", 120)
print("\n--- 8 mẫu nhãn 0 ---")
print(df[df[label_col] == 0][text_col].sample(8, random_state=1).to_string())
print("\n--- 8 mẫu nhãn 1 ---")
print(df[df[label_col] == 1][text_col].sample(8, random_state=1).to_string())