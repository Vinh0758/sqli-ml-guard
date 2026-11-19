import pandas as pd

for f in ["data/raw/sqliv2.csv", "data/raw/SQLiV3.csv"]:
    df = None
    for enc in ["utf-8", "utf-16", "latin-1"]:
        try:
            df = pd.read_csv(f, encoding=enc, on_bad_lines="skip")
            break
        except Exception:
            df = None
    print("\n====", f, "| encoding:", enc)
    if df is None:
        print("không đọc được")
        continue
    print("Kích thước:", df.shape, "| Cột:", df.columns.tolist())
    pd.set_option("display.max_colwidth", 80)
    print(df.head(3).to_string())
    for c in df.columns:
        if df[c].nunique() <= 5:
            print("Phân bố", c, ":", df[c].value_counts().to_dict())