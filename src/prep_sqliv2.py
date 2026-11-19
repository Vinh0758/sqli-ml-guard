import re
import pandas as pd
from preprocess import normalize

# ---- đọc và làm sạch cơ bản ----
df = pd.read_csv("data/raw/sqliv2.csv", encoding="utf-16", on_bad_lines="skip")
df.columns = [c.strip() for c in df.columns]
df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
df = df.dropna(subset=["Sentence", "Label"])
df = df[df["Label"].isin([0, 1])].astype({"Label": int})
df["Sentence"] = df["Sentence"].astype(str)
df = df[df["Sentence"].str.len() <= 1000]
print("Sau đọc và bỏ NaN:", len(df))


def group_key(s):
    return re.sub(r"\d+", "0", normalize(s))


df["norm"] = df["Sentence"].map(normalize)
df["group"] = df["Sentence"].map(group_key)
df = df[df["norm"].str.len() > 0].drop_duplicates(subset="norm")
print("Sau loại trùng:", len(df))

# ---- cùng quy tắc lọc nhãn như clean_labels.py ----
KEEP = re.compile(
    r"(--|#|/\*)"
    r"|\b(select|union|insert|update|delete|drop|create|alter|grant|declare|exec|execute"
    r"|sleep|benchmark|waitfor|xp_cmdshell|load_file|utl_http|utl_inaddr)\b"
    r"|['\"]\s*\)*\s*(or|and|\|\|)"
    r"|\b(or|and)\b\s*\(*\s*['\"\w]+\s*\)*\s*(=|like|<|>|\bin\b|\bbetween\b)"
    r"|\b(or|and)\b\s*\d+\s*(between|in)\b"
    r"|\b0x[0-9a-f]{12,}"
    r"|\|\|\s*\w{3,}"
    r"|@@\w+"
    r"|\bchar\s*\(\s*\d"
)
LDAP = re.compile(r"\(\|\(|\(objectclass|\(mail=")

is1 = df.Label == 1
has_struct = df["norm"].map(lambda s: bool(KEEP.search(s))) | df["Sentence"].str.contains(r"/\*", regex=True)
is_ldap = df["Sentence"].map(lambda s: bool(LDAP.search(s)))
bad1 = is1 & (~has_struct | is_ldap | (df["norm"].str.len() < 3))
print("Nhãn 1 bị loại theo quy tắc cấu trúc:", int(bad1.sum()))
df = df[~bad1]

# ---- loại mọi thứ có thể trùng với tập đánh giá ----
def read_raw(path):
    try:
        ls = [l.rstrip("\r\n") for l in open(path, encoding="utf-8-sig", errors="replace")]
    except FileNotFoundError:
        return []
    out = []
    for l in ls[1:]:
        l = l.strip()
        if len(l) >= 2 and l[0] == '"' and l[-1] == '"':
            l = l[1:-1].replace('""', '"')
        if l:
            out.append(l)
    return out

hold = set(pd.read_csv("data/processed/test.csv")["Sentence"].astype(str).map(group_key))
for p in ["data/raw/external_sqli_full.csv", "data/raw/external_sqli.csv",
          "data/raw/external_fragments.csv", "data/raw/handmade_benign.csv"]:
    hold |= {group_key(s) for s in read_raw(p)}
hold |= set(pd.read_csv("reports/probes_fragments.csv")["Sentence"].astype(str).map(group_key))

before = len(df)
df = df[~df["group"].isin(hold)]
print("Loại vì trùng nhóm với test/external/handmade/probe:", before - len(df))

# ---- giới hạn lặp khuôn ----
att = df[df.Label == 1].groupby("group", group_keys=False).apply(lambda g: g.sample(min(len(g), 4), random_state=42))
ben = df[df.Label == 0].groupby("group", group_keys=False).apply(lambda g: g.sample(min(len(g), 2), random_state=42))
ben = ben.sample(min(len(ben), 6000), random_state=42)

out = pd.concat([att, ben])[["Sentence", "Label"]]
out["source"] = "sqliv2"
out.to_csv("data/processed/extra_train.csv", index=False, encoding="utf-8")
print("\nextra_train:", len(out))
print(out.groupby("Label").size())