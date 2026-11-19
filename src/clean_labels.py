import re
import pandas as pd
from preprocess import normalize

df = pd.read_csv("data/processed/dataset.csv")
df["norm"] = df["Sentence"].map(normalize)

# Dấu hiệu SQLi có cấu trúc: comment, từ khóa lệnh, đóng chuỗi + or/and,
# hex dài, || + tên hàm, biến @@, char(n)
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

# LDAP injection: không phải SQLi, loại khỏi tập
LDAP = re.compile(r"\(\|\(|\(objectclass|\(mail=")

is1 = df.Label == 1
n = df["norm"].str.len()

# /**/ bị normalize biến thành khoảng trắng nên kiểm tra thêm trên chuỗi gốc
has_struct = (
    df["norm"].map(lambda s: bool(KEEP.search(s)))
    | df["Sentence"].str.contains(r"/\*", regex=True)
)
is_ldap = df["Sentence"].map(lambda s: bool(LDAP.search(str(s))))

# Nhãn 1 có cấu trúc SQL, đủ dài, không phải LDAP -> giữ
keep1 = is1 & has_struct & (n >= 3) & ~is_ldap

# Nhãn 1 không đạt: ngắn -> cách ly (probe/mảnh vụn); dài mà không có SQL (kể cả LDAP) -> loại để rà tay
quarantine = is1 & ~keep1 & (n <= 8)
drop_long_no_sql = is1 & ~keep1 & (n > 8)
too_long = n > 1000                       # rác multipart, log dài

# Lưu các nhóm bị tách ra để kiểm tra lại
df[quarantine].drop(columns="norm").to_csv(
    "reports/probes_fragments.csv", index=False, encoding="utf-8")
df[drop_long_no_sql | too_long].drop(columns="norm").to_csv(
    "reports/dropped_labels.csv", index=False, encoding="utf-8")

removed = quarantine | drop_long_no_sql | too_long
out = df[~removed].drop(columns="norm")
out.to_csv("data/processed/dataset_clean.csv", index=False, encoding="utf-8")

print("Cách ly (probe/mảnh vụn):", int(quarantine.sum()))
print("Loại (nhãn 1 dài, không có SQL):", int(drop_long_no_sql.sum()))
print("Loại (quá dài > 1000 ký tự):", int(too_long.sum()))
print(out.groupby(["source", "Label"]).size())