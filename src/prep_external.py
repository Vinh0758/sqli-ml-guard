import re
import sys
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/external_sqli_full.csv"

# đọc thô từng dòng, KHÔNG dùng pd.read_csv (payload chứa dấu phẩy và nháy kép)
lines = [l.rstrip("\r\n") for l in open(src, encoding="utf-8", errors="replace")]
if lines and lines[0].strip().lower() == "sentence":      # bỏ dòng tiêu đề nếu có
    lines = lines[1:]
lines = [l.strip() for l in lines]
lines = [l for l in lines if l and not l.startswith("#")]
lines = list(dict.fromkeys(lines))                         # bỏ trùng, giữ thứ tự

df = pd.DataFrame({"Sentence": lines})
def shape(s):
    s = s.lower()
    s = re.sub(r"\d+", "0", s)                       # che chữ số
    s = re.sub(r"'[a-z]{4}'", "'x'", s)              # chuỗi ngẫu nhiên của sqlmap
    s = re.sub(r"(char\(0\)\+?)+", "char(0)+", s)    # chuỗi CHAR(n)+CHAR(n)+... -> một đơn vị
    s = re.sub(r"(0,)+0", "0,0", s)                  # dãy số 1,2,3,... -> 0,0
    s = re.sub(r"(null,)+null", "null,null", s)      # NULL,NULL,NULL,... -> một đơn vị
    s = re.sub(r"(limit=0)", "limit=0", s)
    return s

df["pattern"] = df["Sentence"].map(shape)
# mảnh vụn ngắn: tách riêng, không tính vào recall
frag = df["Sentence"].str.len() <= 6
df[frag][["Sentence"]].to_csv("data/raw/external_fragments.csv", index=False, encoding="utf-8")

main = df[~frag]
sampled = (main.groupby("pattern", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), 3), random_state=42)))
sampled[["Sentence"]].to_csv("data/raw/external_sqli.csv", index=False, encoding="utf-8")

print("Dòng gốc sau bỏ trùng:", len(df))
print("Mảnh vụn (<=6 ký tự):", int(frag.sum()))
print("Số khuôn khác nhau:", main["pattern"].nunique())
print("Giữ lại để kiểm tra:", len(sampled))