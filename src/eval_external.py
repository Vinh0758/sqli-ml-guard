import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocess import normalize
import joblib
pipe = joblib.load("models/logreg_aug.joblib")

def load(path):
    """Đọc thô từng dòng: an toàn với dấu phẩy và nháy kép."""
    try:
        lines = [l.rstrip("\r\n") for l in open(path, encoding="utf-8-sig", errors="replace")]
    except FileNotFoundError:
        return None
    if lines and lines[0].strip().lower() == "sentence":
        lines = lines[1:]
    lines = [l.strip() for l in lines if l.strip()]
    # file do script tạo (to_csv) có thể bọc cả dòng trong "..." và nhân đôi dấu "
    out = []
    for l in lines:
        if len(l) >= 2 and l[0] == '"' and l[-1] == '"':
            l = l[1:-1].replace('""', '"')
        out.append(l)
    return pd.Series(out, dtype=str)


def report(name, s, expect_attack):
    if s is None:
        print(f"\n[{name}] chưa có file"); return
    p = pipe.predict_proba(s)[:, 1] >= 0.5
    if expect_attack:
        print(f"\nRecall trên {name}: {p.mean():.3f} ({p.sum()}/{len(p)})")
        print("Bị bỏ sót (tối đa 25):"); print(s[~p].head(25).to_string())
    else:
        print(f"\nTỉ lệ bị gắn cờ trên {name}: {p.mean():.3f} ({p.sum()}/{len(p)})")
        if p.sum(): print("Bị chặn nhầm:"); print(s[p].to_string())


report("handmade_benign (kỳ vọng ~0)", load("data/raw/handmade_benign.csv"), False)
report("external_sqli", load("data/raw/external_sqli.csv"), True)
report("external_fragments (chỉ quan sát)", load("data/raw/external_fragments.csv"), False)

probes = pd.read_csv("reports/probes_fragments.csv", encoding="utf-8")["Sentence"].astype(str)
print(f"\nProbe nội bộ bị gắn cờ: {(pipe.predict_proba(probes)[:, 1] >= 0.5).mean():.3f}")