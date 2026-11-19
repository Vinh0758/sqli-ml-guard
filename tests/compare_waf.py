import requests

TARGETS = {
    "no_waf (8001)": "http://127.0.0.1:8001/ping",
    "ml_waf (8000)": "http://127.0.0.1:8000/ping",
    "modsec_crs (8080)": "http://127.0.0.1:8080/ping",
}


def read_raw(path):
    out = []
    ls = [l.rstrip("\r\n") for l in open(path, encoding="utf-8-sig", errors="replace")]
    if ls and ls[0].strip().lower() == "sentence":
        ls = ls[1:]
    for l in ls:
        l = l.strip()
        if len(l) >= 2 and l[0] == '"' and l[-1] == '"':
            l = l[1:-1].replace('""', '"')
        if l:
            out.append(l)
    return out


attacks = read_raw("data/raw/external_sqli.csv")
benign = read_raw("data/raw/handmade_benign.csv")
s = requests.Session()


def count_blocked(url, items):
    blocked = errors = 0
    for p in items:
        try:
            r = s.get(url, params={"name": p}, timeout=3)
            blocked += r.status_code == 403
        except requests.RequestException:
            errors += 1
    return blocked, errors


print(f"{'cấu hình':20s} {'detection':>10s} {'FPR':>8s} {'lỗi':>6s}")
for name, url in TARGETS.items():
    ba, ea = count_blocked(url, attacks)
    bb, eb = count_blocked(url, benign)
    print(f"{name:20s} {ba/len(attacks):10.3f} {bb/len(benign):8.3f} {ea+eb:6d}")