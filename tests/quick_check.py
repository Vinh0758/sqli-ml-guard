import requests

BASE = "http://127.0.0.1:8000"
cases = [
    ("vuln", "Lap"),
    ("vuln", "O'Brien"),
    ("vuln", "zzz' OR 1=1 -- -"),
    ("vuln", "zzz' UNION SELECT id, username, 0 FROM users -- -"),
    ("safe", "zzz' OR 1=1 -- -"),
]
print(requests.get(f"{BASE}/health").json())
for ep, payload in cases:
    r = requests.get(f"{BASE}/{ep}/products", params={"name": payload})
    print(f"{ep:5s} {payload!r:55s} -> {r.status_code} {r.text[:90]}")