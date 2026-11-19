import time
import sys
import numpy as np
import requests

URL = "http://127.0.0.1:8080/ping"
N = 1000
s = requests.Session()
for i in range(50):
    s.get(URL, params={"name": f"warmup{i}"})      # khởi động
lat = []
for i in range(N):
    t0 = time.perf_counter()
    s.get(URL, params={"name": f"Lap{i}"})
    lat.append((time.perf_counter() - t0) * 1000)
p50, p95, p99 = np.percentile(lat, [50, 95, 99])
print(f"{sys.argv[1] if len(sys.argv) > 1 else ''} p50={p50:.2f} p95={p95:.2f} p99={p99:.2f} ms")