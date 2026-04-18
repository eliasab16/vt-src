import sys
import time
from websockets.sync.client import connect

if len(sys.argv) < 2:
    print("Usage: python ws_latency_test.py ws://<IP>:<PORT>/ws")
    sys.exit(1)

URL = sys.argv[1]
N = 50

with connect(URL) as ws:
    ws.send("warmup")
    ws.recv()

    latencies = []
    for _ in range(N):
        t0 = time.perf_counter()
        ws.send("ping")
        ws.recv()
        latencies.append((time.perf_counter() - t0) * 1000)

latencies.sort()
p50 = latencies[N // 2]
p95 = latencies[int(N * 0.95)]
jitter = latencies[-1] - latencies[0]

print(f"\n--- Results ({N} samples) ---")
print(f"min:  {latencies[0]:.1f} ms")
print(f"p50:  {p50:.1f} ms")
print(f"p95:  {p95:.1f} ms")
print(f"max:  {latencies[-1]:.1f} ms")
print(f"jitter (max-min): {jitter:.1f} ms")

print(f"\n--- Assessment ---")
if p50 < 50:
    print(f"EXCELLENT ({p50:.0f}ms)")
elif p50 < 100:
    print(f"GOOD ({p50:.0f}ms)")
elif p50 < 150:
    print(f"OK ({p50:.0f}ms) — pick a server in a closer region to improve")
else:
    print(f"HIGH ({p50:.0f}ms) — needs fixing:")
    print(f"  - Pick a server in a closer region")
    print(f"  - Use direct TCP port, not HTTP proxy URL")
    print(f"  - Use ethernet instead of wifi")

if jitter > 50:
    print(f"\nHIGH JITTER ({jitter:.0f}ms):")
    print(f"  - Use ethernet instead of wifi")
    print(f"  - Avoid cellular-backed ISPs (check with traceroute). Fiber is more consistent.")
