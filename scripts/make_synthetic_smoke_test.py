"""Generates a SYNTHETIC ratings file (item,user,rating,timestamp) ONLY to smoke-test
that the pipeline runs end to end. Metrics from this file are meaningless and must
NOT be reported as results."""
import random, sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
out = sys.argv[2] if len(sys.argv) > 2 else "data/synthetic_smoke_test.csv"
random.seed(1)
with open(out, "w") as f:
    for _ in range(n):
        u, i = random.randint(1, 1500), random.randint(1, 600)
        f.write(f"ITEM{i:05d},USER{u:05d},{random.randint(1, 5)}.0,{1500000000 + random.randint(0, 10**7)}\n")
print("wrote", out)
