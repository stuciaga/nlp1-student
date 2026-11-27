"""Sanity tests: python part3/utils/test_setup.py"""
import numpy as np
from prompts import build_instruction, parse_generated_string as P
from metrics import summarize, bootstrap_ci, paired_bootstrap

for s, want in [("B", 1), ("(B)", 1), ("Answer: B", 1), ("B. positive", 1), ("a", 0),
                ("The answer is positive", 1), ("Both A and B", -1), ("positive and negative", -1),
                ("Because", -1), ("AB", -1), ("", -1)]:
    assert P(s) == want, (s, P(s), want)
# swapped order: A = positive
assert P("A", ("positive", "negative")) == 1 and P("B", ("positive", "negative")) == 0
assert "(A) positive" in build_instruction(1, ("positive", "negative"))

g = np.array([0, 1, 1, 0] * 50)
p = g.copy(); p[:10] = 1 - p[:10]; p[10] = -1
s = summarize(g, p)
assert abs(s["accuracy"] - 189 / 200) < 1e-9 and s["invalid_rate"] == 0.005
lo, hi = bootstrap_ci(g, p); assert lo < s["accuracy"] < hi
r = paired_bootstrap(g, p, g); assert r["diff"] < 0 and r["p_b"] + r["p_tie"] + r["p_a"] == 1
print("ok")
