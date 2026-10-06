#!/usr/bin/env python3
"""Expert-cache profile for the tiered-experts fork: which experts of each (layer, group) llama.cpp keeps in VRAM.

Input: the usage log written with LLAMA_EXPERT_STATS=<file> (one "<gate tensor name> <local expert id>" line per routed
slot computed on the CPU). Output: a ranking of all (layer, group, expert) by uses, most used first; the runtime takes
them in this order until LLAMA_EXPERT_CACHE_MB is spent, so one profile serves every VRAM budget.

  python tools/expert_cache_profile.py stats.txt profile.txt
"""
import collections
import re
import sys

counts = collections.Counter()
for line in open(sys.argv[1]):
    name, e = line.split()
    m = re.match(r"blk\.(\d+)\.ffn_gate_exps(_t2)?\.weight", name)
    if m:
        counts[(int(m.group(1)), 2 if m.group(2) else 1, int(e))] += 1
with open(sys.argv[2], "w") as f:
    f.write("# layer group expert uses (most used first)\n")
    for (layer, group, e), n in counts.most_common():
        f.write(f"{layer} {group} {e} {n}\n")
print(f"{len(counts)} experts ranked from {sum(counts.values())} routed slots")
