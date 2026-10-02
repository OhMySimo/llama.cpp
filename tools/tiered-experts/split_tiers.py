"""Split a qwen4exp GGUF (bf16) into two expert groups for the tiered-experts llama.cpp fork (branch tiered-experts):
per layer, group 1 = the listed experts (the ones that deserve more bits), group 2 = the rest. Router rows
(ffn_gate_inp) are reordered [group 1 | group 2]; ffn_{gate,up,down}_exps keep group 1 (+ one all-zero dummy expert)
and new ffn_{gate,up,down}_exps_t2 hold group 2 (+ dummy). Layers not listed are left untouched. llama-quantize then
gives the two groups different types (--tensor-type "_exps_t2=..." first). Optionally splits the imatrix the same way.

  python split_tiers.py --src m-bf16.gguf --tiers tiers.json --out m-tier.gguf [--imatrix imatrix.gguf --imatrix-out imx-tier.gguf]
  tiers.json: {"<layer>": [expert indices of group 1], ...}
"""
import argparse
import json
import re

import numpy as np
import gguf

EXP = re.compile(r"^blk\.(\d+)\.(ffn_gate_inp|ffn_gate_exps|ffn_up_exps|ffn_down_exps)\.weight(\.in_sum2|\.counts)?$")


def plan(reader, tiers):
    """[(name, raw tensor type, producer)] in output order; producers build the array lazily."""
    out = []
    for t in reader.tensors:
        m = EXP.match(t.name)
        if not m or int(m.group(1)) not in tiers:
            out.append((t.name, t.tensor_type, lambda t=t: t.data))
            continue
        L, kind, suffix = int(m.group(1)), m.group(2), m.group(3) or ""
        n = t.data.shape[0]
        g1 = list(tiers[L])
        g2 = [i for i in range(n) if i not in set(g1)]
        if kind == "ffn_gate_inp" and suffix:  # imatrix of the router: per input column, nothing to reorder
            out.append((t.name, t.tensor_type, lambda t=t: t.data))
            continue
        if kind == "ffn_gate_inp":
            out.append((t.name, t.tensor_type, lambda t=t, p=g1 + g2: np.ascontiguousarray(t.data[p])))
            continue

        def group(t=t, idx=None, suffix=suffix):
            d = t.data[idx]
            if suffix:  # imatrix statistics: the dummy expert gets the mean (never zero importance)
                dummy = d.mean(axis=0, keepdims=True)
            else:       # weights: all-zero dummy expert
                dummy = np.zeros_like(t.data[:1])
            return np.ascontiguousarray(np.concatenate([d, dummy], axis=0))
        out.append((t.name, t.tensor_type, lambda t=t, g=g1: group(t, g)))
        name2 = t.name.replace(f"{kind}.weight", f"{kind}_t2.weight")
        out.append((name2, t.tensor_type, lambda t=t, g=g2: group(t, g)))
    return out


def rewrite(src, dst, tiers):
    reader = gguf.GGUFReader(src)
    arch = reader.fields.get(gguf.Keys.General.ARCHITECTURE)
    writer = gguf.GGUFWriter(dst, arch=arch.contents() if arch else None, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=st)
    items = plan(reader, tiers)
    for name, ttype, make in items:  # shapes only (the producers are cheap views until concatenation)
        a = make()
        writer.add_tensor_info(name, a.shape, a.dtype, a.nbytes, ttype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for name, ttype, make in items:
        writer.write_tensor_data(make(), tensor_endianess=reader.endianess)
    writer.close()
    return len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tiers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imatrix")
    ap.add_argument("--imatrix-out")
    args = ap.parse_args()
    tiers = {int(k): sorted(v) for k, v in json.load(open(args.tiers)).items()}
    n = rewrite(args.src, args.out, tiers)
    res = {"tensors": n, "tiered_layers": len(tiers), "group1_sizes": sorted({len(v) for v in tiers.values()})}
    if args.imatrix:
        res["imatrix_entries"] = rewrite(args.imatrix, args.imatrix_out, tiers)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
