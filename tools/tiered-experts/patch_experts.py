"""In-place expert swap for sensitivity scans on a GGUF: in a WORKING COPY of the Q8_0 model, the selected experts of the
selected layers are replaced by their low-bit version from another GGUF of the same model (dequantized from e.g.
IQ2_XXS and re-quantized to Q8_0, which adds only Q8 rounding), so llama.cpp runs the model "with only those experts
at low precision". --restore copies the original bytes back from the pristine Q8_0 file. No 124 GB rewrite per test.

  python patch_experts.py --work w.gguf --low low.gguf --sel '{"7": "all"}'           # all experts of layer 7
  python patch_experts.py --work w.gguf --low low.gguf --sel '{"0": [3, 17], ...}'     # some experts
  python patch_experts.py --work w.gguf --pristine q8.gguf --sel '{"7": "all"}' --restore
"""
import argparse
import json
import re

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType as T
from gguf.quants import dequantize, quantize

EXP = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def expert_tensors(reader):
    return {t.name: t for t in reader.tensors if EXP.match(t.name)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="Q8_0 working copy, modified in place")
    ap.add_argument("--low", help="low-bit GGUF of the same model (source of the patched experts)")
    ap.add_argument("--pristine", help="untouched Q8_0 (for --restore)")
    ap.add_argument("--sel", required=True, help='json {layer: "all" | [expert indices]}')
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()
    sel = {int(k): v for k, v in json.loads(args.sel).items()}
    work = expert_tensors(GGUFReader(args.work, "r+"))
    src = expert_tensors(GGUFReader(args.pristine if args.restore else args.low, "r"))
    n = 0
    for name, wt in work.items():
        layer = int(EXP.match(name).group(1))
        if layer not in sel:
            continue
        assert wt.tensor_type == T.Q8_0, (name, wt.tensor_type)
        st = src[name]
        n_exp = wt.data.shape[0]
        idx = range(n_exp) if sel[layer] == "all" else sel[layer]
        for e in idx:
            if args.restore:
                wt.data[e] = st.data[e]
            else:
                x = dequantize(np.asarray(st.data[e]), st.tensor_type)        # [rows, cols] float32
                wt.data[e] = quantize(x, T.Q8_0).reshape(wt.data[e].shape)
            n += 1
    for t in work.values():  # flush the memmap
        if hasattr(t.data, "base") and hasattr(t.data.base, "flush"):
            t.data.base.flush()
    print(json.dumps({"patched_expert_slices": n, "restore": args.restore}))


if __name__ == "__main__":
    main()
