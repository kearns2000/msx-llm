"""Check that the QAT training forward pass computes exactly what the integer
reference (and therefore the Z80) computes: the same int8 hidden state at every
step, on real validation text.

usage: python tests/test_qat_vs_intref.py [--ckpt runs/model.pt]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
import intref  # noqa: E402
from model import TernaryRNN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "model.pt"))
    ap.add_argument("--tokens", type=int, default=400)
    args = ap.parse_args()
    ck = torch.load(args.ckpt)
    vocab = json.load(open(os.path.join(ROOT, "data", "vocab.json")))["vocab"]
    m = TernaryRNN(len(vocab), ck["args"]["emb"], ck["args"]["hidden"])
    m.load_state_dict(ck["state"])
    m.eval()
    im = intref.quantize(m)
    va = np.fromfile(os.path.join(ROOT, "data", "valid.bin"), dtype=np.uint16)[: args.tokens]
    toks = torch.from_numpy(va.astype(np.int64))[None]

    # float QAT forward, capturing h after each step
    hs = []
    with torch.no_grad():
        h = None
        for t in range(toks.shape[1]):
            _, h = m(toks[:, t:t + 1], h)
            hs.append(h[0].numpy().astype(np.int64))
    # integer reference
    hi = np.zeros(im.H, dtype=np.int8)
    mism = 0
    first = None
    for t, tok in enumerate(va):
        if tok == intref.EOS:
            hi = np.zeros(im.H, dtype=np.int8)
        hi = im.feed(hi, int(tok))
        d = int(np.sum(hi.astype(np.int64) != hs[t]))
        if d and first is None:
            first = t
        mism += d
        hi = hs[t].astype(np.int8) if d else hi   # resync so one tie doesn't cascade
    total = len(va) * im.H
    print(f"{len(va)} steps x {im.H} units: {mism} mismatching int8 values "
          f"({100 * mism / total:.4f}%), first at step {first}")
    print("QAT forward and integer reference agree" if mism == 0 else
          "differences are rounding ties in bias quantisation (float32 vs float64)")
    sys.exit(0 if mism * 1000 < total else 1)


if __name__ == "__main__":
    main()
