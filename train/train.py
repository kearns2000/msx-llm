"""Train the ternary RNN on tokenised TinyStories (CPU).

usage: python train/train.py --minutes 60 [--hidden 384 --emb 128]
Writes runs/model.pt (latest) and prints train/valid loss.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import TernaryRNN  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(model, data, B=32, T=128, nbatch=8):
    model.eval()
    n = len(data) // B
    streams = torch.from_numpy(data[: n * B].astype(np.int64)).view(B, n)
    h, tot, cnt = None, 0.0, 0
    with torch.no_grad():
        for i in range(nbatch):
            x = streams[:, i * T:(i + 1) * T]
            y = streams[:, i * T + 1:(i + 1) * T + 1]
            if y.shape[1] < T:
                break
            logits, h = model(x, h)
            tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   y.reshape(-1), reduction="sum").item()
            cnt += y.numel()
    model.train()
    return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "model.pt"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1234)

    vocab = json.load(open(os.path.join(ROOT, "data", "vocab.json")))["vocab"]
    train = np.fromfile(os.path.join(ROOT, "data", "train.bin"), dtype=np.uint16)
    valid = np.fromfile(os.path.join(ROOT, "data", "valid.bin"), dtype=np.uint16)
    model = TernaryRNN(len(vocab), args.emb, args.hidden)
    if args.resume and os.path.exists(args.out):
        model.load_state_dict(torch.load(args.out)["state"])
    print(f"params: ternary {model.n_ternary():,}  emb {model.emb.numel():,}  "
          f"tokens {len(train):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95))
    B, T = args.batch, args.seq
    n = len(train) // B
    streams = torch.from_numpy(train[: n * B].astype(np.int64)).view(B, n)
    budget = args.minutes * 60
    t0 = time.time()
    h, pos, step, ema = None, 0, 0, None
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    while True:
        el = time.time() - t0
        if el > budget:
            break
        frac = el / budget
        lr = args.lr * (min(1.0, step / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac))))
        for g in opt.param_groups:
            g["lr"] = lr
        if pos + T + 1 > n:
            pos, h = 0, None
        x = streams[:, pos:pos + T]
        y = streams[:, pos + 1:pos + T + 1]
        pos += T
        logits, h = model(x, h)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 50 == 0:
            tok_s = step * B * T / (time.time() - t0)
            print(f"step {step} loss {ema:.3f} lr {lr:.2e} {tok_s:,.0f} tok/s "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if step % 500 == 0:
            vl = evaluate(model, valid)
            print(f"  valid loss {vl:.3f} (ppl {math.exp(vl):.1f})", flush=True)
            torch.save({"state": model.state_dict(), "args": vars(args),
                        "vocab": len(vocab)}, args.out)
    vl = evaluate(model, valid, nbatch=32)
    print(f"final valid loss {vl:.3f} (ppl {math.exp(vl):.1f}) after {step} steps",
          flush=True)
    torch.save({"state": model.state_dict(), "args": vars(args), "vocab": len(vocab),
                "valid_loss": vl, "steps": step}, args.out)


if __name__ == "__main__":
    main()
