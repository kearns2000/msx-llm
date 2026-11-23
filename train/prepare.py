"""Build the vocabulary and tokenise TinyStories into uint16 .bin files.

usage: python train/prepare.py [--vocab 1024]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tok import EOS, Tokenizer, build_vocab, iter_stories  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def encode_file(tok, path, out, max_bytes=None):
    t0 = time.time()
    chunks = []
    cur = []
    n_chars = 0
    for s in iter_stories(path, max_bytes):
        cur.append(EOS)
        cur.extend(tok.encode(s))
        n_chars += len(s)
        if len(cur) > 1_000_000:
            chunks.append(np.array(cur, dtype=np.uint16))
            cur = []
    chunks.append(np.array(cur, dtype=np.uint16))
    arr = np.concatenate(chunks)
    arr.tofile(out)
    print(f"{out}: {len(arr):,} tokens, {n_chars / len(arr):.2f} chars/token, "
          f"{time.time() - t0:.0f}s")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--vocab-bytes", type=int, default=40_000_000,
                    help="how much training text to scan when picking the vocab")
    args = ap.parse_args()

    train_txt = os.path.join(DATA, "ts_train_part.txt")
    valid_txt = os.path.join(DATA, "ts_valid.txt")
    vocab = build_vocab(iter_stories(train_txt, args.vocab_bytes), args.vocab)
    tok = Tokenizer(vocab)
    tok.save(os.path.join(DATA, "vocab.json"))
    print(f"vocab {len(vocab)} (chars {sum(1 for v in vocab if len(v) == 1)}), "
          f"string bytes {sum(len(v) + 1 for v in vocab)}")

    va = encode_file(tok, valid_txt, os.path.join(DATA, "valid.bin"))
    single = np.isin(va, [i for i, v in enumerate(vocab) if len(v) == 1]).mean()
    print(f"valid: {single * 100:.1f}% of tokens are single-char fallbacks")
    encode_file(tok, train_txt, os.path.join(DATA, "train.bin"))


if __name__ == "__main__":
    main()
