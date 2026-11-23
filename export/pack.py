"""Export a trained checkpoint to ASCII8 MegaROM data banks.

usage: python export/pack.py [--ckpt runs/model.pt] [--temp 0.8 --topk 8]

Outputs (in build/):
  model_int.npz   integer model used by the Python reference (train/intref.py)
  model_data.bin  ROM banks 1..N (bank 0 is the Z80 code, see z80/engine.asm)
  manifest.json   human-readable copy of the on-ROM manifest

ROM data layout (each bank is 8 KB, mapped at 6000h-7FFFh via ASCII8 reg 6800h)
  bank 1 +0000  manifest (96 bytes)
         +0100  LUT_Z  256 x u8   sigmoid, Q7 (0..128), index = clamp(idx)+128
         +0200  LUT_N  256 x s8   tanh * 127
         +0300  LUT_E  256 x u8   softmax weights 255*exp(-d*2^ke*so/127/T)
         +0400  VOCAB  V x (len u8, bytes)  (must fit in bank 1)
  bank 2 +0000  EMB    V x E  s8        (E=128 -> 64 rows/bank, rows never straddle)
         ...    BG     2H x s16 LE     (z biases then n biases, in acc units)
         ...    WG     ternary gates,  chunk-major (see below)
         ...    BO     V x s16 LE
         ...    WO     ternary output, chunk-major
Ternary matrix packing ("chunk-major"): inputs are split in chunks of 64.
For chunk c, for row r (all rows), 16 bytes; byte g holds inputs
64c+4g+0..3 in bits 1:0, 3:2, 5:4, 7:6 with 00=0, 01=+1, 11=-1 (10 unused).
Blocks are 16-byte aligned so a 16-byte row-chunk never straddles a bank.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))

BANK = 8192
BLOCKS = ["VOCAB", "EMB", "BG", "WG", "BO", "WO", "LUTZ", "LUTN", "LUTE"]


def pack_ternary(w):
    """w: (rows, n) int8 in {-1,0,1}, n % 64 == 0 -> bytes, chunk-major."""
    rows, n = w.shape
    assert n % 64 == 0
    code = np.where(w > 0, 1, np.where(w < 0, 3, 0)).astype(np.uint8)
    code = code.reshape(rows, n // 64, 16, 4)
    byte = code[..., 0] | (code[..., 1] << 2) | (code[..., 2] << 4) | (code[..., 3] << 6)
    return np.ascontiguousarray(byte.transpose(1, 0, 2)).tobytes()   # (chunk,row,16)


class RomImage:
    def __init__(self):
        self.buf = bytearray()       # starts at bank 1
        self.loc = {}

    def align(self, a):
        while len(self.buf) % a:
            self.buf.append(0xFF)

    def put(self, name, data, align=16):
        self.align(align)
        self.loc[name] = len(self.buf)
        self.buf += data

    def put_at(self, name, off, data):
        if len(self.buf) < off:
            self.buf += b"\xff" * (off - len(self.buf))
        assert len(self.buf) == off, (name, off, len(self.buf))
        self.loc[name] = off
        self.buf += data

    def bank_off(self, name):
        a = self.loc[name]
        return 1 + a // BANK, a % BANK


def build(im, vocab):
    E, H, V = im.E, im.H, im.V
    assert E == 128, "engine hard-codes E=128 (64 embedding rows per bank)"
    assert (E + H) % 64 == 0 and H % 64 == 0
    assert 2 * H <= 1024 and V <= 1024, "Y buffer is 1024 words"
    assert E + H <= 512, "VEC buffer is 512 bytes"
    vb = bytearray()
    for i, s in enumerate(vocab):
        b = s.encode("ascii")
        assert len(b) < 256
        vb += bytes([len(b)]) + b
    assert 0x400 + len(vb) <= BANK, f"vocab too big for bank 1: {len(vb)}"

    rom = RomImage()
    rom.put_at("MANIFEST", 0, b"\0" * 96)
    rom.put_at("LUTZ", 0x100, im.lutz.astype(np.uint8).tobytes())
    rom.put_at("LUTN", 0x200, im.lutn.astype(np.int8).tobytes())
    rom.put_at("LUTE", 0x300, im.lute.astype(np.uint8).tobytes())
    rom.put_at("VOCAB", 0x400, bytes(vb))
    rom.put("EMB", im.emb.astype(np.int8).tobytes(), align=BANK)
    rom.put("BG", np.concatenate([im.bz, im.bn]).astype("<i2").tobytes())
    rom.put("WG", pack_ternary(np.concatenate([im.wz, im.wn], 0)))
    rom.put("BO", im.bo.astype("<i2").tobytes())
    rom.put("WO", pack_ternary(im.wo))

    man = bytearray(b"MSXLLM01")
    man += struct.pack("<HHH", E, H, V)
    man += bytes([(E + H) // 64, H // 64, im.kz, im.kn, im.ke, im.topk])
    man += struct.pack("<HH", (1 << im.kz) >> 1, (1 << im.kn) >> 1)
    man += bytes([im.max_new])
    man += b"\0" * (32 - len(man))
    for name in BLOCKS:
        b, o = rom.bank_off(name)
        man += struct.pack("<BBH", b, 0, o)
    assert len(man) <= 96
    rom.buf[0:len(man)] = man

    # total ROM (code bank + data) padded to a power of two
    total = BANK + len(rom.buf)
    size = 1 << (total - 1).bit_length()
    rom.buf += b"\xff" * (size - total)
    info = {
        "E": E, "H": H, "V": V, "kz": im.kz, "kn": im.kn, "ke": im.ke,
        "topk": im.topk, "max_new": im.max_new, "temperature": float(im.temperature),
        "rom_bytes": size, "data_bytes_used": total - BANK,
        "blocks": {n: dict(zip(("bank", "offset"), rom.bank_off(n))) for n in BLOCKS},
        "ternary_weights": int(im.wz.size + im.wn.size + im.wo.size),
        "zero_fraction": float(np.mean(np.concatenate(
            [im.wz.ravel(), im.wn.ravel(), im.wo.ravel()]) == 0)),
    }
    return bytes(rom.buf), info


def main():
    import torch
    import intref
    from model import TernaryRNN

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "model.pt"))
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--eval", action="store_true", help="int-model valid loss")
    args = ap.parse_args()

    ck = torch.load(args.ckpt)
    a = ck["args"]
    vocab = json.load(open(os.path.join(ROOT, "data", "vocab.json")))["vocab"]
    model = TernaryRNN(len(vocab), a["emb"], a["hidden"])
    model.load_state_dict(ck["state"])
    im = intref.quantize(model, args.temp, args.topk, args.max_new)
    os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
    intref.save(im, os.path.join(ROOT, "build", "model_int.npz"))
    data, info = build(im, vocab)
    open(os.path.join(ROOT, "build", "model_data.bin"), "wb").write(data)
    info["valid_loss_float_ckpt"] = ck.get("valid_loss")
    if args.eval:
        va = np.fromfile(os.path.join(ROOT, "data", "valid.bin"), dtype=np.uint16).astype(np.int64)
        info["valid_loss_int"] = intref.int_loss(im, va)
    json.dump(info, open(os.path.join(ROOT, "build", "manifest.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
