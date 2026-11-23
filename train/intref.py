"""Bit-exact Python reference of the Z80 integer inference.

Every operation here mirrors an instruction sequence in z80/engine.asm.
If you change one, change the other, then run tests/run_golden.py.

  quantize(model, ...) -> IntModel     (float QAT checkpoint -> integers + LUTs)
  IntModel.feed(h, tok)  -> h'          one recurrent step
  IntModel.logits(h)     -> int16[V]
  IntModel.sample(...)                  top-K sampling with LUT softmax + xorshift
  IntModel.generate(...)                full prompt -> tokens loop, as on the MSX
"""
import math

import numpy as np

EOS = 0


def wrap16(x):
    return ((np.asarray(x, dtype=np.int64) + 32768) & 0xFFFF) - 32768


def xorshift16(s):
    """Literal emulation of the 'rand' routine in engine.asm (J. Metcalf's
    16-bit xorshift, 7/9/8), including the carry-flag plumbing."""
    h, l = s >> 8, s & 255
    a = (l >> 1) | ((h & 1) << 7)          # ld a,h / rra / ld a,l / rra
    a ^= h                                  # xor h   (clears carry)
    h = a                                   # ld h,a
    c = l & 1                               # ld a,l / rra (carry in 0)
    a = (h >> 1) | (c << 7)                 # ld a,h / rra
    a ^= l                                  # xor l
    l = a                                   # ld l,a
    a ^= h                                  # xor h
    h = a                                   # ld h,a
    return (h << 8) | l


class IntModel:
    def __init__(self, d):
        self.__dict__.update(d)
        self.E = int(self.E)
        self.H = int(self.H)
        self.V = int(self.V)
        for k in ("kz", "kn", "ke", "topk", "max_new"):
            setattr(self, k, int(getattr(self, k)))
        # ternary matrices as int64 for exact numpy matmul
        self._wg = np.concatenate([self.wz, self.wn], 0).astype(np.int64)  # (2H, E+H)
        self._bg = np.concatenate([self.bz, self.bn]).astype(np.int64)
        self._wo = self.wo.astype(np.int64)

    # ---- one recurrent step --------------------------------------------------
    @staticmethod
    def lut_index(y, k):
        rc = (1 << k) >> 1
        t = wrap16(y + rc) >> k          # arithmetic shift (sra h / rr l) k times
        return np.clip(t, -128, 127)

    def gate_y(self, h, tok):
        v = np.concatenate([self.emb[tok].astype(np.int64), h.astype(np.int64)])
        return wrap16(self._bg + self._wg @ v)

    def feed(self, h, tok):
        y = self.gate_y(h, tok)
        H = self.H
        z = self.lutz[self.lut_index(y[:H], self.kz) + 128].astype(np.int64)
        n = self.lutn[self.lut_index(y[H:], self.kn) + 128].astype(np.int64)
        h = h.astype(np.int64)
        p = z * (n - h) + 64
        q = (p >> 7) & 0xFF                     # add hl,hl ; take H
        hn = (h + q) & 0xFF
        return np.where(hn > 127, hn - 256, hn).astype(np.int8)

    def logits(self, h):
        return wrap16(self.bo.astype(np.int64) + self._wo @ h.astype(np.int64))

    # ---- sampling ----------------------------------------------------------
    def topk_select(self, y):
        K = self.topk
        tv = [-32768] * K
        ti = [0] * K
        for i, v in enumerate(int(x) for x in y):
            if v > tv[K - 1]:
                j = K - 1
                while j > 0 and v > tv[j - 1]:
                    tv[j], ti[j] = tv[j - 1], ti[j - 1]
                    j -= 1
                tv[j], ti[j] = v, i
        return tv, ti

    def sample(self, y, rng, greedy=False):
        tv, ti = self.topk_select(y)
        if greedy:
            return ti[0], rng
        w = [int(self.lute[min(255, (tv[0] - v) >> self.ke)]) for v in tv]
        S = sum(w)
        rng = xorshift16(rng)
        t = ((rng >> 8) * S) >> 8
        cum = 0
        for j, wj in enumerate(w):
            cum += wj
            if t < cum:
                return ti[j], rng
        raise AssertionError("unreachable")

    def generate(self, prompt_ids, seed, max_new=None, greedy=False):
        """prompt_ids excludes the leading <eos>; mirrors the MSX main loop."""
        max_new = self.max_new if max_new is None else max_new
        h = np.zeros(self.H, dtype=np.int8)
        hist = [EOS] + list(prompt_ids)
        for t in hist:
            h = self.feed(h, t)
        rng = seed
        out = []
        y = None
        h_before = h
        while True:
            y = self.logits(h)
            tok, rng = self.sample(y, rng, greedy)
            out.append(tok)
            h_before = h
            if tok == EOS or len(out) >= max_new:
                break
            h = self.feed(h, tok)
        return {"tokens": out, "h": h_before, "logits": y, "rng": rng}


# ---- float checkpoint -> integer model ----------------------------------------
def quantize(model, temperature=0.8, topk=8, max_new=48):
    import torch
    from model import SIG_RANGE, TANH_RANGE, lut_shift  # noqa: F401

    with torch.no_grad():
        E, H, V = model.E, model.H, model.V
        emb = torch.round(torch.clamp(model.emb, -1, 1) * 127).to(torch.int8).numpy()

        def tern(w):
            s = float(w.abs().mean().clamp(min=1e-5))
            q = torch.clamp(torch.round(w / s), -1, 1).to(torch.int8).numpy()
            return q, s

        wz, sz = tern(model.wz)
        wn, sn = tern(model.wn)
        wo, so = tern(model.wo)
        kz = lut_shift(sz, SIG_RANGE)
        kn = lut_shift(sn, TANH_RANGE)

        def bint(b, s):
            v = np.round(b.numpy().astype(np.float64) * 127.0 / s)
            return np.clip(v, -32768, 32767).astype(np.int16)

        bz, bn, bo = bint(model.bz, sz), bint(model.bn, sn), bint(model.bo, so)
        # LUTs; computed in float32 like the trainer so rounding ties agree
        idx = np.arange(-128, 128, dtype=np.float32)
        stepz = np.float32(sz / 127.0 * (1 << kz))
        stepn = np.float32(sn / 127.0 * (1 << kn))
        lutz = torch.round(128 * torch.sigmoid(torch.from_numpy(idx * stepz))).numpy().astype(np.uint8)
        lutn = torch.round(127 * torch.tanh(torch.from_numpy(idx * stepn))).numpy().astype(np.int8)
        # softmax LUT over logit differences: w(d) = 255*exp(-(d<<ke)*so/127/T)
        unit = so / 127.0 / temperature           # nats per logit unit
        ke = max(0, min(15, round(math.log2(12.0 / 255.0 / unit))))
        d = np.arange(256, dtype=np.float64) * (1 << ke)
        lute = np.round(255 * np.exp(-d * unit)).astype(np.uint8)
    return IntModel(dict(E=E, H=H, V=V, emb=emb, wz=wz, wn=wn, wo=wo,
                         bz=bz, bn=bn, bo=bo, kz=kz, kn=kn, ke=ke,
                         lutz=lutz, lutn=lutn, lute=lute, topk=topk, max_new=max_new,
                         sz=sz, sn=sn, so=so, temperature=temperature))


def load(path):
    d = dict(np.load(path, allow_pickle=False))
    return IntModel(d)


def save(im, path):
    keys = ["E", "H", "V", "emb", "wz", "wn", "wo", "bz", "bn", "bo", "kz", "kn",
            "ke", "lutz", "lutn", "lute", "topk", "max_new", "sz", "sn", "so",
            "temperature"]
    np.savez(path, **{k: np.asarray(getattr(im, k)) for k in keys})


def int_loss(im, tokens, nstreams=16, steps=256):
    """Cross-entropy (nats/token) of the INTEGER model on a token stream, using
    logits * so/127 (what the sampler sees at temperature 1)."""
    n = len(tokens) // nstreams
    st = tokens[: n * nstreams].reshape(nstreams, n)
    H = im.H
    h = np.zeros((nstreams, H), dtype=np.int64)
    wg, bg, wo = im._wg, im._bg, im._wo
    tot, cnt = 0.0, 0
    for t in range(min(steps, n - 1)):
        tok = st[:, t]
        h[tok == EOS] = 0
        v = np.concatenate([im.emb[tok].astype(np.int64), h], 1)
        y = wrap16(bg + v @ wg.T)
        z = im.lutz[IntModel.lut_index(y[:, :H], im.kz) + 128].astype(np.int64)
        nn_ = im.lutn[IntModel.lut_index(y[:, H:], im.kn) + 128].astype(np.int64)
        hn = (h + (((z * (nn_ - h) + 64) >> 7) & 0xFF)) & 0xFF
        h = np.where(hn > 127, hn - 256, hn)
        lg = wrap16(im.bo.astype(np.int64) + h @ wo.T) * (float(im.so) / 127.0)
        lg = lg - lg.max(1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
        tot -= lp[np.arange(nstreams), st[:, t + 1]].sum()
        cnt += nstreams
    return tot / cnt
