"""Ternary gated RNN ("GRU without reset gate"), trained quantisation-aware.

Why an RNN and not a transformer (see README for the long version):
  * constant work per token, no KV cache (64 KB RAM total on an MSX1);
  * attention needs activation x activation products (no hardware multiply on
    the Z80); a ternary matvec needs only adds;
  * dropping the GRU reset gate makes every gate a function of [x; h], so ONE
    pass over the input vector feeds both gates (fewer lookup-table builds,
    no r*h multiply before a second matvec).

Recurrence (all integer on the Z80, emulated exactly here in float32):
    v      = [x_q ; h_q]                      int8, scale 1/127
    A_z    = Wz_t . v + bz_int                int16, Wz_t ternary {-1,0,1}
    A_n    = Wn_t . v + bn_int
    z_q    = LUT_Z[clamp((A_z + rc_z) >> k_z)]   in [0,128]   (sigmoid, Q7)
    n_q    = LUT_N[clamp((A_n + rc_n) >> k_n)]   in [-127,127] (tanh)
    h_q'   = h_q + ((z_q * (n_q - h_q) + 64) >> 7)
    logits = Wo_t . h_q' + bo_int             int16, scale s_o/127
Weights are BitNet b1.58 style: W_t = clamp(round(W / mean|W|), -1, 1) with
the per-tensor scale s = mean|W| folded into the LUTs (no multiplies).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SIG_RANGE = 8.0    # LUT input range (pre-activation units) for sigmoid
TANH_RANGE = 4.0   # ... and for tanh


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def ste_floor(x):
    return x + (torch.floor(x) - x).detach()


def ternary(w):
    """BitNet b1.58 absmean ternarisation with straight-through gradient.
    Returns (integer-valued ternary tensor with STE grad, scale)."""
    s = w.abs().mean().clamp(min=1e-5).detach()
    u = w / s
    q = torch.clamp(torch.round(u), -1, 1)
    return u + (q - u).detach(), s


def lut_shift(s, rng):
    """Right-shift k so that 256 LUT entries cover about +-rng of
    pre-activation: one LUT step = (s/127) * 2^k ~= rng/128."""
    k = round(math.log2(rng / 128.0 * 127.0 / float(s)))
    return max(0, min(14, k))


class TernaryRNN(nn.Module):
    def __init__(self, vocab=1024, d_emb=128, hidden=384):
        super().__init__()
        self.V, self.E, self.H = vocab, d_emb, hidden
        self.emb = nn.Parameter(torch.empty(vocab, d_emb).uniform_(-0.5, 0.5))
        fan = d_emb + hidden
        self.wz = nn.Parameter(torch.randn(hidden, fan) / math.sqrt(fan))
        self.wn = nn.Parameter(torch.randn(hidden, fan) / math.sqrt(fan))
        self.bz = nn.Parameter(torch.zeros(hidden))
        self.bn = nn.Parameter(torch.zeros(hidden))
        self.wo = nn.Parameter(torch.randn(vocab, hidden) / math.sqrt(hidden))
        self.bo = nn.Parameter(torch.zeros(vocab))

    # -- pieces shared with the exporter -------------------------------------
    def quant_emb(self):
        return ste_round(torch.clamp(self.emb, -1, 1) * 127.0)

    def gate_params(self):
        wz, sz = ternary(self.wz)
        wn, sn = ternary(self.wn)
        kz = lut_shift(sz, SIG_RANGE)
        kn = lut_shift(sn, TANH_RANGE)
        return wz, sz, kz, wn, sn, kn

    @staticmethod
    def lut_index(acc, b, s, k):
        """acc: integer-valued pre-activation (sum of ternary * int8).
        b: float bias in pre-activation units -> integer acc units."""
        a = acc + ste_round(b * 127.0 / s)
        rc = (1 << k) >> 1
        idx = ste_floor((a + rc) / float(1 << k))
        return torch.clamp(idx, -128, 127)

    def forward(self, tokens, h=None, eos=0):
        """tokens: (B,T) long. Returns logits (B,T,V) and last int8 state."""
        B, T = tokens.shape
        x = F.embedding(tokens, self.quant_emb())                   # (B,T,E) ints
        wz, sz, kz, wn, sn, kn = self.gate_params()
        wo, so = ternary(self.wo)
        E = self.E
        wx = torch.cat([wz[:, :E], wn[:, :E]], 0)                   # (2H,E)
        wh = torch.cat([wz[:, E:], wn[:, E:]], 0)                   # (2H,H)
        xpart = x @ wx.t()                                          # (B,T,2H)
        keep = (tokens != eos).float().unsqueeze(-1)                # reset at <eos>
        if h is None:
            h = torch.zeros(B, self.H)
        stepz = float(sz) / 127.0 * (1 << kz)
        stepn = float(sn) / 127.0 * (1 << kn)
        hs = []
        for t in range(T):
            h = h * keep[:, t]
            acc = xpart[:, t] + h @ wh.t()
            iz = self.lut_index(acc[:, :self.H], self.bz, sz, kz)
            inn = self.lut_index(acc[:, self.H:], self.bn, sn, kn)
            zq = ste_round(128.0 * torch.sigmoid(iz * stepz))
            nq = ste_round(127.0 * torch.tanh(inn * stepn))
            h = h + ste_floor((zq * (nq - h) + 64.0) / 128.0)
            hs.append(h)
        hseq = torch.stack(hs, 1)                                   # (B,T,H) ints
        logits = (hseq @ wo.t()) * (so / 127.0) + self.bo
        return logits, h.detach()

    def n_ternary(self):
        return self.wz.numel() + self.wn.numel() + self.wo.numel()
