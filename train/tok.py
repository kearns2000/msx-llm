"""Tiny word/char tokenizer that the Z80 can run.

Vocabulary = <eos> + every printable ASCII char seen + the most frequent
"pieces" (words with/without a leading space, short punctuation runs).

Encoding is GREEDY LONGEST MATCH over the raw text. That is trivially
implementable on a Z80 (scan the vocab, keep the longest prefix match) and is
what both the trainer and the MSX use, so tokenisation is bit-identical.

Text normalisation: unicode quotes/dashes are folded to ASCII, anything else
non-printable is dropped, whitespace runs collapse to one space, and every
story is prefixed with a single space (so the first word looks like any other).
"""
import json
import re
from collections import Counter

EOS = 0
_FOLD = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}
_ws = re.compile(r"\s+")
_piece = re.compile(r" ?[A-Za-z]+(?:'[a-z]+)?| ?[^A-Za-z ]+")


def normalise(text: str) -> str:
    for k, v in _FOLD.items():
        text = text.replace(k, v)
    text = "".join(ch for ch in text if 32 <= ord(ch) < 127 or ch in "\n\t")
    text = _ws.sub(" ", text).strip()
    return " " + text if text else ""


def iter_stories(path, max_bytes=None):
    """Yield normalised stories from a TinyStories '<|endoftext|>' file."""
    buf = []
    read = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            read += len(line)
            if "<|endoftext|>" in line:
                head, _, tail = line.partition("<|endoftext|>")
                buf.append(head)
                s = normalise(" ".join(buf))
                if s:
                    yield s
                buf = [tail]
                if max_bytes and read > max_bytes:
                    return
            else:
                buf.append(line)
    s = normalise(" ".join(buf))
    if s:
        yield s


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = list(vocab)          # id -> string ('' for <eos>)
        self.index = {s: i for i, s in enumerate(self.vocab) if s}
        self.maxlen = max(len(s) for s in self.vocab)
        self._cache = {}

    # --- greedy longest match (reference semantics, mirrored on the Z80) ---
    def _encode_seg(self, seg):
        out = []
        p = 0
        n = len(seg)
        while p < n:
            for L in range(min(self.maxlen, n - p), 0, -1):
                t = self.index.get(seg[p:p + L])
                if t is not None:
                    out.append(t)
                    p += L
                    break
            else:  # unknown char: skip it (Z80 does the same)
                p += 1
        return out

    def encode(self, text):
        """Encode already-normalised text. Tokens never contain a space except
        as their first char, so greedy matching splits cleanly at spaces and we
        can cache per space-delimited segment."""
        out = []
        for i, w in enumerate(text.split(" ")):
            if not w:
                continue
            seg = " " + w if (i > 0) else w
            ids = self._cache.get(seg)
            if ids is None:
                ids = self._encode_seg(seg)
                if len(self._cache) < 2_000_000:
                    self._cache[seg] = ids
            out.extend(ids)
        return out

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab}, f, indent=0)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["vocab"])


def build_vocab(stories, size, min_count=5):
    """Classic BPE (Sennrich et al.) learned on pre-token counts.
    Pre-tokens never contain an inner space, so neither does any token.
    All intermediate merges are kept as vocab entries; encoding is then
    greedy longest match (not merge replay), which is what the Z80 does."""
    chars = Counter()
    pieces = Counter()
    for s in stories:
        chars.update(s)
        pieces.update(_piece.findall(s))
    base = sorted(c for c in chars if 32 <= ord(c) < 127)
    vocab = [""] + base
    words = [[list(w), n] for w, n in pieces.items() if n >= min_count]
    while len(vocab) < size:
        pairs = Counter()
        for sym, n in words:
            for a, b in zip(sym, sym[1:]):
                pairs[a, b] += n
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        ab = a + b
        vocab.append(ab)
        for w in words:
            sym = w[0]
            if len(sym) < 2:
                continue
            i, out = 0, []
            while i < len(sym):
                if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
                    out.append(ab)
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            w[0] = out
    return vocab
