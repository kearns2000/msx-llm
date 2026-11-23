"""Generate the "MSX-LLM" banner in a 5x5 pixel font, one TMS9918A palette
colour per letter:
  docs/banner.svg   README banner (rounded blocks on a dark panel)
  build/splash.bin  SCREEN 2 boot-splash bitmap for the Z80 engine:
                    5 character rows (y 56..95) x 32 columns, 1280 bytes of
                    pattern data followed by 1280 bytes of colour data, laid
                    out exactly as VRAM 0700h / 2700h in SCREEN 2.

usage: python docs/make_banner.py
"""
import os

FONT = {
    "M": ["X...X", "XX.XX", "X.X.X", "X...X", "X...X"],
    "S": [".XXX", "X...", ".XX.", "...X", "XXX."],
    "X": ["X...X", ".X.X.", "..X..", ".X.X.", "X...X"],
    "-": ["...", "...", "XXX", "...", "..."],
    "L": ["X...", "X...", "X...", "X...", "XXXX"],
}
WORD = "MSX-LLM"
# TMS9918A colours: medium red, dark yellow, cyan, white,
# medium green, light blue, magenta
COLOURS = ["#fc5554", "#d4c154", "#42ebf5", "#ffffff", "#21c842", "#7d76fc", "#c95bba"]
BG, DOT = "#1b2b3a", "#24384b"
CELL, GAP, PAD = 20, 4, 24
TMS = [8, 10, 7, 15, 2, 5, 13]     # same colours as TMS9918A palette indices
SPLASH_ROW = 7                     # first character row of the banner (y = 56)


def banner_columns():
    """[(letter index or None, [5 bits]), ...] - 36 columns incl. 1-col gaps"""
    cols = []
    for i, ch in enumerate(WORD):
        g = FONT[ch]
        for c in range(len(g[0])):
            cols.append((i, [g[r][c] == "X" for r in range(5)]))
        if i < len(WORD) - 1:
            cols.append((None, [False] * 5))
    return cols


def splash_bin():
    """SCREEN 2 bitmap: 6x6 blocks with cut corners on a 7-pixel pitch.
    Each 8x1 pixel span may hold only one foreground colour (TMS9918A rule);
    the one-column gap between letters guarantees that, and we assert it."""
    W, H = 256, 40
    px = [[None] * W for _ in range(H)]
    cols = banner_columns()
    x0 = (W - (len(cols) * 7 - 1)) // 2
    for ci, (li, bits) in enumerate(cols):
        for r, on in enumerate(bits):
            if not on:
                continue
            for dy in range(6):
                for dx in range(6):
                    if dy in (0, 5) and dx in (0, 5):
                        continue
                    px[r * 7 + dy][x0 + ci * 7 + dx] = TMS[li]
    pat = bytearray(1280)
    col = bytearray(1280)
    for crow in range(5):
        for ccol in range(32):
            for line in range(8):
                y = crow * 8 + line
                span = px[y][ccol * 8:ccol * 8 + 8]
                fg = {c for c in span if c is not None}
                assert len(fg) <= 1, "colour clash"
                k = crow * 256 + ccol * 8 + line
                pat[k] = sum(0x80 >> b for b, c in enumerate(span) if c is not None)
                col[k] = ((fg.pop() if fg else 1) << 4) | 1     # fg / black bg
    return bytes(pat + col)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root, "build"), exist_ok=True)
    open(os.path.join(root, "build", "splash.bin"), "wb").write(splash_bin())
    print("wrote build/splash.bin (2560 bytes)")
    cols = sum(len(FONT[c][0]) + 1 for c in WORD) - 1
    w = PAD * 2 + cols * (CELL + GAP) - GAP
    h = PAD * 2 + 5 * (CELL + GAP) - GAP
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
           f'width="{w}" height="{h}" role="img" aria-label="MSX-LLM">',
           f'<rect width="{w}" height="{h}" rx="12" fill="{BG}"/>']
    x0 = 0
    for i, ch in enumerate(WORD):
        glyph = FONT[ch]
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                x = PAD + (x0 + c) * (CELL + GAP)
                y = PAD + r * (CELL + GAP)
                fill = COLOURS[i] if bit == "X" else DOT
                out.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                           f'rx="4" fill="{fill}"/>')
        x0 += len(glyph[0]) + 1
    out.append("</svg>")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "banner.svg")
    open(path, "w").write("\n".join(out) + "\n")
    print("wrote", path, f"({w}x{h})")


if __name__ == "__main__":
    main()
