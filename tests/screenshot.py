"""Run the ROM headless, type a prompt, wait for generation to finish and
render the MSX SCREEN 0 text display straight from VRAM to a PNG
(openMSX cannot screenshot with renderer=none, so we decode VRAM ourselves).

usage: python tests/screenshot.py --prompt "Once upon a time" --out docs/screen.png
"""
import argparse
import os
import struct
import subprocess
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_golden import BUILD, OPENMSX, ROOT, symbols, tcl_escape  # noqa: E402

# TMS9918A palette
PAL = [(0, 0, 0), (0, 0, 0), (33, 200, 66), (94, 220, 120), (84, 85, 237),
       (125, 118, 252), (212, 82, 77), (66, 235, 245), (252, 85, 84), (255, 121, 120),
       (212, 193, 84), (230, 206, 128), (33, 176, 59), (201, 91, 186), (204, 204, 204),
       (255, 255, 255)]


def png(path, w, h, rows):
    raw = b"".join(b"\0" + bytes(r) for r in rows)

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    open(path, "wb").write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--runs", type=int, default=1, help="prompts typed in a row")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "screen.png"))
    ap.add_argument("--scale", type=int, default=3)
    args = ap.parse_args()
    s = symbols()
    dump = os.path.join(BUILD, "vram.txt").replace("\\", "/")
    tcl = f"""
set renderer none
set throttle off
set ::n 0
debug set_bp {s['ev_ready']} {{}} {{
  if {{$::n == 0}} {{ set ::n 1; poke {s['MAN'] + 24} {args.max_new};
     after time 4 {{type "{tcl_escape(args.prompt)}\\r"}} }}
}}
debug set_bp {s['ev_done']} {{}} {{
  set f [open "{dump}" w]
  binary scan [debug read_block VRAM 0 16384] H* v
  puts $f "[vdpreg 2] [vdpreg 4] [vdpreg 7] $v"
  close $f
  exit
}}
after time 4000 {{exit}}
"""
    p = os.path.join(BUILD, "shot.tcl")
    open(p, "w").write(tcl)
    subprocess.run([OPENMSX, "-machine", "C-BIOS_MSX1", "-carta",
                    os.path.join(BUILD, "msxllm.rom"), "-romtype", "ASCII8", "-script", p],
                   cwd=ROOT, timeout=1800, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    r2, r4, r7, v = open(dump).read().split()
    vram = bytes.fromhex(v)
    name, pat = int(r2) * 0x400, (int(r4) & 7) * 0x800
    fg, bg = PAL[int(r7) >> 4], PAL[int(r7) & 15]
    W, H, S, B = 240, 192, args.scale, 8
    rows = []
    for y in range(H + 2 * B):
        row = []
        for x in range(W + 2 * B):
            cx, cy = x - B, y - B
            on = False
            if 0 <= cx < W and 0 <= cy < H:
                ch = vram[name + (cy // 8) * 40 + cx // 6]
                on = bool(vram[pat + ch * 8 + cy % 8] & (0x80 >> (cx % 6)))
            row.extend((fg if on else bg) * S)
        for _ in range(S):
            rows.append(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    png(args.out, (W + 2 * B) * S, (H + 2 * B) * S, rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
