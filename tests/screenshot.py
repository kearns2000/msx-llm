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
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "screen.png"))
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--stage", default="done", choices=["splash", "ready", "typed", "done"],
                    help="splash: SCREEN 2 boot splash; ready: waiting for input; "
                         "typed: prompt typed, before RETURN; done: after generation")
    ap.add_argument("--splash", action="store_true", help="same as --stage splash")
    args = ap.parse_args()
    stage = "splash" if args.splash else args.stage
    s = symbols()
    dump = os.path.join(BUILD, "vram.txt").replace("\\", "/")
    grab = f"""
  set f [open "{dump}" w]
  binary scan [debug read_block VRAM 0 16384] H* v
  puts $f "[vdpreg 0] [vdpreg 2] [vdpreg 3] [vdpreg 4] [vdpreg 7] $v"
  close $f
  exit
"""
    head = "set renderer none\nset throttle off\nset ::n 0\nafter time 4000 {exit}\n"
    if stage == "splash":
        tcl = head + f"debug set_bp {s['splash.w']} {{}} {{ after time 1 {{ {grab} }} }}\n"
    elif stage == "ready":
        tcl = head + f"""debug set_bp {s['ev_ready']} {{}} {{
  if {{$::n == 0}} {{ set ::n 1; after time 1 {{ {grab} }} }} }}
"""
    elif stage == "typed":
        tcl = head + f"""debug set_bp {s['ev_ready']} {{}} {{
  if {{$::n == 0}} {{ set ::n 1; after time 4 {{type "{tcl_escape(args.prompt)}"}};
     after time 12 {{ {grab} }} }} }}
"""
    else:
        tcl = head + f"""debug set_bp {s['ev_ready']} {{}} {{
  if {{$::n == 0}} {{ set ::n 1; poke {s['MAN'] + 24} {args.max_new};
     after time 4 {{type "{tcl_escape(args.prompt)}\\r"}} }} }}
debug set_bp {s['ev_done']} {{}} {{ {grab} }}
"""
    if os.path.exists(dump):
        os.remove(dump)
    p = os.path.join(BUILD, "shot.tcl")
    open(p, "w").write(tcl)
    subprocess.run([OPENMSX, "-machine", "C-BIOS_MSX1", "-carta",
                    os.path.join(BUILD, "msxllm.rom"), "-romtype", "ASCII8", "-script", p],
                   cwd=ROOT, timeout=1800, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(dump):
        raise SystemExit("openMSX produced no VRAM dump (timeout or crash)")
    r0, r2, r3, r4, r7, v = open(dump).read().split()
    vram = bytes.fromhex(v)
    r0, r2, r3, r4, r7 = map(int, (r0, r2, r3, r4, r7))
    name = r2 * 0x400
    S, B = args.scale, 8
    border = PAL[r7 & 15]
    if r0 & 2:          # SCREEN 2 (Graphic 2): 2 colours per 8x1 span
        W, H = 256, 192
        pbase = 0x2000 if r4 & 4 else 0
        cbase = 0x2000 if r3 & 0x80 else 0

        def pixel(cx, cy):
            n = vram[name + (cy // 8) * 32 + cx // 8]
            off = (cy // 64) * 0x800 + n * 8 + cy % 8
            c = vram[cbase + off]
            on = vram[pbase + off] & (0x80 >> (cx % 8))
            k = (c >> 4) if on else (c & 15)
            return PAL[k] if k else border
    else:               # SCREEN 0 (text 40x24)
        W, H = 240, 192
        pat = (r4 & 7) * 0x800
        fg, bg = PAL[r7 >> 4], PAL[r7 & 15]

        def pixel(cx, cy):
            ch = vram[name + (cy // 8) * 40 + cx // 6]
            return fg if vram[pat + ch * 8 + cy % 8] & (0x80 >> (cx % 6)) else bg
    rows = []
    for y in range(H + 2 * B):
        row = []
        for x in range(W + 2 * B):
            cx, cy = x - B, y - B
            row.extend((pixel(cx, cy) if 0 <= cx < W and 0 <= cy < H else border) * S)
        for _ in range(S):
            rows.append(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    png(args.out, (W + 2 * B) * S, (H + 2 * B) * S, rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
