"""One-shot build: checkpoint -> ROM data banks -> assembled .rom

usage: python build.py [--ckpt runs/model.pt] [--temp 0.8] [--topk 8] [--eval]
Produces build/msxllm.rom (ASCII8 MegaROM) and build/msxllm.sym
(also regenerates docs/banner.svg and the boot-splash bitmap build/splash.bin).
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SJASM = os.environ.get("SJASMPLUS", os.path.join(
    ROOT, "tools", "sjasmplus", "sjasmplus-1.24.0.win", "sjasmplus.exe"))


def main():
    py = sys.executable
    subprocess.run([py, os.path.join(ROOT, "docs", "make_banner.py")], check=True, cwd=ROOT)
    subprocess.run([py, os.path.join(ROOT, "export", "pack.py")] + sys.argv[1:],
                   check=True, cwd=ROOT)
    r = subprocess.run([SJASM, "--nologo", "--msg=war", "--sym=build/msxllm.sym",
                        "z80/engine.asm"], cwd=ROOT, capture_output=True, text=True)
    # the INCBIN of the data banks runs past 0FFFFh in sjasmplus' address space;
    # that warning is expected (raw output mode, addresses are irrelevant there)
    msgs = [l for l in (r.stdout + r.stderr).splitlines() if "RAM limit exceeded" not in l]
    if r.returncode or any("error" in l.lower() for l in msgs):
        print("\n".join(msgs))
        sys.exit("assembly failed")
    rom = os.path.join(ROOT, "build", "msxllm.rom")
    print(f"built {rom} ({os.path.getsize(rom) // 1024} KB)")


if __name__ == "__main__":
    main()
