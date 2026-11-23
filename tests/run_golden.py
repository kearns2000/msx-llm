"""Golden test: run the ROM headless in openMSX (C-BIOS MSX1, Z80 3.58 MHz),
type a prompt, let it generate, dump RAM, and compare bit-for-bit with the
Python integer reference (train/intref.py). Also measures emulated s/token.

usage: python tests/run_golden.py [--prompt "Once upon a time"] [--greedy]
                                  [--max-new 12] [--machine C-BIOS_MSX1]
Exit code 0 = all comparisons passed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
import intref  # noqa: E402
from tok import Tokenizer, normalise  # noqa: E402

OPENMSX = os.environ.get("OPENMSX", os.path.join(ROOT, "tools", "openmsx", "openmsx.exe"))
BUILD = os.path.join(ROOT, "build")


def symbols():
    syms = {}
    for line in open(os.path.join(BUILD, "msxllm.sym")):
        m = re.match(r"(\S+): EQU 0x([0-9A-Fa-f]+)", line)
        if m:
            syms[m.group(1)] = int(m.group(2), 16)
    return syms


TCL = r"""
set renderer none
set throttle off
set ::typed 0
set ::times {}
set ::events {}
proc hexdump {addr len} {
    binary scan [debug read_block memory $addr $len] H* h
    return $h
}
proc on_ready {} {
    if {$::typed} return
    set ::typed 1
    poke %(greedy_addr)d %(greedy)d
    poke %(maxnew_addr)d %(max_new)d
    after time 4 {type "%(prompt)s\r"}
}
proc on_done {} {
    set f [open "%(out)s" w]
    puts $f "times [join $::times { }]"
    puts $f "events [join $::events { }]"
    if {[catch {set scr [get_screen]} err]} {set scr "ERR $err"}
    puts $f "screen [string map [list \n |] $scr]"
    puts $f "seed [hexdump %(seed0)d 2]"
    puts $f "ntok [hexdump %(ntok)d 2]"
    puts $f "hist [hexdump %(hist)d 512]"
    puts $f "plen [hexdump %(plen)d 1]"
    puts $f "prompt [hexdump %(prompt_buf)d 128]"
    puts $f "h [hexdump %(hvec)d %(H)d]"
    puts $f "y [hexdump %(y)d %(ybytes)d]"
    close $f
    exit
}
debug set_bp %(ev_ready)d {} on_ready
debug set_bp %(ev_token)d {} {lappend ::times [machine_info time]}
debug set_bp %(ev_done)d {} on_done
after time 4000 {exit}
%(profile)s
"""

PROFILE_LABELS = {  # label -> phase that starts there
    "feed": "embedding copy", "matvec": "bias copy", "matvec.chunk": "table build",
    "matvec.seg": "ternary kernel", "gate_update": "gate update (LUT, mul)",
    "out_layer": "setup", "sample": "top-K + sampling", "print_tok": "print token",
    "ev_token": "bookkeeping",
}


def tcl_escape(s):
    return "".join("\\" + c if c in '\\"[]$' else c for c in s)


def run_msx(prompt, greedy, max_new, machine, profile=False):
    s = symbols()
    prof = ""
    if profile:
        prof = "\n".join(
            "debug set_bp %d {} {if {[llength $::times]} {lappend ::events %s [machine_info time]}}"
            % (s[lab], lab) for lab in PROFILE_LABELS)
    im = intref.load(os.path.join(BUILD, "model_int.npz"))
    out = os.path.join(BUILD, "golden_out.txt").replace("\\", "/")
    if os.path.exists(out):
        os.remove(out)
    tcl = TCL % dict(
        greedy_addr=s["greedy"], greedy=int(greedy), maxnew_addr=s["MAN"] + 24,
        max_new=max_new, prompt=tcl_escape(prompt), out=out,
        seed0=s["seed0"], ntok=s["ntok"], hist=0xAE00, plen=s["plen"],
        prompt_buf=0xAD00, hvec=0xA800 + 128, H=im.H, y=0xA000, ybytes=2 * im.V,
        ev_ready=s["ev_ready"], ev_token=s["ev_token"], ev_done=s["ev_done"],
        profile=prof)
    tclpath = os.path.join(BUILD, "golden.tcl")
    open(tclpath, "w").write(tcl)
    t0 = time.time()
    subprocess.run([OPENMSX, "-machine", machine,
                    "-carta", os.path.join(BUILD, "msxllm.rom"), "-romtype", "ASCII8",
                    "-script", tclpath], cwd=ROOT, timeout=1800,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wall = time.time() - t0
    if not os.path.exists(out):
        raise SystemExit("openMSX produced no dump (timeout or crash)")
    d = {}
    for line in open(out):
        k, _, v = line.strip().partition(" ")
        d[k] = v
    return d, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Once upon a time there was a little")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--machine", default="C-BIOS_MSX1")
    ap.add_argument("--json", help="write results here")
    ap.add_argument("--profile", action="store_true", help="per-phase time breakdown")
    args = ap.parse_args()

    im = intref.load(os.path.join(BUILD, "model_int.npz"))
    tok = Tokenizer.load(os.path.join(ROOT, "data", "vocab.json"))
    d, wall = run_msx(args.prompt, args.greedy, args.max_new, args.machine, args.profile)

    b = lambda k: bytes.fromhex(d[k])  # noqa: E731
    seed = int.from_bytes(b("seed"), "little")
    ntok = int.from_bytes(b("ntok"), "little")
    hist = np.frombuffer(b("hist"), dtype="<u2")[:ntok].tolist()
    h_z80 = np.frombuffer(b("h"), dtype=np.int8)
    y_z80 = np.frombuffer(b("y"), dtype="<i2").astype(np.int64)
    times = [float(t) for t in d["times"].split()] if d.get("times") else []

    ptxt = normalise(args.prompt)
    p_ids = tok.encode(ptxt)
    ref = im.generate(p_ids, seed, max_new=args.max_new, greedy=args.greedy)
    n_prompt = 1 + len(p_ids)
    z_prompt = hist[1:n_prompt]
    z_gen = hist[n_prompt:]

    checks = {
        "prompt buffer": b("prompt")[: b("plen")[0]].decode("ascii", "replace") == ptxt,
        "prompt tokens": z_prompt == p_ids,
        "generated tokens": z_gen == ref["tokens"],
        "final h (int8 x H)": np.array_equal(h_z80, ref["h"]),
        "final logits (int16 x V)": np.array_equal(y_z80, ref["logits"]),
    }
    per_tok = np.diff(times)
    res = {
        "prompt": args.prompt, "seed": seed, "greedy": args.greedy,
        "z80_text": tok.decode(z_gen), "ref_text": tok.decode(ref["tokens"]),
        "z80_tokens": z_gen, "ref_tokens": ref["tokens"],
        "checks": checks,
        "emulated_s_per_token": float(per_tok.mean()) if len(per_tok) else None,
        "emulated_s_per_token_min_max": [float(per_tok.min()), float(per_tok.max())] if len(per_tok) else None,
        "emulated_s_prompt_to_first_token": None,
        "tokens_timed": len(times), "wall_clock_s": round(wall, 1),
    }
    print(f"prompt : {ptxt!r}  ({len(p_ids)} tokens)  seed {seed:#06x}")
    print(f"MSX    : {res['z80_text']!r}")
    print(f"python : {res['ref_text']!r}")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if len(per_tok):
        print(f"emulated time per token: {per_tok.mean():.2f} s "
              f"(min {per_tok.min():.2f}, max {per_tok.max():.2f}, n={len(per_tok)})")
    if not checks["final logits (int16 x V)"]:
        diff = np.nonzero(y_z80 != ref["logits"])[0]
        print("  first logit diffs:", [(int(i), int(y_z80[i]), int(ref['logits'][i])) for i in diff[:8]])
    if not checks["final h (int8 x H)"]:
        diff = np.nonzero(h_z80 != ref["h"])[0]
        print("  first h diffs:", [(int(i), int(h_z80[i]), int(ref['h'][i])) for i in diff[:8]])
    if args.profile and d.get("events"):
        ev = d["events"].split()
        ev = [(ev[i], float(ev[i + 1])) for i in range(0, len(ev), 2)]
        pts = [i for i, (lab, _) in enumerate(ev) if lab == "print_tok"]
        ev = ev[pts[0]:pts[-1] + 1]       # whole cycles: print -> feed -> out -> sample
        ntoks = len(pts) - 1
        acc = {}
        for (lab, t0), (_, t1) in zip(ev, ev[1:]):
            ph = PROFILE_LABELS[lab]
            acc[ph] = acc.get(ph, 0.0) + (t1 - t0)
        tot = sum(acc.values())
        print(f"profile over {ntoks} tokens (s/token, share):")
        for ph, v in sorted(acc.items(), key=lambda kv: -kv[1]):
            print(f"  {ph:26s} {v / ntoks:7.3f}  {100 * v / tot:5.1f}%")
        res["profile_s_per_token"] = {k: v / ntoks for k, v in acc.items()}
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
