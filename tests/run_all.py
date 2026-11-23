"""Run the golden-test suite and the timing profile; write build/test_results.json.

usage: python tests/run_all.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_golden import BUILD, run_case  # noqa: E402

CASES = [
    dict(prompt="Once upon a time there was a little", max_new=16),
    dict(prompt="Lily wanted to play with her", greedy=True, max_new=12),
    dict(prompt="", max_new=12),
    dict(prompt='Tom said, "Can we go to the park?" Mom', greedy=True, max_new=8),
    dict(prompt="The dog was sad because", max_new=8, machine="C-BIOS_MSX1_EU"),
    dict(prompt="One day, a bird", max_new=8, profile=True),
]


def main():
    results = [run_case(**c) for c in CASES]
    ok = all(r["passed"] for r in results)
    spt = [r["emulated_s_per_token"] for r in results if r["emulated_s_per_token"]]
    summary = {
        "all_passed": ok, "cases": len(results),
        "emulated_s_per_token_mean": sum(spt) / len(spt),
        "results": results,
    }
    json.dump(summary, open(os.path.join(BUILD, "test_results.json"), "w"), indent=1)
    print(f"\n{sum(r['passed'] for r in results)}/{len(results)} cases bit-exact; "
          f"mean emulated {summary['emulated_s_per_token_mean']:.2f} s/token")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
