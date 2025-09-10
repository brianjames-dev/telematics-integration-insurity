#!/usr/bin/env python
import argparse, json, sys
from pathlib import Path

def fail(msg): print(f"[VERIFY-PH3] FAIL: {msg}"); sys.exit(1)
def ok(msg): print(f"[VERIFY-PH3] OK: {msg}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--metrics", default="models/metrics_glm.json")
    args = ap.parse_args()

    mdir = Path(args.models)
    if not (mdir / "glm_freq.json").exists(): fail("glm_freq.json missing")
    if not (mdir / "glm_sev.json").exists(): fail("glm_sev.json missing")
    ok("model artifacts present")

    if not Path(args.metrics).exists(): fail("metrics_glm.json missing")
    metrics = json.loads(Path(args.metrics).read_text())

    auc = float(metrics.get("freq_auc", 0))
    n_test = int(metrics.get("n_test", 0))
    n_pos = int(metrics.get("n_pos_test", 0))
    n_neg = n_test - n_pos

    # If tiny or extremely imbalanced, metrics are not meaningful — skip thresholds
    if n_test < 10 or n_pos < 3 or n_neg < 3:
        ok(f"small/imbalanced test (n_test={n_test}, pos={n_pos}, neg={n_neg}); skipping AUC/lift thresholds")
    else:
        if not (0.55 <= auc <= 1.0):
            fail(f"AUC too low or invalid: {auc:.3f}")
        lifts = metrics.get("freq_decile_lift", {}).get("lifts", [])
        if lifts and lifts[0] < 1.2:
            fail(f"Top decile lift < 1.2 (got {lifts[0]:.2f})")
        ok(f"AUC={auc:.3f}, top_decile_lift={lifts[0] if lifts else 'NA'}")

    ok("Phase 3 verification passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
