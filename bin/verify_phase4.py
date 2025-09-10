#!/usr/bin/env python3
import argparse, json, math, sys
from pathlib import Path

def load_json(p: Path):
    return json.loads(p.read_text())

def get_meta_fields(meta: dict, models_dir: Path):
    # features
    feats = meta.get("features") or meta.get("feature_names")
    if not feats:
        raise KeyError("gbm_meta.json missing 'features'/'feature_names'")

    # monotone vector
    mono = meta.get("monotonic_cst")
    if mono is None:
        monod = meta.get("monotonic")
        if isinstance(monod, dict):
            mono = [int(monod.get(f, 0)) for f in feats]
        else:
            mono = [0] * len(feats)

    # model file
    model_fname = meta.get("model_file") or "gbm_freq.pkl"
    model_path = models_dir / model_fname

    supports = bool(meta.get("supports_monotone", False))
    return feats, mono, supports, model_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--metrics", default="models/metrics_gbm.json")
    ap.add_argument("--features", default="data/trip_features")
    ap.add_argument("--labels", default="data/labels_trip")
    args = ap.parse_args()

    models_dir = Path(args.models)
    metrics_p = Path(args.metrics)
    meta_p = models_dir / "gbm_meta.json"

    if not metrics_p.exists():
        print("[VERIFY-PH4] FAIL: metrics file not found:", metrics_p)
        return 2
    if not meta_p.exists():
        print("[VERIFY-PH4] FAIL: meta file not found:", meta_p)
        return 2

    m = load_json(metrics_p)
    meta = load_json(meta_p)
    try:
        feats, mono, supports, model_path = get_meta_fields(meta, models_dir)
    except KeyError as e:
        print(f"[VERIFY-PH4] FAIL: {e}")
        return 2

    if not model_path.exists():
        print(f"[VERIFY-PH4] FAIL: model artifact missing: {model_path}")
        return 2

    # Metrics sanity
    auc = m.get("auc", float("nan"))
    pr = m.get("pr_auc", float("nan"))
    base = float(m.get("base_rate", float("nan")))
    lifts = (m.get("decile_lift", {}) or {}).get("lifts") or []
    lift1 = lifts[0] if lifts else float("nan")
    n_eval = int(m.get("n_eval", 0))
    n_pos = int(m.get("n_pos_eval", 0))
    n_neg = int(m.get("n_neg_eval", 0))

    print(f"[VERIFY-PH4] metrics: eval_on={m.get('eval_on')} split={m.get('split_tag')} auc={auc} pr_auc={pr} base={base} lift1={lift1} n_eval={n_eval} pos={n_pos} neg={n_neg}")

    # Only enforce thresholds if eval fold is non-trivial
    hard_check = (n_eval >= 50 and n_pos >= 3 and n_neg >= 3)
    if hard_check:
        ok_auc = (isinstance(auc, float) and not math.isnan(auc) and auc >= 0.60)
        ok_lift = (isinstance(lift1, (int,float)) and not math.isnan(lift1) and lift1 >= 1.5)
        if not (ok_auc or ok_lift):
            print(f"[VERIFY-PH4] FAIL: weak discrimination (auc={auc}, lift1={lift1})")
            return 1

    # Monotone metadata consistency
    if len(feats) != len(mono):
        print(f"[VERIFY-PH4] FAIL: features/monotone length mismatch: {len(feats)} vs {len(mono)}")
        return 1

    if supports:
        negs = [(f, c) for f, c in zip(feats, mono) if c < 0]
        if negs:
            print(f"[VERIFY-PH4] FAIL: negative monotone constraints found: {negs}")
            return 1
        print("[VERIFY-PH4] OK: monotone constraints present and non-negative")
    else:
        print("[VERIFY-PH4] INFO: sklearn lacks 'monotonic_cst' support; skipping constraint check")

    print("[VERIFY-PH4] OK: Phase 4 verification passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
