#!/usr/bin/env python3
"""Evaluate PhotoValid against a labeled photo set.

This is the calibration harness: every accuracy change (thresholds, glasses,
pose, background, etc.) should be measured here against real photos rather than
guessed — guessing is what produced the false positives fixed earlier.

Setup — drop photos into:
    eval/photos/accepted/   DV-compliant photos       (the tool SHOULD pass them)
    eval/photos/rejected/   non-compliant photos      (the tool SHOULD flag them)

Run (use the backend venv so MediaPipe/OpenCV are available):
    backend/.venv/bin/python eval/evaluate.py

Framing — we treat the tool as a DETECTOR OF NON-COMPLIANT photos:
    positive class = "non-compliant" (the rejected/ folder)
    detection      = overall status == "fail"
  so  Recall    = fraction of bad photos we catch   (high = few FALSE ACCEPTS = safe)
      Precision = of those we failed, how many were truly bad (high = few false rejects)
The dangerous error is a FALSE ACCEPT (a non-compliant photo not failed).

A per-check pass-rate table is also printed: any check that passes on <80% of the
ACCEPTED photos is likely miscalibrated (false-positive-prone) and worth a look.

Photos live under eval/photos/ which is .gitignored (they're personal/PII images).
"""
from __future__ import annotations

import glob
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.main import _validate_raw  # noqa: E402

PHOTOS = os.path.join(ROOT, "eval", "photos")
ACCEPTED_DIR = os.path.join(PHOTOS, "accepted")
REJECTED_DIR = os.path.join(PHOTOS, "rejected")


def _photos(directory: str) -> list[str]:
    files: list[str] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files += glob.glob(os.path.join(directory, ext))
        files += glob.glob(os.path.join(directory, ext.upper()))
    return sorted(set(files))


def _evaluate(path: str):
    with open(path, "rb") as fh:
        raw = fh.read()
    t0 = time.perf_counter()
    _status_code, summary, _pil = _validate_raw(raw, "image/jpeg", None)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return summary, elapsed_ms


def _ran_passrate(counts: dict):
    """Pass-rate among checks that actually RAN (skipped excluded).

    Returns (display_string, fraction_or_None). 'skip' means the check was
    skipped on every photo in the group (manual-review only), so it must not be
    read as a calibration problem.
    """
    total = sum(counts.values())
    ran = total - counts.get("skipped", 0)
    if total == 0:
        return "  -", None
    if ran == 0:
        return "skip", None
    frac = counts.get("pass", 0) / ran
    return f"{100 * frac:.0f}%", frac


def main() -> None:
    os.makedirs(ACCEPTED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

    accepted = _photos(ACCEPTED_DIR)
    rejected = _photos(REJECTED_DIR)

    if not accepted and not rejected:
        print("No labeled photos found. Add images to:")
        print(f"  {ACCEPTED_DIR}  (DV-compliant — should PASS)")
        print(f"  {REJECTED_DIR}  (non-compliant — should be FAILED)")
        return

    dist = {"accepted": defaultdict(int), "rejected": defaultdict(int)}
    per_check = defaultdict(lambda: {"acc": defaultdict(int), "rej": defaultdict(int)})
    times: list[float] = []

    def run_set(files: list[str], group: str) -> None:
        print(f"\n{group.title()} ({len(files)}):")
        for path in files:
            summary, ms = _evaluate(path)
            times.append(ms)
            status = summary.get("status", "?")
            dist[group][status] += 1
            grp = "acc" if group == "accepted" else "rej"
            for check in summary.get("checks", []):
                per_check[check.get("name")][grp][check.get("status")] += 1
            print(f"  [{status:>7}] {os.path.basename(path)}  ({ms:.0f} ms)")

    run_set(accepted, "accepted")
    run_set(rejected, "rejected")

    a, r = len(accepted), len(rejected)
    # Detector framing: positive = non-compliant, detection = status == "fail".
    tp = dist["rejected"]["fail"]                 # bad photo correctly failed
    fn = r - tp                                   # bad photo NOT failed -> FALSE ACCEPT
    fp = dist["accepted"]["fail"]                 # good photo wrongly failed
    tn = a - fp

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) else float("nan"))

    print("\n=== Verdict distribution ===")
    print(f"  accepted/  pass={dist['accepted']['pass']}  warning={dist['accepted']['warning']}  fail={dist['accepted']['fail']}  (fails are FALSE REJECTS)")
    print(f"  rejected/  pass={dist['rejected']['pass']}  warning={dist['rejected']['warning']}  fail={dist['rejected']['fail']}  (pass/warning are FALSE ACCEPTS)")

    print("\n=== Detector metrics (positive = non-compliant, detection = status 'fail') ===")
    print(f"  Recall   (bad photos caught):       {recall:.2f}   <- safety: higher = fewer false accepts")
    print(f"  Precision(of failed, truly bad):    {precision:.2f}")
    print(f"  F1:                                 {f1:.2f}")
    print(f"  FALSE ACCEPTS (bad photo not failed): {fn}/{r}")
    print(f"  False rejects (good photo failed):    {fp}/{a}")
    if times:
        ts = sorted(times)
        p90 = ts[max(0, int(len(ts) * 0.9) - 1)]
        print(f"  Latency: mean {sum(times)/len(times):.0f} ms, p90 {p90:.0f} ms")

    print("\n=== Per-check pass-rate on checks that RAN (skipped excluded) ===")
    print(f"  {'Check':34}{'accepted':>10}{'rejected':>10}")
    for name in sorted(per_check):
        ap_str, ap = _ran_passrate(per_check[name]["acc"])
        rp_str, _rp = _ran_passrate(per_check[name]["rej"])
        flag = "  <- often fails/warns on GOOD photos (check calibration)" if (ap is not None and ap < 0.8) else ""
        print(f"  {name:34}{ap_str:>10}{rp_str:>10}{flag}")


if __name__ == "__main__":
    main()
