# PhotoValid — Backlog

Captured from the production-roadmap review (May 2026). Ordered by priority; notes
preserve the rationale and caveats so context isn't lost.

## Prerequisites (cross-cutting)

- **Labeled dataset of accepted/rejected DV photos.** The highest-leverage
  investment. Needed to calibrate thresholds and validate any ML work below.
  Without it, accuracy changes are guesswork — the root cause of the false
  positives fixed in the May 2026 hardening pass.
- **Mind Render free-tier limits** (512 MB RAM, cold starts after 15 min idle).
  MediaPipe + OpenCV already strain it; profile memory before adding models — a
  heavier model may require a paid plan or a separate inference service.
- **Privacy/legal.** Anything that stores user photos handles sensitive biometric
  PII of visa applicants. Default to stateless; if storing, get retention,
  consent, and GDPR right. Especially relevant now the project is proprietary.

## Tier 1 — high ROI

### 1. PDF report generation
- Downloadable validation report: overall score, per-check results, and the
  annotated overlay (top-of-head / eye / chin guides).
- Effort: **low** (reportlab or weasyprint). Generate on the fly / stateless
  (no PII storage).
- Strong candidate to be a **premium (paid)** feature.

### 2. Additional languages
- i18n already exists (EN/RU) — just extend the dictionary in `frontend/index.html`.
- Prioritize by DV applicant volume (e.g. Spanish, French, Arabic, Amharic,
  Portuguese).
- Effort: **low**. Good growth lever (DV applicants are global).

### 3. Glasses detection (lightweight model)
- Closes the biggest real accuracy gap: eyeglasses are a top DV disqualifier and
  are currently only a flag-only edge heuristic (`bio.check_glasses`).
- Use a **small classifier on the eye-region ROI**, not a heavy DL model (memory).
- **Blocked on the labeled dataset** for calibration/validation.
- Headgear/hats is lower priority (rarer; has a religious exception → a detector
  must warn, never hard-fail).

## Tier 2 — monetization

### 4. Payment integration (premium features)
- Stripe is the obvious choice.
- Depends on: (a) a sellable feature shipped (PDF report, batch), (b) a clear
  free vs. paid line, and (c) user accounts (auth + history).
- Recommended sequence: ship a premium-worthy feature → add accounts → integrate
  Stripe → gate the premium feature behind the paywall.
- ⚠️ Couples tightly to the privacy/legal note above (accounts imply stored PII).
