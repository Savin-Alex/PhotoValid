# DV Photo Validator — Technical Summary

A FastAPI service that checks photos against the U.S. Department of State
Diversity Visa (DV) Lottery photo requirements, with a single-page browser UI.

**Repository**: https://github.com/Savin-Alex/PhotoValid

> This document describes what the code **actually does** today. Some DV
> requirements are not yet auto-validated and are reported as "manual review"
> rather than silently passed — see "Known limitations" below.

---

## What is validated

Each check returns a status of `pass`, `warning`, `fail`, or `skipped`
(`skipped` = could not be auto-verified → manual review). The overall **score**
is the percentage of checks that pass; the overall **status** is `fail` if any
check fails, `warning` if any check warns/skips (and none fail), else `pass`.

### Technical (`backend/validators/tech.py`)
- **Dimensions** — square, 600×600 to 1200×1200 px
- **File size** — ≤ 240 KB
- **Compression ratio** — ≤ 20:1 (uncompressed `w*h*3` ÷ file size)
- **File format** — JPEG (declared content-type and detected format)
- **Color model** — 24-bit color in sRGB (rejects grayscale; flags non-sRGB ICC profiles)
- **Brightness / contrast** — heuristic exposure check on the central (face)
  region, so a plain white background doesn't skew the reading
- **Image noise** — flat-region noise floor (10th-percentile block std); low = clean
- **Photo date** — ≤ 6 months old *when an EXIF capture date is present*;
  reported as unverified otherwise (no guessing from file timestamps)

### Biometric (`backend/validators/bio.py`, MediaPipe Face Detection + Face Mesh)
- **One person only** — exactly one detected face
- **Head height** — 50–69% of image height
- **Eye level** — 56–69% from the bottom
- **Head centering** — within ±5% of horizontal center (heuristic)
- **Background** — plain white/off-white, sampled at the top corners in LAB
  (brightness/uniformity/neutrality); avoids the subject's hair/shoulders
- **Sharpness** (face-region Laplacian variance) and **face lighting** balance (heuristics)
- **Eyes open** — Eye Aspect Ratio (EAR) from eyelid landmarks
- **Gaze / looking at camera** — horizontal iris centering between the eye corners
- **Neutral expression** — mouth-openness (Mouth Aspect Ratio)
- **Head tilt** — in-plane roll from the inter-eye line angle
- **Glasses** — best-effort edge heuristic on the nose bridge; warns on likely
  eyeglasses, never confidently clears (a negative stays manual review)
- **Red-eye** — best-effort color heuristic on the iris regions; warns on likely
  red-eye, never confidently clears

### Tamper heuristics (`backend/validators/tamper.py`)
- Error Level Analysis (recompression difference)
- EXIF/metadata presence
- Image noise level (flat-region noise floor)
- Color-channel balance

These tamper checks are heuristics, not forensic guarantees.

---

## Detection pipeline (biometric)

1. EXIF orientation correction (`ImageOps.exif_transpose`).
2. Face detection (MediaPipe FaceDetection, short-range model).
3. Landmarks (MediaPipe FaceMesh, 468 points + iris via `refine_landmarks`).
   The FaceDetection and FaceMesh graphs are created once and reused across
   requests (lock-guarded) to avoid per-request rebuild latency.
4. Top-of-head estimate: contrast/gradient scan upward from the forehead
   landmark, with a margin fallback when no clear edge is found.
5. Eye level: iris centers → eye corners → bounding-box midpoint fallback, with
   a sanity check (30–80% from bottom).
6. Measurements: head ratio, eye level, horizontal center offset.
7. Manual-override mode: the UI can send normalized `top`/`eye`/`chin` line
   positions, which bypass detection and drive the geometry checks directly.

The response always includes a `faceBox` (with `method`/`top_method`/`eye_method`
metadata) so the frontend overlay can render even when detection is uncertain.

---

## Known limitations (NOT auto-validated)

These DV requirements are real but not implemented; they are surfaced for manual
review rather than passed silently:

- Headgear/hats, yaw/pitch head rotation (only in-plane tilt is checked)
- Filter / digital-alteration detection
- 300-dpi resolution (only relevant for scanned prints, not digital uploads)
- File-name rules and metadata stripping
- The glasses heuristic is uncalibrated (no labelled training/test set) — it can
  miss glasses and can false-positive; it is an aid, not a guarantee.

---

## Tech stack

- **Backend**: FastAPI, Uvicorn, Pillow, OpenCV (headless), MediaPipe, NumPy, piexif
- **Frontend**: HTML/CSS/vanilla JS, Canvas overlay, EN/RU i18n, dark mode
- **Runtime**: Python 3.11.9 (see `runtime.txt` / `render.yaml`)

## Project structure

```
PhotoValid/
├── backend/
│   ├── main.py                # FastAPI app, POST /api/validate, serves frontend
│   └── validators/
│       ├── utils.py           # helpers, json_param
│       ├── tech.py            # technical checks
│       ├── bio.py             # biometric checks (MediaPipe)
│       └── tamper.py          # tamper heuristics
├── frontend/index.html        # single-page UI with overlay
├── tests/                     # pytest suite
├── conftest.py                # makes repo root importable for tests
├── render.yaml / runtime.txt  # Render deployment
└── requirements*.txt
```

## Running

```bash
# Backend (also serves the frontend)
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000

# Tests (from repo root)
pip install -r requirements-dev.txt
python -m pytest
```

---

## Disclaimer

Automated analysis based on publicly available DV Lottery photo requirements.
Always verify against the official U.S. Department of State guidance before
submitting. This tool does not guarantee acceptance.
