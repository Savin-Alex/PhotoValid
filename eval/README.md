# PhotoValid evaluation harness

Calibration ground truth for accuracy work. Before changing any threshold or
adding a model (glasses, pose, background, blur…), measure it here against real
labeled photos instead of guessing.

## Layout

```
eval/
├── evaluate.py          # the harness (committed)
└── photos/              # your labeled photos (gitignored — personal/PII)
    ├── accepted/        # DV-COMPLIANT photos — the tool SHOULD pass them
    └── rejected/        # NON-compliant photos — the tool SHOULD flag them
```

`eval/photos/` is gitignored, so your images never get committed. Create the
folders by just running the script once (it makes them for you), then drop
images in.

## Running

Use the backend virtualenv so MediaPipe/OpenCV are available:

```bash
backend/.venv/bin/python eval/evaluate.py
```

## How to read the output

The tool is scored as a **detector of non-compliant photos**:
- positive class = "non-compliant" (the `rejected/` folder)
- detection = overall `status == "fail"`

So:
- **Recall** = fraction of bad photos caught → *safety* metric. Low recall means
  **false accepts** (a non-compliant photo slipped through) — the dangerous error.
- **Precision** = of the photos we failed, how many were genuinely bad → high
  precision means we rarely fail good photos.
- **Per-check pass-rate**: any check that passes on **< 80% of accepted photos** is
  flagged as likely miscalibrated (false-positive-prone) — exactly how the
  background/sharpness/brightness false positives were caught earlier.

## Tips for building the set

- Aim for at least ~30–50 photos per folder to make the rates meaningful.
- Include hard cases in `rejected/`: glasses, head covering, off-center, tilted,
  too small/large head, busy/colored background, blurry, over/under-exposed.
- Include variety in `accepted/`: different skin tones, hair, lighting, ages.
- A photo's "ground truth" is your judgment of DV compliance per the official
  State Department photo requirements.

## Future extension

For per-check ground truth (e.g. "this photo specifically has glasses"), add an
optional `labels.csv` (`filename,label,failing_checks`) and extend `evaluate.py`
to compute per-check precision/recall. Folder-based overall labels are the v1.
