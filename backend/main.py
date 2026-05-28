from __future__ import annotations
from fastapi import FastAPI, File, UploadFile, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from typing import Optional, Callable, Tuple
import io
import logging
import os
import platform
import sys
import json

# --- Ensure internal imports work ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from validators.utils import load_image_bytes, pil_to_cv
from validators.utils import json_param
from validators.tech import TechValidator
from validators.bio import BioValidator
from validators.tamper import TamperValidator
from validators import bio as _bio
from report import build_report_pdf

__version__ = "0.2.0"

logger = logging.getLogger("photo_valid")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "20000000"))

# --- FastAPI App ---
app = FastAPI(title="DV Photo Validator", version="0.1.0")

# --- CORS ---
# Public, credential-free API: allow any origin but DO NOT allow credentials.
# Wildcard origin + credentials is invalid per the CORS spec and would let any
# site make credentialed cross-origin requests; this API uses no cookies/auth.
# Set CORS_ORIGINS (comma-separated) to lock this down to your domain(s).
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
# Hard DV disqualifiers. A FAIL in one of these makes the overall result "fail";
# a fail in any other (advisory) check only produces a "warning".
CRITICAL_CHECKS = {
    "Photo Dimensions",
    "File Size",
    "File Format",
    "Color Model",
    "Compression Ratio",
    "Face Detection",
    "One Person Only",
    "Head Height",
    "Eye Level",
    # Whole-validator crash fallbacks (from _safe_validator). A crashed Technical
    # or Biometric validator means must-pass checks never ran -> must not look like
    # a clean pass. (Tamper is advisory, so its fallback is intentionally NOT here.)
    "Technical Validation",
    "Biometric Validation",
}


def _summarize_response(
    technical: list[dict],
    biometric: list[dict],
    tamper: list[dict],
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict:
    all_results = technical + biometric + tamper

    # Stamp the response contract: every check carries a `critical` flag, and any
    # skipped check carries a `skipped_reason` (so the UI/clients can reason about it).
    for r in all_results:
        r["critical"] = r.get("name") in CRITICAL_CHECKS
        if r.get("status") == "skipped":
            r.setdefault("skipped_reason", r.get("recommendation"))

    # "skipped" = could not be auto-verified. These do NOT count toward the score.
    # A NON-critical skip is harmless manual-review. A CRITICAL skip means a
    # must-pass requirement could not be verified, so the result must NOT look like
    # a clean pass (it forces at least a "warning").
    ran = [r for r in all_results if r.get("status") != "skipped"]
    passed = [r for r in ran if r.get("status") == "pass"]
    failed = [r for r in all_results if r.get("status") == "fail"]
    warned = [r for r in all_results if r.get("status") == "warning"]
    skipped = [r for r in all_results if r.get("status") == "skipped"]

    def _line(r: dict) -> str:
        return f"{r.get('name', 'Check')}: {r.get('value', '')}"

    # Top-level errors (e.g. corrupt/oversize upload) and critical check failures
    # are fatal; non-critical failures and warnings are advisory.
    response_errors = list(errors or [])
    response_warnings = list(warnings or [])
    for r in failed:
        (response_errors if r["critical"] else response_warnings).append(_line(r))
    response_warnings.extend(_line(r) for r in warned)

    manual_review = []
    for r in skipped:
        if r["critical"]:
            response_warnings.append(f"{r.get('name', 'Check')}: not verified")
        else:
            manual_review.append(_line(r))

    overall = round(len(passed) / len(ran) * 100) if ran else 0
    status = "fail" if response_errors else ("warning" if response_warnings else "pass")

    return {
        "ok": status == "pass",
        "status": status,
        "overall_score": overall,
        "errors": response_errors,
        "warnings": response_warnings,
        "manual_review": manual_review,
        "technical": technical,
        "biometric": biometric,
        "tamper": tamper,
        "checks": all_results,
    }


def _safe_validator(category: str, fn: Callable[[], list[dict]]) -> list[dict]:
    try:
        return fn()
    except Exception as exc:
        logger.exception("%s validator failed", category)
        return [
            json_param(
                f"{category} Validation",
                "Skipped",
                "Validator completed",
                False,
                status="skipped",
                rec=f"{category} validation could not run. Try a valid JPEG; details are in the server logs.",
                fix="Try a valid JPEG and check server logs if this persists.",
            )
        ]


def _parse_overrides(raw_overrides: Optional[str]) -> tuple[dict[str, float] | None, list[str]]:
    if not raw_overrides:
        return None, []
    try:
        parsed = json.loads(raw_overrides)
    except json.JSONDecodeError:
        return None, ["Manual overrides were ignored because they were not valid JSON."]

    if not isinstance(parsed, dict):
        return None, ["Manual overrides were ignored because they were not an object."]

    clean: dict[str, float] = {}
    for key in ("top", "eye", "chin"):
        if key not in parsed:
            continue
        try:
            clean[key] = max(0.0, min(1.0, float(parsed[key])))
        except (TypeError, ValueError):
            return None, [f"Manual override '{key}' was ignored because it was not numeric."]

    return clean or None, []


def _validate_raw(
    raw: bytes, content_type: Optional[str], overrides_raw: Optional[str]
) -> Tuple[int, dict, Optional[Image.Image]]:
    """Run the full validation. Returns (http_status, summary, pil_or_None).

    Shared by /api/validate (JSON) and /api/report (PDF) so both agree exactly.
    """
    if not raw:
        return 400, _summarize_response([], [], [], errors=["Uploaded file is empty."]), None

    if len(raw) > MAX_UPLOAD_BYTES:
        return (
            413,
            _summarize_response(
                [], [], [],
                errors=[f"Uploaded file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB safety limit."],
            ),
            None,
        )

    try:
        probe = Image.open(io.BytesIO(raw))
        detected_format = probe.format
        probe.verify()
        pil = load_image_bytes(raw).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        logger.warning("Rejected invalid image upload: %s", exc)
        return 400, _summarize_response([], [], [], errors=["Uploaded file is not a valid supported image."]), None

    manual_overrides, override_warnings = _parse_overrides(overrides_raw)
    bgr = pil_to_cv(pil)

    # Full validation with all features (PIL + OpenCV + MediaPipe).
    tech_results = _safe_validator(
        "Technical",
        lambda: TechValidator(pil, raw, content_type or "", detected_format).run(),
    )
    bio_results = _safe_validator(
        "Biometric",
        lambda: BioValidator(bgr).run(manual_overrides=manual_overrides),
    )
    tamper_results = _safe_validator(
        "Tamper",
        lambda: TamperValidator(pil).run(),
    )

    summary = _summarize_response(tech_results, bio_results, tamper_results, warnings=override_warnings)
    return 200, summary, pil


@app.get("/healthz")
async def healthz():
    """Lightweight liveness probe — no model init, no side effects."""
    return {
        "status": "ok",
        "version": __version__,
        "python": platform.python_version(),
        "opencv_available": _bio.CV2_AVAILABLE,
        "mediapipe_available": _bio.MP_AVAILABLE,
    }


@app.get("/readyz")
async def readyz():
    """Readiness probe — initializes the CV models once (cached) and reports status.

    Separate from /healthz so liveness checks stay cheap and side-effect-free.
    """
    fd_ready = await run_in_threadpool(lambda: _bio._get_face_detection() is not None)
    fm_ready = await run_in_threadpool(lambda: _bio._get_face_mesh() is not None)
    return {
        "ready": bool(_bio.CV2_AVAILABLE and _bio.MP_AVAILABLE and fd_ready and fm_ready),
        "opencv_available": _bio.CV2_AVAILABLE,
        "mediapipe_available": _bio.MP_AVAILABLE,
        "face_detection_ready": fd_ready,
        "face_mesh_ready": fm_ready,
    }


@app.post("/api/validate")
async def validate(file: UploadFile = File(...), overrides: Optional[str] = Form(None)):
    raw = await file.read()
    # CV work is CPU-bound and synchronous; run it off the event loop so it
    # doesn't block health checks / static serving / other requests.
    status_code, summary, _ = await run_in_threadpool(
        _validate_raw, raw, file.content_type, overrides
    )
    return JSONResponse(status_code=status_code, content=summary)


@app.post("/api/report")
async def report(file: UploadFile = File(...), overrides: Optional[str] = Form(None)):
    raw = await file.read()
    status_code, summary, pil = await run_in_threadpool(
        _validate_raw, raw, file.content_type, overrides
    )
    if status_code != 200 or pil is None:
        return JSONResponse(status_code=status_code, content=summary)
    try:
        pdf = await run_in_threadpool(build_report_pdf, pil, summary)
    except Exception:
        logger.exception("PDF report generation failed")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "status": "fail", "errors": ["Could not generate the PDF report."]},
        )
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="dv-photo-report.pdf"'},
    )

# --- Serve Frontend ---
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")