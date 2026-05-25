from __future__ import annotations
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from typing import Optional, Callable
import io
import logging
import os
import sys
import json

# --- Ensure internal imports work ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from validators.utils import load_image_bytes, pil_to_cv
from validators.utils import json_param
from validators.tech import TechValidator
from validators.bio import BioValidator
from validators.tamper import TamperValidator

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
}


def _summarize_response(
    technical: list[dict],
    biometric: list[dict],
    tamper: list[dict],
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict:
    all_results = technical + biometric + tamper

    # "skipped" = could not be auto-verified (manual review). These do NOT count
    # toward the score and do NOT block a "pass" — they are surfaced separately.
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
        if r.get("name") in CRITICAL_CHECKS:
            response_errors.append(_line(r))
        else:
            response_warnings.append(_line(r))
    response_warnings.extend(_line(r) for r in warned)
    manual_review = [_line(r) for r in skipped]

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
                rec=f"{category} validation could not run: {exc}",
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


@app.post("/api/validate")
async def validate(file: UploadFile = File(...), overrides: Optional[str] = Form(None)):
    raw = await file.read()
    warnings: list[str] = []

    if not raw:
        body = _summarize_response([], [], [], errors=["Uploaded file is empty."])
        return JSONResponse(status_code=400, content=body)

    if len(raw) > MAX_UPLOAD_BYTES:
        body = _summarize_response(
            [],
            [],
            [],
            errors=[f"Uploaded file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB safety limit."],
        )
        return JSONResponse(status_code=413, content=body)

    try:
        probe = Image.open(io.BytesIO(raw))
        detected_format = probe.format
        probe.verify()
        pil = load_image_bytes(raw).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        logger.warning("Rejected invalid image upload: %s", exc)
        body = _summarize_response([], [], [], errors=["Uploaded file is not a valid supported image."])
        return JSONResponse(status_code=400, content=body)

    manual_overrides, override_warnings = _parse_overrides(overrides)
    warnings.extend(override_warnings)
    bgr = pil_to_cv(pil)

    # Full validation with all features (PIL + OpenCV + MediaPipe).
    tech_results = _safe_validator(
        "Technical",
        lambda: TechValidator(pil, raw, file.content_type or "", detected_format).run(),
    )
    bio_results = _safe_validator(
        "Biometric",
        lambda: BioValidator(bgr).run(manual_overrides=manual_overrides),
    )
    tamper_results = _safe_validator(
        "Tamper",
        lambda: TamperValidator(pil).run(),
    )

    return _summarize_response(tech_results, bio_results, tamper_results, warnings=warnings)

# --- Serve Frontend ---
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")