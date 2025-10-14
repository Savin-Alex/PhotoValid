from __future__ import annotations
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from typing import Optional
import io
import json
import numpy as np
import sys
import os

# Fix Python path for Vercel deployment
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)  # so 'validators' folder is included
sys.path.append(os.path.join(current_dir, "validators"))  # in case needed

from validators.utils import load_image_bytes, pil_to_cv
from validators.tech import TechValidator
from validators.bio import BioValidator
from validators.tamper import TamperValidator

app = FastAPI(title="DV Photo Validator API", version="0.1.0")

# CORS configuration for both local dev and production
allowed_origins = [
    "http://localhost:5173",  # Local frontend dev server
    "http://127.0.0.1:5173",  # Local frontend dev server
    "https://*.vercel.app",    # Vercel deployments
    "https://dvphoto.vercel.app",  # Specific production domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class ValidationResponse(BaseModel):
    status: str
    overall_score: int
    technical: list[dict]
    biometric: list[dict]
    recommendations: list[str]

@app.post("/validate", response_model=ValidationResponse)
async def validate(
    file: UploadFile = File(...),
    overrides: Optional[str] = Form(None)  # ✅ HYBRID: JSON {"top":0.18,"eye":0.58,"chin":0.86}
):
    raw = await file.read()
    pil = load_image_bytes(raw)
    
    # Technical validation
    tech = TechValidator(pil, raw, file.content_type)
    technical = tech.run()

    # ✅ HYBRID: Parse manual overrides if provided
    manual = None
    if overrides:
        try:
            manual = json.loads(overrides)
        except Exception:
            manual = None  # Invalid JSON - ignore and use auto

    # Biometric validation (with optional manual overrides)
    bgr = pil_to_cv(pil)
    bio = BioValidator(bgr)
    biometric = bio.run(manual_overrides=manual)
    
    # Tampering detection
    tamper = TamperValidator(pil).run()

    all_params = technical + biometric + tamper
    passed = sum(1 for p in all_params if p.get('status') == 'pass')
    overall = round(passed/len(all_params)*100) if all_params else 0
    status = 'pass' if overall >= 80 else ('warning' if overall >= 60 else 'fail')
    recs = [p.get('recommendation') for p in all_params if p.get('status')!='pass' and p.get('recommendation')]

    return ValidationResponse(
        status=status,
        overall_score=overall,
        technical=technical,
        biometric=biometric+tamper,
        recommendations=recs + ['Automated tampering detection is an estimate only — not proof of editing.'],
    )

@app.get("/")
async def root():
    return {"message": "DV Validator API up"}

