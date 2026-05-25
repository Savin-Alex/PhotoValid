# DV Photo Validator – Production Starter

## 🎯 Overview

AI-powered photo validation system for U.S. Department of State Diversity Visa (DV) Lottery program. This production-ready web service validates photos against 25+ technical and biometric requirements specified by the DV program.

## ✨ Features

### Technical Validations
- ✅ Photo dimensions (600×600 to 1200×1200 px, square)
- ✅ File size (≤ 240 KB)
- ✅ Compression ratio (≤ 20:1)
- ✅ File format (JPEG only)
- ✅ Color model (24-bit color, sRGB; no grayscale)
- ✅ Brightness & contrast analysis
- ✅ EXIF date validation (≤ 6 months old)
- ✅ Image noise (flat-region noise floor)

### Biometric & Composition Validations
- ✅ Face detection (exactly one face)
- ✅ Head height (50–69% of image height)
- ✅ Eye level positioning (56–69% from bottom)
- ✅ Head centering (horizontal offset)
- ✅ Background uniformity (plain white/off-white, sampled at the top corners)
- ✅ Sharpness (face region, Laplacian variance)
- ✅ Face lighting balance (left vs right)
- ✅ Eyes open (Eye Aspect Ratio)
- ✅ Gaze / looking at camera (iris centering)
- ✅ Neutral expression (mouth openness)
- ✅ Head tilt (inter-eye line angle)
- ⚠️ Glasses — best-effort heuristic; warns on likely eyeglasses (prohibited), never confidently clears
- ⚠️ Red-eye — best-effort heuristic; warns on likely red-eye, never confidently clears
- ⚠️ Headgear/hats — manual review (not auto-detected)

### Output
- Overall compliance score (0–100%)
- Status for each parameter (✅ Pass / ❌ Fail / ⚠️ Warning)
- Measured vs expected values
- Actionable recommendations for failed parameters
- Visual overlay showing eye level and head position guidelines

## 🏗️ Project Structure

```
PhotoValid/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── validators/
│   │   ├── __init__.py
│   │   ├── utils.py            # Helper functions
│   │   ├── tech.py             # Technical validations
│   │   └── bio.py              # Biometric validations
│   └── requirements.txt
├── frontend/
│   └── index.html              # Single-page web interface
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.11 or 3.12** (MediaPipe supports both; Render deploys on 3.11.9 per `runtime.txt`)
- pip package manager
- (Optional) Homebrew on macOS for easy Python installation

### Backend Setup

#### Installing Python 3.12 (if needed)

**macOS (Homebrew):**
```bash
brew install python@3.12
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/)

**Linux:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

#### Setting Up the Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate a virtual environment with Python 3.12:
```bash
# macOS/Linux (if installed via Homebrew):
python3.12 -m venv .venv
source .venv/bin/activate

# Or use the full path:
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate

# Windows:
python3.12 -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- FastAPI & Uvicorn (web framework)
- Pillow (image processing)
- OpenCV (computer vision)
- **MediaPipe** (face detection & landmarks)
- NumPy, piexif, and other utilities

4. Start the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd ../frontend
```

2. Start a simple HTTP server:
```bash
# Python 3
python -m http.server 5173

# Or use any other static file server
# Node.js: npx serve -p 5173
```

3. Open your browser and navigate to:
```
http://localhost:5173
```

## 📡 API Endpoints

### `POST /api/validate`

Validates an uploaded photo against DV Lottery requirements.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` field containing a JPEG image (optional `overrides` field: JSON string with `top`/`eye`/`chin` normalized 0–1 line positions)

**Response:**
```json
{
  "ok": false,
  "status": "pass|warning|fail",
  "overall_score": 85,
  "errors": [],
  "warnings": [],
  "manual_review": ["Red-Eye: Not auto-checked"],
  "technical": [
    {
      "name": "Photo Dimensions",
      "value": "600×600px",
      "expected": "600×600–1200×1200 px",
      "status": "pass"
    }
  ],
  "biometric": [
    {
      "name": "Head Height",
      "value": "61.2%",
      "expected": "50–69% of image height",
      "status": "pass",
      "head_height_pct": 61.2
    }
  ],
  "tamper": [],
  "checks": []
}
```

### `GET /`

Serves the static frontend (single-page web interface) and returns `200` — use it as
the liveness probe. There is no separate JSON health endpoint; `GET /api/validate`
returns `404` because the catch-all static mount at `/` handles non-POST paths.

## 🔧 Technology Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **Pillow** - Python Imaging Library for image processing
- **OpenCV** - Computer vision library for image analysis
- **MediaPipe** - Google's ML solution for face detection & landmarks
- **NumPy** - Numerical computing for array operations
- **piexif** - EXIF metadata extraction

### Frontend
- Pure HTML5, CSS3, and vanilla JavaScript
- Canvas API for visual overlays
- Responsive design (mobile-friendly)

## 🎨 Using the Web Interface

1. **Upload Photo**: Click the drop zone or drag & drop a JPEG file
2. **Validate**: Click the "Validate" button
3. **Review Results**: 
   - Overall score and status at the top
   - Detailed technical parameters
   - Biometric & composition analysis
   - Visual overlay showing alignment guides
4. **Fix Issues**: Follow recommendations for failed parameters
5. **Re-upload**: Submit corrected photo for re-validation

## 🔍 Validation Logic

### Scoring
- **Overall score** = the percentage of *auto-run* checks that pass. Checks that
  cannot be auto-verified (`skipped` / manual review) are excluded from the score.
- **Status**:
  - `fail` — a **critical** check failed (dimensions, file size, format, color model,
    compression ratio, face detected, one person, head height, eye level) or the upload
    was invalid.
  - `warning` — only **advisory** checks failed or warned (e.g. brightness, sharpness,
    centering, lighting, expression, gaze, head tilt, likely glasses).
  - `pass` — no failures or warnings (a clean photo can still pass even though some
    items, e.g. red-eye/headgear, are left for manual review).
- Manual-review items are returned in a separate `manual_review` list so they don't
  silently pass *or* block a clean result.

### Critical Failures
The following parameters cause immediate rejection:
- Non-square dimensions or out of range (600–1200 px)
- File size > 240 KB
- Non-JPEG format
- No face detected or multiple faces
- Head height outside 50–69% range
- Eyes outside 56–69% range from bottom

### Minor Failures
The following parameters trigger warnings but allow re-upload:
- Brightness slightly off (30–40% or 60–70%)
- Fair contrast (40–60)
- Slight sharpness issues
- Missing EXIF date

## 🚀 Production Deployment

### Docker (Recommended)

Create a `Dockerfile` in the backend directory:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t dv-validator .
docker run -p 8000:8000 dv-validator
```

### Cloud Deployment Options
- **AWS**: Elastic Beanstalk, ECS, or Lambda
- **Google Cloud**: Cloud Run, App Engine
- **Azure**: App Service, Container Instances
- **Heroku**: Easy deployment with buildpacks
- **DigitalOcean**: App Platform

### Environment Variables
```bash
export HOST=0.0.0.0
export PORT=8000
export CORS_ORIGINS="https://yourdomain.com"
```

## 🧪 Testing

### Manual Testing
1. Prepare test images meeting DV requirements
2. Upload through web interface
3. Verify all parameters pass

### API Testing with curl
```bash
curl -X POST "http://localhost:8000/validate" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_photo.jpg"
```

## 📝 Notes

- **MediaPipe Face Detection & Face Mesh** provide precise head and eye positioning
- Background uniformity is checked on the border band (10px)
- Some parameters (glasses, red-eye, expression) use heuristic methods and can be enhanced with specialized models
- EXIF date validation requires the photo to have capture date metadata

## 🔮 Future Enhancements

Potential improvements for production:
- ✨ Deep learning models for glasses/headgear detection
- ✨ Advanced expression analysis (smile detection)
- ✨ Red-eye correction algorithm
- ✨ Compression artifact detection (DCT analysis)
- ✨ Skin tone manipulation detection
- ✨ Batch processing API
- ✨ User authentication & history
- ✨ PDF report generation
- ✨ Multi-language support
- ✨ Payment integration for premium features

## 📄 License

This project is provided as-is for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## ⚠️ Disclaimer

This tool provides automated analysis based on publicly available DV Lottery photo requirements. Always verify with official U.S. Department of State guidelines before submission. This tool does not guarantee acceptance of your DV Lottery application.

## 📞 Support

For questions or issues, please open an issue on the repository.

---

**Built with ❤️ for DV Lottery applicants worldwide**

