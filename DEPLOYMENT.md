# Render Deployment Guide - PhotoValid

## 🚀 One-Click Deployment to Render

### Repository Structure
```
PhotoValid/
├── render.yaml              # Render configuration
├── backend/
│   ├── main.py             # FastAPI app with static serving
│   ├── requirements.txt    # Python dependencies
│   └── validators/         # Validation modules
└── frontend/
    └── index.html          # Static frontend
```

### Deployment Steps

1. **Push to GitHub** ✅ (All changes committed)

2. **Go to Render**: https://render.com
   - Sign up/Login with GitHub
   - Click "New +" → "Web Service"

3. **Connect Repository**:
   - Select `Savin-Alex/PhotoValid`
   - Render auto-detects `render.yaml`

4. **Deploy**:
   - Click "Create Web Service"
   - Wait for build (5-10 minutes)

### Configuration (render.yaml)
```yaml
services:
  - type: web
    name: photo-valid
    env: python
    plan: free
    region: frankfurt
    runtime: python-3.11
    buildCommand: pip install -r backend/requirements.txt
    startCommand: uvicorn backend.main:app --host 0.0.0.0 --port 10000
    autoDeploy: true
    envVars:
      - key: PORT
        value: 10000
```

### Python Version Fix
- **Issue**: MediaPipe doesn't support Python 3.13 yet
- **Solution**: Using Python 3.11 with MediaPipe 0.10.14 (compatible version)
- **Result**: All dependencies install successfully

### Post-Deployment URLs
- **Frontend**: `https://photo-valid.onrender.com`
- **API**: `https://photo-valid.onrender.com/api/validate`
- **Health Check**: `https://photo-valid.onrender.com/api/validate` (GET)

### Features Enabled
- ✅ **Full MediaPipe face detection** (no size limits!)
- ✅ **OpenCV image processing**
- ✅ **Complete biometric validation**
- ✅ **Tamper detection**
- ✅ **Technical validations**
- ✅ **Static frontend serving**

### Testing
1. Visit homepage → UI loads
2. Upload photo → hits `/api/validate`
3. Manual API test:
   ```bash
   curl https://photo-valid.onrender.com/api/validate
   # Should return: {"detail":"Method Not Allowed"}
   ```

### Optional: Domain Lockdown
After deployment works, update CORS in `backend/main.py`:
```python
allow_origins=["https://photo-valid.onrender.com"]
```

### Performance Notes
- **Free Tier**: 95% uptime, sleeps after 15min idle
- **Paid Tier**: $7/month for always-on, better performance
- **Build Time**: ~5-10 minutes (MediaPipe + OpenCV)

---

**Your DV Photo Validator is ready for production deployment!** 🎉