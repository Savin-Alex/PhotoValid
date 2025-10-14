# Vercel Deployment Guide

## Quick Deploy to Vercel

### 1. Prerequisites
- GitHub repository pushed to GitHub
- Vercel account (free tier works)

### 2. Vercel Project Setup
1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. Configure project:
   - **Framework Preset**: Other
   - **Root Directory**: `/` (project root)
   - **Build Command**: (leave blank)
   - **Output Directory**: (leave blank)

### 3. Deployment Configuration
The `vercel.json` file is already configured:
- Frontend served from `/frontend/` as static files
- API routes (`/api/*`) forwarded to FastAPI backend
- Automatic routing for single-page app

### 4. Environment Variables (if needed)
If you need any environment variables, add them in Vercel dashboard:
- Project Settings → Environment Variables

### 5. Deploy
- Click "Deploy" in Vercel dashboard
- Wait for build to complete
- Your app will be available at `https://your-project.vercel.app`

## Local Testing

To test the deployment configuration locally:

```bash
# Install Vercel CLI
npm i -g vercel

# Test locally
vercel dev
```

## API Endpoints

- `POST /api/validate` - Photo validation endpoint
- All other routes serve the frontend SPA

## File Structure for Vercel

```
/
├── vercel.json          # Vercel configuration
├── backend/
│   ├── main.py         # FastAPI app (entry point)
│   ├── requirements.txt # Python dependencies
│   └── validators/     # Validation modules
└── frontend/
    └── index.html      # Static frontend
```

## Troubleshooting

### MediaPipe/OpenCV Size Issues
If you hit the 250MB serverless function limit:
- Consider using lighter alternatives
- Or switch to a VPS/container deployment

### CORS Issues
The backend is configured with permissive CORS for public API usage.

### Build Failures
Check Vercel build logs for:
- Missing dependencies in requirements.txt
- Import errors
- Python version compatibility
