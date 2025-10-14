# Render Deployment Guide for PhotoValid

## 🚀 Complete Deployment to Render

### Why Render?
- ✅ **No size limits** (handles MediaPipe + OpenCV easily)
- ✅ **Full Python support** with all dependencies
- ✅ **Automatic HTTPS** and custom domains
- ✅ **Free tier available** with reasonable limits
- ✅ **Simple deployment** from GitHub

## 📋 Step-by-Step Deployment

### Step 1: Prepare Repository
✅ **Already Done**:
- `backend/main.py` - Updated with static file serving
- `requirements.txt` - All dependencies included
- `frontend/index.html` - API calls configured for same-origin
- CORS simplified for Render deployment

### Step 2: Deploy to Render

1. **Go to Render**: https://render.com
2. **Sign up/Login** with GitHub
3. **Click "New +"** → **"Web Service"**
4. **Connect GitHub** → Select `Savin-Alex/PhotoValid`

### Step 3: Configure Web Service

| Setting | Value |
|---------|-------|
| **Name** | `photo-valid` |
| **Environment** | `Python 3` |
| **Region** | Choose nearest (Frankfurt/Oregon) |
| **Branch** | `main` |
| **Root Directory** | `/` (repo root) |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |

### Step 4: Environment Variables (Optional)
No environment variables needed for basic deployment.

### Step 5: Deploy
Click **"Create Web Service"** and wait for deployment.

## 🌐 Post-Deployment

### Your App Will Be Available At:
- **Frontend**: `https://photo-valid.onrender.com`
- **API**: `https://photo-valid.onrender.com/validate`
- **Health Check**: `https://photo-valid.onrender.com/api`

### Test Endpoints:
```bash
# Test API health
curl https://photo-valid.onrender.com/api

# Test validation endpoint (should return method not allowed for GET)
curl https://photo-valid.onrender.com/validate
```

## 🔧 Technical Details

### How It Works:
1. **Render** starts the FastAPI backend
2. **FastAPI** serves static files from `/frontend/`
3. **API endpoints** available at `/validate`
4. **Frontend** calls same-origin API (no CORS issues)
5. **Full features** available (MediaPipe + OpenCV)

### File Structure:
```
/
├── requirements.txt     # Python dependencies
├── backend/
│   ├── main.py         # FastAPI app + static serving
│   └── validators/     # All validation modules
└── frontend/
    └── index.html      # Static frontend
```

## 📊 Render Free Tier Limits

| Resource | Free Tier | Paid Tier |
|----------|-----------|-----------|
| **Build Time** | 500 min/month | Unlimited |
| **Uptime** | 95% | 99.95% |
| **Sleep** | After 15 min idle | Always on |
| **Bandwidth** | 100 GB/month | 1 TB/month |
| **Memory** | 512 MB | Up to 32 GB |

## 🚀 Performance Optimization

### For Production:
1. **Upgrade to Paid Plan** ($7/month) for:
   - Always-on service (no sleep)
   - Better performance
   - Higher limits

2. **Custom Domain** (Optional):
   - Settings → Custom Domain
   - Add your domain (e.g., `dvphoto.yourdomain.com`)
   - Automatic HTTPS included

## 🔍 Monitoring & Logs

### View Logs:
- Go to your service dashboard
- Click **"Logs"** tab
- Filter by `uvicorn` for backend logs

### Health Monitoring:
- Render provides built-in health checks
- Monitor uptime and performance
- Set up alerts if needed

## 🆘 Troubleshooting

### Common Issues:

1. **Build Fails**:
   - Check `requirements.txt` syntax
   - Ensure all dependencies are listed
   - Check Render build logs

2. **App Won't Start**:
   - Verify start command is correct
   - Check that `backend.main:app` exists
   - Review startup logs

3. **Static Files Not Loading**:
   - Ensure `frontend/` directory exists
   - Check file permissions
   - Verify static mount path

4. **API Not Working**:
   - Test `/api` endpoint first
   - Check CORS configuration
   - Verify endpoint paths

## 🎯 Success Indicators

After successful deployment, you should see:
- ✅ Frontend loads at your Render URL
- ✅ API responds at `/api` endpoint
- ✅ Photo validation works with full features
- ✅ All MediaPipe/OpenCV features functional
- ✅ No size limit errors

## 📈 Next Steps

1. **Test thoroughly** with various photos
2. **Monitor performance** and usage
3. **Consider upgrading** to paid plan for production
4. **Set up custom domain** if desired
5. **Add monitoring/analytics** as needed

---

**Your DV Photo Validator is now live with full features on Render!** 🎉
