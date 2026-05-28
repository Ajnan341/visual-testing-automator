import os
import shutil
import base64
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from processing.preprocessor import align_images
from processing.diff_engine import compute_diff
from analysis.gemini_analyzer import analyze_pair
from config import settings

app = FastAPI(title="Visual Testing Agent")

# Mount static directory for JS/CSS if needed
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def encode_image_b64(img_path: str) -> str:
    try:
        return base64.b64encode(Path(img_path).read_bytes()).decode("utf-8")
    except Exception:
        return ""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return (static_dir / "index.html").read_text(encoding="utf-8")

@app.post("/api/analyze")
async def analyze_endpoint(
    figma_image: UploadFile = File(...),
    web_image: UploadFile = File(...)
):
    try:
        figma_path = UPLOAD_DIR / figma_image.filename
        web_path = UPLOAD_DIR / web_image.filename
        
        with open(figma_path, "wb") as buffer:
            shutil.copyfileobj(figma_image.file, buffer)
        with open(web_path, "wb") as buffer:
            shutil.copyfileobj(web_image.file, buffer)
            
        # Run Pipeline
        figma_img, web_img = align_images(str(figma_path), str(web_path))
        diff = compute_diff(figma_img, web_img)
        
        heatmap_path = UPLOAD_DIR / "heatmap.png"
        diff["heatmap"].save(heatmap_path)
        
        # Run AI Analysis
        ai_result = analyze_pair(str(figma_path), str(web_path), diff["ssim_score"], settings.model)
        
        return {
            "ssim_score": diff["ssim_score"],
            "heatmap_b64": encode_image_b64(str(heatmap_path)),
            "markdown_report": ai_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        figma_image.file.close()
        web_image.file.close()
