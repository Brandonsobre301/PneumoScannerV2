from typing import Optional, Literal
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import sys

# Add parent directory to sys.path to import predict.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# predict_image is assumed OOD-aware, falling back to hardcoded thresholds if ood_stats.pth is missing
from predict import load_model, load_gatekeeper_model, predict_image

app = FastAPI(title="Pneumonia Detection API")

# Setup CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
model = load_model('../resnet18_pneumonia.pth')
# gatekeeper_model is None (pre-flight gate skipped) if train_gatekeeper.py hasn't been run yet
gatekeeper_model = load_gatekeeper_model('../gatekeeper_mobilenet.pth')


class OODMetrics(BaseModel):
    # Optional: predict.py omits fields that a given rejection reason didn't compute
    # (e.g. mc_variance when short-circuited earlier, gatekeeper_confidence when the
    # gate passed or isn't loaded).
    cosine_distance: Optional[float] = None
    mc_variance: Optional[float] = None
    gatekeeper_confidence: Optional[float] = None


class PredictionResponse(BaseModel):
    status: Literal["success", "ood"]
    prediction: Optional[Literal["NORMAL", "PNEUMONIA"]] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    ood_metrics: OODMetrics


@app.post("/analyze", response_model=PredictionResponse)
async def analyze_xray(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
         raise HTTPException(status_code=400, detail="File provided is not an image.")

    # Strip directory components to prevent path traversal via the client-supplied filename
    safe_filename = os.path.basename(file.filename)
    temp_path = f"temp_{safe_filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run prediction (pipeline decides success vs. ood internally)
        result = predict_image(temp_path, model, gatekeeper_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # OOD detections still return HTTP 200 so the frontend can render the rejection state
    return PredictionResponse(**result)
