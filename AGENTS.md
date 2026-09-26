# PneumoScan Agent Guidelines

Full architecture, dataset, API reference, and setup steps: [README.md](README.md).

## Build and Test

No automated test suite exists — validate changes by running the app manually:

```bash
pip install -r requirements.txt
python train.py                              # requires local dataset, produces resnet18_pneumonia.pth
cd backend && python -m uvicorn main:app --reload   # API on :8000
cd frontend && npm install && npm run dev           # UI on :5173
cd frontend && npm run lint                         # only available check
```

`train.py` hardcodes `DATA_ROOT` to a local path — update it before running rather than assuming the dataset ships with the repo.

## Critical Convention: Keep Preprocessing in Sync

The image size (256/224), normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), and 0.5 sigmoid threshold are duplicated across [train.py](train.py), [predict.py](predict.py), and used indirectly by [backend/main.py](backend/main.py). If you change one, update all three or inference will silently degrade.

Model path is also duplicated: `train.py` saves to `resnet18_pneumonia.pth` (repo root), `backend/main.py` loads `../resnet18_pneumonia.pth` (relative to `backend/`). Keep these consistent when restructuring folders.

## In-Progress Multi-Agent Contract (OOD Detection)

[DAG.txt](DAG.txt) and [contract.json](contract.json) describe a planned out-of-distribution (OOD) detection feature split across three parallel workstreams: `predict.py` (ML), a not-yet-created `compute_stats.py` (offline OOD stats), and `frontend/src/App.tsx` (UI). `contract.json` is the shared response schema (`status`, `prediction`, `confidence`, `ood_metrics.cosine_distance`, `ood_metrics.mc_variance`) all three sides must agree on.

**Current state does not match the contract yet**: `predict_image()` in `predict.py` only returns `{"prediction", "confidence"}`, and `backend/main.py`'s `/predict` endpoint passes that through unchanged. Don't assume `ood_metrics` or `status` fields exist in the live API — check `contract.json` for the target shape when implementing this feature, and update all three sides (ML, offline script, frontend) together.

## Frontend Notes

The frontend calls the API at a hardcoded `http://localhost:8000/predict` (in [App.tsx](frontend/src/App.tsx)) — no env-based config exists. CORS in `backend/main.py` allows all origins (`*`), which is fine for local dev but should be tightened before any real deployment.
