# PneumoScan: AI Pneumonia Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

A full-stack AI web application that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** in real time using a fine-tuned ResNet-18 model.

---

## Objective

Automate pneumonia detection from chest X-rays using transfer learning, then surface predictions through a production-style REST API and a React front-end — demonstrating the full lifecycle from model training to interactive deployment.

---

## Architecture

| Layer | Technology | Role |
|---|---|---|
| Model | ResNet-18 (ImageNet pretrained) | Binary classification via transfer learning |
| Inference | PyTorch + `predict.py` | Loads weights, applies transforms, returns prediction + confidence |
| API | FastAPI + Uvicorn | `POST /predict` endpoint — accepts image upload, returns JSON |
| Frontend | React + TypeScript + Vite | Dark-themed drag-and-drop UI, scanning animation, live results |

**Why ResNet-18?** Fast to fine-tune, lightweight at inference time, and strong enough for binary medical image classification with limited data.

---

## Results

Trained for 10 epochs on the Kermany et al. chest X-ray dataset (100 train / 200 val / 100 test images).

| Metric | Value |
|---|---|
| Best Validation Accuracy | **90.00%** (Epoch 3) |
| Test Accuracy | **76.00%** |
| Test F1-Score | **0.7647** |

> **Note:** Results reflect training on a small 100-image subset. Fine-tuning on the full ~5,800-image dataset will significantly improve test accuracy.

---

## Dataset

Kermany, D.S. et al. — [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) via Kaggle.  
Folder structure expected:

```
data/chestxrays/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## Project Structure

```
pneumoscan/
├── train.py              # Training script — produces resnet18_pneumonia.pth
├── predict.py            # Inference helper used by the API
├── notebook.ipynb        # Exploratory training notebook
├── requirements.txt      # Python dependencies
├── backend/
│   └── main.py           # FastAPI app with POST /predict endpoint
└── frontend/
    ├── src/
    │   ├── App.tsx        # Main UI component
    │   ├── App.css        # Dark theme styles
    │   └── main.tsx
    └── package.json
```

---

## Getting Started

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Point `DATA_ROOT` in `train.py` to your dataset, then run:

```bash
python train.py
```

This saves `resnet18_pneumonia.pth` to the project root (excluded from git — generate locally).

### 3. Start the API

```bash
cd backend
python -m uvicorn main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
# App available at http://localhost:5173
```

---

## API Reference

### `POST /predict`

Accepts a multipart image upload and returns a prediction.

**Request**
```
Content-Type: multipart/form-data
Body: file=<image>
```

**Response**
```json
{
  "prediction": "Pneumonia",
  "confidence": 0.94
}
```

---

## License

This project is licensed under the MIT License.
