import os
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import transforms
from PIL import Image

# ── OOD / MC-dropout config ──────────────────────────────────────────────────
# Produced offline by compute_embeddings_stats.py (see contract.json / DAG.txt);
# this file does not exist until that script is run, so we degrade gracefully.
OOD_STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ood_stats.pth")
DEFAULT_COSINE_THRESHOLD = 0.3  # cosine distance above this = OOD (overridable via stats file)
N_PASSES = 40                    # MC-dropout forward passes; higher = less noisy variance estimate near threshold
MC_DROPOUT_P = 0.5               # empirically, 0.3 produced ~0 variance even for confidently-wrong OOD inputs

# ── Gatekeeper (X-ray vs non-X-ray) config ───────────────────────────────────
# Produced offline by train_gatekeeper.py; runs before the cosine/MC-dropout checks so
# obviously-wrong inputs (product photos, random pictures) short-circuit immediately.
GATEKEEPER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gatekeeper_mobilenet.pth")
DEFAULT_GATEKEEPER_THRESHOLD = 0.5

# Same preprocessing used in training
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD)
])

_ood_stats_cache = None


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    # torchvision's Compose stub returns the input type generically; ToTensor makes this a real Tensor at runtime.
    return cast(torch.Tensor, _transform(image)).unsqueeze(0)  # Add batch dimension


def load_model(model_path='resnet18_pneumonia.pth'):
    # Initialize ResNet18
    model = models.resnet18(weights=None)
    # Modify final layer as done in training
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def load_gatekeeper_model(model_path=GATEKEEPER_PATH):
    """Load the X-ray vs non-X-ray gatekeeper, or None if train_gatekeeper.py hasn't been run yet."""
    if not os.path.exists(model_path):
        print(f"[predict.py] WARNING: gatekeeper model not found at '{model_path}'. "
              "Run train_gatekeeper.py to enable the pre-flight gate; skipping it for now.")
        return None

    model = models.mobilenet_v3_small(weights=None)
    # Sequential.__getitem__ is typed generically as Module; cast narrows it for the type checker.
    in_features = cast(nn.Linear, model.classifier[3]).in_features
    model.classifier[3] = nn.Linear(in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def is_chest_xray(input_tensor, gatekeeper_model, threshold=DEFAULT_GATEKEEPER_THRESHOLD):
    """Return (is_xray, confidence) from the pre-flight gatekeeper classifier."""
    with torch.no_grad():
        logit = gatekeeper_model(input_tensor)
        confidence = torch.sigmoid(logit).item()
    return confidence > threshold, round(confidence, 4)


def load_ood_stats(stats_path=OOD_STATS_PATH):
    """Load the in-distribution embedding centroid + thresholds from compute_embeddings_stats.py output."""
    global _ood_stats_cache
    if _ood_stats_cache is not None:
        return _ood_stats_cache

    if not os.path.exists(stats_path):
        print(f"[predict.py] WARNING: OOD stats file not found at '{stats_path}'. "
              "Run compute_embeddings_stats.py to enable OOD rejection; skipping OOD check for now.")
        _ood_stats_cache = {
            "centroid": None,
            "cosine_distance_threshold": DEFAULT_COSINE_THRESHOLD,
            "mc_variance_threshold": None,
            "mc_passes": N_PASSES,
            "dropout_p": MC_DROPOUT_P,
        }
        return _ood_stats_cache

    raw = torch.load(stats_path, map_location=torch.device("cpu"))

    _ood_stats_cache = {
        "centroid": raw["centroid"],
        "cosine_distance_threshold": float(raw.get("cosine_distance_threshold", DEFAULT_COSINE_THRESHOLD)),
        "mc_variance_threshold": raw.get("mc_variance_threshold"),
        "mc_passes": int(raw.get("mc_passes", N_PASSES)),
        "dropout_p": float(raw.get("dropout_p", MC_DROPOUT_P)),
    }
    return _ood_stats_cache


def _forward_trunk(model, input_tensor):
    """Run the backbone up through layer3 (deterministic, eval mode)."""
    x = model.conv1(input_tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    return model.layer3(x)


def _forward_head(model, trunk_features):
    """Run layer4 -> avgpool -> flatten, producing the 512-d penultimate embedding."""
    x = model.layer4(trunk_features)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def get_embedding(model, input_tensor):
    """Deterministic 512-d penultimate (post-avgpool) feature vector for the cosine OOD check."""
    model.eval()
    with torch.no_grad():
        trunk_features = _forward_trunk(model, input_tensor)
        embedding = _forward_head(model, trunk_features)
    return embedding


def cosine_ood_check(embedding, centroid, threshold=DEFAULT_COSINE_THRESHOLD):
    """Return (is_ood, cosine_distance) comparing the embedding to the in-distribution centroid."""
    if centroid is None:
        return False, None  # no stats available yet; can't gate, so let it through

    similarity = F.cosine_similarity(embedding, centroid.unsqueeze(0)).item()
    distance = 1 - similarity
    return distance > threshold, round(distance, 4)


def mc_dropout_predict(model, trunk_features, n_passes=N_PASSES, dropout_p=MC_DROPOUT_P):
    """Monte Carlo dropout over the pre-layer4 trunk features -> (mean_probability, variance).

    Dropout is injected before layer4 (not just at the fc head) so each pass re-derives
    higher-level features stochastically; fc-only dropout let confidently-wrong inputs
    (e.g. high-contrast non-X-ray images) sail through with ~0 variance. The conv1-layer3
    trunk runs once in eval mode beforehand, so BatchNorm2d never leaves eval mode.
    """
    model.eval()
    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            dropped = F.dropout(trunk_features, p=dropout_p, training=True)
            embedding = _forward_head(model, dropped)
            logit = model.fc(embedding)
            probs.append(torch.sigmoid(logit).item())

    probs_t = torch.tensor(probs)
    return probs_t.mean().item(), probs_t.var(unbiased=False).item()


def predict_image(image_path, model, gatekeeper_model=None):
    input_tensor = preprocess_image(image_path)

    if gatekeeper_model is not None:
        is_xray, gatekeeper_confidence = is_chest_xray(input_tensor, gatekeeper_model)
        if not is_xray:
            return {
                "status": "ood",
                "prediction": None,
                "confidence": None,
                "message": "Invalid image: this does not appear to be a chest X-ray. Please upload a valid chest radiograph.",
                "ood_metrics": {"cosine_distance": None, "mc_variance": None, "gatekeeper_confidence": gatekeeper_confidence},
            }

    model.eval()
    with torch.no_grad():
        trunk_features = _forward_trunk(model, input_tensor)
        embedding = _forward_head(model, trunk_features)

    stats = load_ood_stats()
    is_ood, cosine_distance = cosine_ood_check(embedding, stats["centroid"], stats["cosine_distance_threshold"])

    if is_ood:
        # Short-circuit here: skip the MC-dropout passes entirely to save compute on rejected inputs.
        return {
            "status": "ood",
            "prediction": None,
            "confidence": None,
            "message": "Uncertain / Out-of-Distribution input. Please upload a valid chest radiograph.",
            "ood_metrics": {"cosine_distance": cosine_distance, "mc_variance": None, "gatekeeper_confidence": None},
        }

    mean_probability, mc_variance = mc_dropout_predict(
        model, trunk_features, n_passes=stats["mc_passes"], dropout_p=stats["dropout_p"]
    )

    mc_variance_threshold = stats["mc_variance_threshold"]
    if mc_variance_threshold is not None and mc_variance > mc_variance_threshold:
        return {
            "status": "ood",
            "prediction": None,
            "confidence": None,
            "message": "Uncertain / Out-of-Distribution input. Please upload a valid chest radiograph.",
            "ood_metrics": {"cosine_distance": cosine_distance, "mc_variance": round(mc_variance, 4), "gatekeeper_confidence": None},
        }

    prediction = "PNEUMONIA" if mean_probability > 0.5 else "NORMAL"
    confidence = mean_probability if prediction == "PNEUMONIA" else 1 - mean_probability

    return {
        "status": "success",
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "ood_metrics": {
            "cosine_distance": cosine_distance,
            "mc_variance": round(mc_variance, 4),
            "gatekeeper_confidence": None,
        },
    }

if __name__ == "__main__":
    # Test script locally with a dummy path if desired
    # print(predict_image("path_to_sample_xray.jpg", load_model()))
    pass
