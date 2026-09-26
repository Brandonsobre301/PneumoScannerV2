"""
Offline OOD calibration script for PneumoScan.

Loads the trained ResNet-18 classifier, runs the training set through it to
extract 512-d embeddings (the avgpool output, pre-fc), computes the
in-distribution centroid, and derives safe thresholds for:
  - Cosine distance (embedding vs. centroid) -> catches inputs unlike any
    training X-ray (e.g. non-chest images).
  - MC Dropout variance (stochastic forward passes) -> catches inputs the
    model is uncertain about.

Output: ood_stats.pth (a dict consumed by predict.py at inference time).

Must stay in sync with train.py / predict.py: image size (256/224) and
normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT   = r"C:\Users\brand\Downloads\workspace(1)\workspace\data\chestxrays"
TRAIN_DIR   = DATA_ROOT + r"\train"
MODEL_PATH  = "resnet18_pneumonia.pth"
SAVE_PATH   = "ood_stats.pth"
BATCH_SIZE  = 32
MC_PASSES   = 40     # number of stochastic forward passes for MC Dropout (higher = less noisy variance estimate)
DROPOUT_P   = 0.5    # dropout applied to the pre-layer4 trunk features for MC sampling (0.3 gave ~0 variance)
PERCENTILE  = 95.0   # percentile of the in-distribution MC-variance used as threshold
COSINE_K_SIGMA = 2.0 # cosine threshold = mean + k*std of in-distribution cosine distances

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_feature_model(model_path: str, device: torch.device) -> ResNet:
    """Load the trained classifier, matching the architecture in train.py/predict.py."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def extract_trunk(model: ResNet, inputs: torch.Tensor) -> torch.Tensor:
    """Run the backbone up through layer3 (features fed into layer4 during MC Dropout)."""
    x = model.conv1(inputs)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    return model.layer3(x)


def extract_embedding(model: ResNet, inputs: torch.Tensor) -> torch.Tensor:
    """Run the backbone up to (and including) avgpool, returning flattened 512-d features."""
    x = extract_trunk(model, inputs)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


@torch.no_grad()
def compute_embeddings(model: ResNet, loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Deterministic (dropout-free) embeddings for every training image."""
    all_embeddings = []
    for inputs, _ in loader:
        inputs = inputs.to(device)
        embeddings = extract_embedding(model, inputs)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def compute_mc_variances(model: ResNet, loader: DataLoader, device: torch.device,
                          mc_passes: int, dropout_p: float) -> torch.Tensor:
    """MC Dropout variance of the predicted probability, per training image.

    Dropout is applied to the pre-layer4 trunk features (not just the fc head), matching
    predict.py, so each pass re-derives higher-level features instead of just perturbing
    the final linear layer -- fc-only dropout produced ~0 variance on confidently-wrong inputs.
    """
    all_variances = []
    for inputs, _ in loader:
        inputs = inputs.to(device)
        trunk = extract_trunk(model, inputs)  # deterministic backbone features up to layer3

        probs = []
        for _ in range(mc_passes):
            dropped = F.dropout(trunk, p=dropout_p, training=True)
            x = model.layer4(dropped)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            logits = model.fc(x)
            probs.append(torch.sigmoid(logits))
        probs = torch.stack(probs, dim=0)  # (mc_passes, batch, 1)

        variance = probs.var(dim=0, unbiased=False).squeeze(1)  # (batch,)
        all_variances.append(variance.cpu())
    return torch.cat(all_variances, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = ImageFolder(TRAIN_DIR, transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train set: {len(train_ds)} images | classes: {train_ds.classes}")

    model = load_feature_model(MODEL_PATH, device)

    print("Extracting embeddings...")
    embeddings = compute_embeddings(model, train_loader, device)  # (N, 512)

    # Centroid of the in-distribution embeddings (L2-normalized before averaging
    # so cosine distance at inference is scale-invariant).
    normalized = F.normalize(embeddings, dim=1)
    centroid = normalized.mean(dim=0)
    centroid = F.normalize(centroid, dim=0)

    cosine_similarities = normalized @ centroid
    cosine_distances = 1.0 - cosine_similarities

    cosine_threshold = float(cosine_distances.mean() + COSINE_K_SIGMA * cosine_distances.std())
    print(f"Cosine distance: mean={cosine_distances.mean():.4f} "
          f"std={cosine_distances.std():.4f} threshold(mean+{COSINE_K_SIGMA}\u03c3)={cosine_threshold:.4f}")

    print(f"Running {MC_PASSES} MC Dropout passes...")
    mc_variances = compute_mc_variances(model, train_loader, device, MC_PASSES, DROPOUT_P)
    mc_variance_threshold = float(np.percentile(mc_variances.numpy(), PERCENTILE))
    print(f"MC variance: mean={mc_variances.mean():.6f} "
          f"std={mc_variances.std():.6f} p{PERCENTILE:.0f}={mc_variance_threshold:.6f}")

    ood_stats = {
        "centroid": centroid,
        "cosine_distance_threshold": cosine_threshold,
        "mc_variance_threshold": mc_variance_threshold,
        "mc_passes": MC_PASSES,
        "dropout_p": DROPOUT_P,
        "percentile": PERCENTILE,
        "embedding_dim": centroid.shape[0],
        "num_train_samples": len(train_ds),
    }

    torch.save(ood_stats, SAVE_PATH)
    print(f"\nSaved OOD stats to: {SAVE_PATH}")
    print({k: v for k, v in ood_stats.items() if k != "centroid"})


if __name__ == "__main__":
    main()
