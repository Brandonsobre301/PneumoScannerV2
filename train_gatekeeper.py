"""
Pre-flight gatekeeper: X-ray vs non-X-ray binary classifier for PneumoScan.

Trains a MobileNetV3-Small (ImageNet-pretrained, most layers frozen) to distinguish real
chest X-rays (positive class) from arbitrary photos (negative class, sampled from
FashionMNIST) so obviously-wrong inputs (product photos, random pictures) get rejected
before ever reaching the pneumonia classifier or the cosine/MC-dropout OOD checks.

Output: gatekeeper_mobilenet.pth (state_dict, consumed by predict.py at inference time).

Must stay in sync with train.py / predict.py: image size (256/224) and normalization
(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder, FashionMNIST

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT     = r"C:\Users\brand\Downloads\workspace(1)\workspace\data\chestxrays"
TRAIN_DIR     = DATA_ROOT + r"\train"
NEGATIVE_ROOT = "data/fashion_mnist"  # CIFAR-10's host has an expired cert; FashionMNIST is a reachable stand-in
SAVE_PATH     = "gatekeeper_mobilenet.pth"
NUM_EPOCHS    = 5
BATCH_SIZE    = 16
LR            = 1e-3
NEGATIVE_COUNT = 300  # negative-class images sampled as the "non-X-ray" class

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert("RGB")),  # FashionMNIST images are single-channel
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class BinaryLabelDataset(Dataset):
    """Wraps a dataset, discarding its original label in favor of a fixed binary one."""

    def __init__(self, base_dataset, label: float):
        self.base_dataset = base_dataset
        self.label = label

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        return image, self.label


def build_gatekeeper_model() -> nn.Module:
    """MobileNetV3-Small with a fresh single-logit head; only the last block + head train."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[-1].parameters():
        param.requires_grad = True

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 1)  # binary logit: 1 = chest X-ray
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    xray_ds = ImageFolder(TRAIN_DIR, transform=transform)
    xray_ds = BinaryLabelDataset(xray_ds, label=1.0)
    print(f"X-ray (positive) images: {len(xray_ds)}")

    negative_ds = FashionMNIST(NEGATIVE_ROOT, train=True, download=True, transform=transform)
    negative_subset = torch.utils.data.Subset(negative_ds, list(range(NEGATIVE_COUNT)))
    negative_subset = BinaryLabelDataset(negative_subset, label=0.0)
    print(f"Non-X-ray (negative) images: {len(negative_subset)}")

    full_ds = ConcatDataset([xray_ds, negative_subset])
    val_size = max(1, int(0.15 * len(full_ds)))
    train_ds, val_ds = random_split(
        full_ds, [len(full_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(101010),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_gatekeeper_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_loss += loss.item() * inputs.size(0)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                preds = (torch.sigmoid(model(inputs)) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch}/{NUM_EPOCHS}]  train_loss: {train_loss:.4f}  "
              f"train_acc: {train_acc:.4f}  val_acc: {val_acc:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nGatekeeper model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
