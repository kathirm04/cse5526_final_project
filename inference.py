import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import pandas as pd

LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


# model
class ResNet50_MLP(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V2")
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# same image transform as training
img_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img_transform(img).unsqueeze(0)


# run the test inference
def run_batch_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load model
    model = ResNet50_MLP(num_classes=14).to(device)
    model.load_state_dict(torch.load("models/res_net_50_mlp.pth", map_location=device))
    model.eval()
    print("Loaded best_model.pth")

    # Folder of test examples
    test_folder = "test_examples"
    image_files = [
        f
        for f in os.listdir(test_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(image_files) == 0:
        print("No images found in test_examples/")
        return

    results = []

    for filename in image_files:
        path = os.path.join(test_folder, filename)
        img_tensor = load_image(path).to(device)

        # Inference
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Store results
        result = {"image": filename}
        for label, p in zip(LABELS, probs):
            result[label] = float(p)

        results.append(result)

        print(f"Processed: {filename}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("predictions.csv", index=False)
    print("\nPredictions saved to predictions.csv")


if __name__ == "__main__":
    run_batch_inference()
