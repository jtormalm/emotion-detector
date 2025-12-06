import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ======= Setup =======
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ======= Same transforms as training =======
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ======= Load test dataset =======
test_data = datasets.ImageFolder('data/test', transform=test_transforms)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ======= Load model =======
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)  # 7 emotion classes
model.load_state_dict(torch.load("models/fer_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# ======= Evaluate =======
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# ======= Metrics =======
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_data.classes, digits=3))

# ======= Confusion Matrix =======
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data.classes)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Facial Emotion Recognition - Confusion Matrix")
plt.show()
