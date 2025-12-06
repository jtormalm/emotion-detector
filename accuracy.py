import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# --- Device ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Data transforms (must match what you used for training) ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # only if you trained grayscale images
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- Datasets ---
train_dataset = datasets.ImageFolder(root="data/train", transform=transform)
test_dataset  = datasets.ImageFolder(root="data/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- Load model ---
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 7)  # assuming 7 emotion classes
model.load_state_dict(torch.load("models/fer_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# --- Function to evaluate accuracy ---
def evaluate(loader, name="set"):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"{name.capitalize()} Accuracy: {acc:.2f}%")
    return acc

# --- Run evaluation ---
train_acc = evaluate(train_loader, "training")
test_acc = evaluate(test_loader, "testing")
