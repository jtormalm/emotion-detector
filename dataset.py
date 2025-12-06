from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_tf = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_tf = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.ImageFolder(root='data/train', transform=train_tf)
test_data = datasets.ImageFolder(root='data/test', transform=test_tf)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print(train_data.classes)

import matplotlib.pyplot as plt
import torchvision

images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images[:16], nrow=4, normalize=True)
plt.figure(figsize=(6,6))
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.show()
