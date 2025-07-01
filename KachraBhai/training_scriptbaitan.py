import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

# Version Info
print('System Version:', sys.version)
print('PyTorch version:', torch.__version__)
# print('Torchvision version:', torchvision.__version__)
print('Numpy version:', np.__version__)
print('Pandas version:', pd.__version__)

# Custom Dataset Wrapper
class wastedata(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

# Transform (resize and tensor conversion)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Data paths
train_folder = "C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/train/dataset-resized"
valid_folder = "C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/valid/dataset-resized"
test_folder = "C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/test/dataset-resized"

# Datasets
train_dataset = wastedata(train_folder, transform=transform)
val_dataset = wastedata(valid_folder, transform=transform)
test_dataset = wastedata(test_folder, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Reverse class mapping (optional)
target_to_class = {v: k for k, v in ImageFolder(train_folder).class_to_idx.items()}
print("Class Index to Label Mapping:", target_to_class)

print("Train classes:", train_dataset.classes)
print("Num classes:", len(train_dataset.classes))
print("Train samples:", len(train_dataset))


# Model Definition
class wasteClassifer(nn.Module):
    def __init__(self, num_classes=3):
        super(wasteClassifer, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Training setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = wasteClassifer(num_classes=len(train_dataset.classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 15
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), r"C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/tp.pt")

# Plot losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Reload and Evaluate
model = wasteClassifer(num_classes=len(train_dataset.classes))
model.load_state_dict(torch.load(r"C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/tp.pt"))
model.to(device)

def check_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f'Accuracy: {acc:.2f}% ({correct}/{total})')
    return acc

# Final Accuracy on Test Data
check_accuracy(test_loader, model, device)
