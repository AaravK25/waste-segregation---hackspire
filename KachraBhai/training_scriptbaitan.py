import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)



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
    

dataset = wastedata(
    data_dir="C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/train/TrashNet Dataset/trashnet-master/data/dataset-resized/dataset-resized"
)



len(dataset)



image, label = dataset[3]
print(label)
image



data_dir = "C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/train/TrashNet Dataset/trashnet-master/data/dataset-resized/dataset-resized"
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
print(target_to_class)



transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

data_dir = "C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/train/TrashNet Dataset/trashnet-master/data/dataset-resized/dataset-resized"
dataset = wastedata(data_dir, transform)



image, label = dataset[1]
image.shape



# iterate over dataset
for image, label in dataset:
    break



dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for images, labels in dataloader:
    break
print(images.shape, labels.shape)


print(labels)



class wasteClassifer(nn.Module):
    def __init__(self, num_classes=53):
        super(wasteClassifer, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output
    

    

model = wasteClassifer(num_classes=3)
print(str(model)[:500])



example_out = model(images)
example_out.shape # [batch_size, num_classes]



# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)



criterion(example_out, labels)
print(example_out.shape, labels.shape)



transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_folder = "C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/train"
valid_folder = "C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/valid"
test_folder = "C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/test"

train_dataset = wastedata(train_folder, transform=transform)
val_dataset = wastedata(valid_folder, transform=transform)
test_dataset = wastedata(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



# Simple training loop
num_epochs = 5
train_losses, val_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = wasteClassifer(num_classes=53)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):
        # Move inputs and labels to the device
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
        for images, labels in tqdm(val_loader, desc='Validation loop'):
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
         
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")
    

torch.save(model.state_dict(),r"C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/tp.pt" )

    
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()



model = wasteClassifer(num_classes=53)
model.load_state_dict(torch.load(r"C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/tp.pt", weights_only=True))
model.eval()


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient calculations
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)  # Get predicted class indices

            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

    accuracy = (num_correct / num_samples) * 100
    print(f"Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}%")
    model.train()  # Set model back to training mode
    return accuracy
check_accuracy(dataloader,model,device)