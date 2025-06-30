import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
import torchvision
from torchvision.transforms import InterpolationMode
import numpy as np
import matplotlib.pyplot as plt

main_transform = transforms.Compose ([
    transforms.Resize((256,256), interpolation= InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=([0.485, 0.456, 0.406]), std=([0.229, 0.224, 0.225]))
])

class wastedataset(Dataset):
    def __init__(self, datasets,  transform=None):
        self.data = ImageFolder( datasets,transform=main_transform)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes


data_dir=r"C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/train"
dataset = wastedataset(data_dir)

len(dataset)

images, label = dataset[3]
print(label)
images

# Print class mapping
target_to_class = {v: k for k, v in dataset.data.class_to_idx.items()}
print(target_to_class)

# DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True)


def show_augmented_batch(data):
    images, labels = data
    grid = torchvision.utils.make_grid(images, nrow=8)
    npimg = grid.numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

# Show a few batches
for i, (images, labels) in enumerate(loader):
    print(f"Batch {i+1} with {len(images)} images")
    show_augmented_batch((images, labels))
    
    if i == 2:  # Stop after 3 batches
        break