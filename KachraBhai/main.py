import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import timm
import os
import cv2 as cv

vid = cv.VideoCapture(0)

for i in range(3):
    return_value, image = vid.read()
    cv.imwrite('opencv'+str(i)+'.jpg', image)
del(vid)
imgPath = r"C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/opencv1.jpg"


# Loading model and all tht stufffff
MODEL_PATH = r"C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/tp.pt"
DATASET_PATH = r"C:/Users/aarav/OneDrive/Desktop/文档/Hackspire/KachraBhai/train/dataset-resized"
IMAGE_PATH = (imgPath)  
IMAGE_SIZE = (128, 128)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. Define model
class wasteClassifer(nn.Module):
    def __init__(self, num_classes=3):
        super(wasteClassifer, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 3. Load classes
class_names = ImageFolder(DATASET_PATH).classes
num_classes = len(class_names)

# 4. Load model
model = wasteClassifer(num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 5. Define transform (same as training)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# 6. Predict image function
def predict_image(image_path, model, transform, class_names, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)

    predicted_class = class_names[predicted.item()]
    return predicted_class

# 7. Showing da image with prediction
def show_prediction(image_path, model, transform, class_names, device):
    image = Image.open(image_path).convert("RGB")
    pred = predict_image(image_path, model, transform, class_names, device)

    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {pred}", fontsize=16)
    plt.show()
    return pred

# 8. Run prediction
if os.path.exists(IMAGE_PATH):
    prediction = show_prediction(IMAGE_PATH, model, transform, class_names, device)
    print(f" Predicted class: {prediction}")
else:
    print(f" Image not found: {IMAGE_PATH}")
