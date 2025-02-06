import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler
import os
from sklearn.metrics import classification_report

# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs = 15
learning_rate = 0.0001
model_path = "marine_animal_classifier.pth"

# Dataset Paths
data_dir = "/content/drive/MyDrive/marine_animals"

# Data Transforms with Augmentation
# EfficientNet-B0 default input size is 224x224, but you can use larger sizes for better performance
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Can be increased to 240, 260, etc. for better performance
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    # EfficientNet normalization values
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Data
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model Definition
model = models.efficientnet_v2_l(pretrained=True)  # You can choose b0 through b7
for param in model.parameters():  # Freeze all layers initially
    param.requires_grad = False

# Unfreeze the last few layers for fine-tuning
# EfficientNet has different layer names compared to ResNet
for param in model.features[-3:].parameters():  # Unfreeze last 3 blocks
    param.requires_grad = True

# Replace classifier
num_classes = len(train_dataset.classes)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, num_classes)  # 1280 is the output features for efficientnet_b0
)
model = model.to(device)

# Loss (with label smoothing) and Optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# Only optimize parameters that require gradients
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=learning_rate, 
                       weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
scaler = GradScaler()  # Mixed Precision Training

# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():  # Mixed Precision
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        lr_scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

        # Save Best Model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_path)
            print(f"Model saved with accuracy: {accuracy * 100:.2f}%")

train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
print(f"Final model saved at {model_path}")