import sys
import os
import glob
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Custom Dataset Class
class IconDataset(Dataset):
    def __init__(self, positive_paths, negative_paths, transform=None):
        self.images = positive_paths + negative_paths
        self.labels = [1] * len(positive_paths) + [0] * len(negative_paths)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Updated IconCNN with Adaptive Pooling
class IconCNN(nn.Module):
    def __init__(self):
        super(IconCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Adaptive pooling to 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)  # Adaptive pooling
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logging.info(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        logging.info(f"Validation Accuracy: {accuracy}%")
    return model

def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <icon_name>   e.g. python train.py jump5")
        print("  Expects: traindata/<icon_name>-1/*.png (positive), traindata/<icon_name>-0/*.png (negative)")
        print("  Saves:   model_cnn/<icon_name>_classifier.pth  (used by utils.find_icon_cnn)")
        print("  Also need: icon/<icon_name>-1.png as template for matchTemplate in utils.")
        sys.exit(1)

    icon_name = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging setup
    logging.basicConfig(level=logging.INFO)

    # Load all classes
    traindata_dir = '../traindata'
    all_classes = [os.path.basename(d) for d in glob.glob(os.path.join(traindata_dir, '*')) if os.path.isdir(d)]
    logging.info(f"所有类别: {all_classes}")

    # Load positive and negative samples
    positive_dir = os.path.join(traindata_dir, f'{icon_name}-1')
    negative_dir = os.path.join(traindata_dir, f'{icon_name}-0')
    positive_paths = sorted(glob.glob(os.path.join(positive_dir, '*.png')))
    negative_paths = sorted(glob.glob(os.path.join(negative_dir, '*.png')))

    if not positive_paths:
        logging.error(f"No positive samples in {positive_dir}. Need e.g. traindata/{icon_name}-1/*.png")
        sys.exit(1)
    if not negative_paths:
        logging.warning(f"No {icon_name}-0 samples in {negative_dir}; training may be unbalanced")
    if len(negative_paths) < len(positive_paths):
        logging.warning(f"Negative samples ({len(negative_paths)}) fewer than positive ({len(positive_paths)})")

    logging.info(f"Loaded {len(positive_paths)} positive samples from {positive_dir}")
    logging.info(f"Loaded {len(negative_paths)} negative samples from {negative_dir}")

    # Data augmentation without resizing
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=40, translate=(0.5, 0.5), scale=(0.6, 1.4)),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Dataset and splitting
    dataset = IconDataset(positive_paths, negative_paths, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Initialize and train model
    model = IconCNN().to(device)
    logging.info(f"Epoch 1: Training on {train_size} samples, Batch size: 8")
    model = train_model(model, train_loader, val_loader, num_epochs=50, device=device)

    # Save model
    model_dir = 'model_cnn'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{icon_name}_classifier.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"模型已保存到: {model_path}")

if __name__ == "__main__":
    main()