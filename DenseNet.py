import os
import random
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, Subset
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==============================
# 1. SET DATA PATH
# ==============================
data_dir = "/content/StyleClassificationIndoors/StyleClassificationIndoors/train"
Class_Mapping_Path = "/content/StyleClassificationIndoors/StyleClassificationIndoors/class_mapping.txt"
# ==============================
# 2. CHECK DATASET STRUCTURE
# ==============================
print("Folders in dataset:", os.listdir(data_dir))
class_to_id = {}


with open(Class_Mapping_Path, "r") as file:
    for line in file:
        line = line.strip()


        if line == "":
            continue


        class_name, class_id = line.split(":")
        class_name = class_name.strip()
        class_id = int(class_id.strip())

        class_to_id[class_name] = class_id

id_to_class = {v: k for k, v in class_to_id.items()}

print("Class to ID mapping:")
for class_name, class_id in class_to_id.items():
    print(class_name, "->", class_id)

num_classes = len(class_to_id)
print("\nNumber of classes:", num_classes)

# ==============================
# 3. DEFINE TRANSFORMS
# ==============================
image_size = 128
batch_size = 32

train_transforms = transforms.Compose([
   transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225])    # ImageNet std
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# 4. LOAD DATASET (WITH CLEANING & COUNTING)
# ==============================
stats = {
    "corrupted": 0,
    "grayscale_palette": 0,
    "too_small": 0,
    "bad_aspect_ratio": 0,
    "blank_or_solid": 0,
    "valid": 0
}

def check_and_count_invalid(path):
    try:
        with Image.open(path) as img:
            img.verify()


        with Image.open(path) as img:
            # Basic Mode Check
            if img.mode in ['L', 'P']:
                stats["grayscale_palette"] += 1
                return False


            # Resolution Check
            w, h = img.size
            if w < 64 or h < 64:
                stats["too_small"] += 1
                return False


            # Aspect Ratio Check
            ratio = w / h
            if ratio > 3.0 or ratio < 0.33:
                stats["bad_aspect_ratio"] += 1
                return False


            # Content Check (Convert to numpy to check for blankness)
            img_rgb = img.convert('RGB')
            img_array = np.array(img_rgb)
            if np.std(img_array) < 2: # Very low variance = solid color
                stats["blank_or_solid"] += 1
                return False


        stats["valid"] += 1
        return True


    except Exception:
        stats["corrupted"] += 1
        return False

train_path = data_dir
full_dataset = datasets.ImageFolder(
    root=train_path,
    transform=train_transforms,
    is_valid_file=check_and_count_invalid
)

print("\n--- Dataset Cleaning Results ---")
print(f"Corrupted images excluded: {stats['corrupted']}")
print(f"B&W or Palette images excluded: {stats['grayscale_palette']}")
print(f"too_small images excluded: {stats['too_small']}")
print(f"bad_aspect_ratio images excluded: {stats['bad_aspect_ratio']}")
print(f"blank_or_solid images excluded: {stats['blank_or_solid']}")
print(f"Total valid images kept: {stats['valid']}")
print(f"Actual images in Dataset object: {len(full_dataset)}")

# ==============================
# 5. CHECK CLASS DISTRIBUTION
# ==============================
labels_all = [label for _, label in full_dataset]
print("\nClass distribution:", Counter(labels_all))

# ==============================
# 6. SPLIT DATASET
# ==============================
train_full_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms, is_valid_file=check_and_count_invalid)
val_full_dataset   = datasets.ImageFolder(root=train_path, transform=test_transforms, is_valid_file=check_and_count_invalid)

indices = np.arange(len(train_full_dataset))
targets = train_full_dataset.targets


# Split indices 80/20 with stratification
train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=SEED,
    stratify=targets
)


# Apply those indices to the two different versions
train_dataset = Subset(train_full_dataset, train_indices) # Gets the augmented version
val_dataset   = Subset(val_full_dataset, val_indices)     # Gets the clean version

# ==============================
# 7. CREATE DATALOADERS
# ==============================
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# ==============================
# 8. VISUALIZE SAMPLE IMAGE
# ==============================
images, labels = next(iter(train_loader))
print("\nBatch shape:", images.shape)  

def imshow(img, title=""):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = img * std + mean          # un-normalize
    img  = img.clamp(0, 1)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.title(title)
    plt.axis("off")
    plt.show()

imshow(images[0], title=f"Label: {full_dataset.classes[labels[0]]}")


# ==============================
# 9. CUSTOM DENSENET MODEL
# ==============================
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),


    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module(f'denselayer{i + 1}', layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class CustomDenseNet121(nn.Module):
    def __init__(self, num_classes=17):
        super().__init__()


        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


        # Dense Blocks & Transitions
        num_features = 64
        block_config = (6, 12, 24, 16)
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size=4, growth_rate=32)
            self.features.add_module(f'denselayer{i + 1}', block)
            num_features = num_features + num_layers * 32
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2


        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)


    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = F.dropout(out, p=0.5, training=self.training) #NEW!!
        out = self.classifier(out)
        return out

class KaggleTestDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = sorted(os.listdir(main_dir))

        self.test_stats = {
            "total_files": len(self.all_imgs),
            "corrupted": 0,
            "grayscale_palette": 0,
            "too_small": 0,
            "valid": 0
        }

    def __len__(self):
        return len(self.all_imgs)


    def __getitem__(self, idx):
        img_name = self.all_imgs[idx]
        img_loc = os.path.join(self.main_dir, img_name)


        try:
            with Image.open(img_loc) as img:
                if img.mode in ['L', 'P']:
                    self.test_stats["grayscale_palette"] += 1
                    image = img.convert("RGB")
                else:
                    image = img.convert("RGB")

                w, h = img.size
                if w < 64 or h < 64:
                    self.test_stats["too_small"] += 1


                self.test_stats["valid"] += 1


        except Exception:
            self.test_stats["corrupted"] += 1
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))


        tensor_image = self.transform(image)
        return tensor_image, img_name

# ==============================
# 10. INITIALIZE & LOAD WEIGHTS
# ==============================
import torch.nn.functional as F

model = CustomDenseNet121(num_classes=17).to(device)

from torchvision.models import densenet121, DenseNet121_Weights
pretrained_state = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).state_dict()

custom_state = model.state_dict()
for name, param in pretrained_state.items():
    if name in custom_state and "classifier" not in name:
        custom_state[name].copy_(param)


model.load_state_dict(custom_state)
print("Custom DenseNet-121 initialized and pre-trained weights loaded.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001 , weight_decay=1e-4) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

def train_one_epoch_classification(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0


    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)


        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()


    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def validate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_labels = []


    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)


            outputs = model(inputs)
            loss = criterion(outputs, labels)


            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()


            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy, all_predictions, all_labels

num_epochs = 30 
best_val_acc = 0.0
best_cnn_state = None
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}


# ==============================
# 11. THE MAIN LOOP
# ==============================
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch_classification(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc, _, _ = validate_model(
        model, val_loader, criterion, device
    )

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_cnn_state = model.state_dict()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)


    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
    print("-" * 30)

print("\n" + "="*30)
print(f"TRAINING FINISHED!")
print(f"Final Best Validation Accuracy: {best_val_acc*100:.2f}%")
print("="*30)

if best_cnn_state:
    model.load_state_dict(best_cnn_state)

model.to(device)

final_val_loss, final_val_accuracy, val_predictions, val_labels = validate_model(
    model=model,
    dataloader=val_loader,
    criterion=criterion,
    device=device
)

print("\nFinal Validation Results")
print("------------------------")
print("Validation Loss:", f"{final_val_loss:.4f}")
print("Validation Accuracy:", f"{final_val_accuracy*100:.2f}%")

# ==============================
# 12. PLOT RESULTS
# ==============================
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# ==============================
# 13. KAGGLE SUBMISSION
# ==============================
test_data_path ="/content/StyleClassificationIndoors/StyleClassificationIndoors/test"
test_data = KaggleTestDataset(test_data_path, transform=test_transforms)
test_loader_kaggle = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model.eval()
submission_data = []

print("Generating predictions for Kaggle...")
with torch.no_grad():
    for imgs, filenames in test_loader_kaggle:
        imgs = imgs.to(device)
        outputs = model(imgs)

        predictions = torch.argmax(outputs, dim=1)

        for i in range(len(filenames)):
           
            submission_data.append({
                "imageName": filenames[i],
                "ClassLabel": predictions[i].item()
            })
results = test_data.test_stats

print("\n" + "="*30)
print("TEST SET DIAGNOSTIC REPORT")
print("="*30)
print(f"Total files found:         {results['total_files']}")
print(f"Successfully processed:    {results['valid']}")
print(f"Corrupted (Unreadable):    {results['corrupted']}")
print(f"Non-RGB (L or P mode):     {results['grayscale_palette']}")
print(f"Low Resolution (<64px):    {results['too_small']}")
print("-" * 30)
print("Note: Corrupted images were replaced with black tensors for submission.")
print("="*30)

submission_df = pd.DataFrame(submission_data)

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'!")
