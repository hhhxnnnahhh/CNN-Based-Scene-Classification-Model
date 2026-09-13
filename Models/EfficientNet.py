import os
import random
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
#import matplotlib.cm as cm
import torchvision


from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, Subset
import torch.nn as nn
import torch.optim as optim  
from collections import Counter  
from sklearn.model_selection import train_test_split
#from sklearn.decomposition import PCA  
#from sklearn.manifold import TSNE  
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#import seaborn as sns  
#import warnings  
#warnings.filterwarnings("ignore")



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
data_dir = r"D:\UNI\Spring 2026\Introduction to Artificial Intelligence (CSE281)\Project\cse-281-spring-26-scene-style-classification\StyleClassificationIndoors\StyleClassificationIndoors\train"
Class_Mapping_Path = r"D:\UNI\Spring 2026\Introduction to Artificial Intelligence (CSE281)\Project\cse-281-spring-26-scene-style-classification\StyleClassificationIndoors\StyleClassificationIndoors\class_mapping.txt"
test_data_path = r"D:\UNI\Spring 2026\Introduction to Artificial Intelligence (CSE281)\Project\cse-281-spring-26-scene-style-classification\StyleClassificationIndoors\StyleClassificationIndoors\test"


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


batch_size = 32
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
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
# Initialize counters
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
            # 1. Basic Mode Check
            if img.mode in ['L', 'P']:
                stats["grayscale_palette"] += 1
                return False
           
            # 2. Resolution Check (Avoid tiny thumbnails)
            w, h = img.size
            if w < 64 or h < 64:
                stats["too_small"] += 1
                return False
           
            # 3. Aspect Ratio Check
            ratio = w / h
            if ratio > 3.0 or ratio < 0.33:
                stats["bad_aspect_ratio"] += 1
                return False
           
            # 4. Content Check (Convert to numpy to check for blankness)
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
# Load the dataset using the updated function
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
# 1. Load twice from the 'train' folder
train_full_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms, is_valid_file=check_and_count_invalid)
val_full_dataset   = datasets.ImageFolder(root=train_path, transform=test_transforms, is_valid_file=check_and_count_invalid)





# 2. Use 'train_full_dataset' to get the list of indices and the labels (targets)
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)  #drop_last dih bt5aly law a5r batch fiha image wa7da bs to drop it
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)




print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ==============================
# 8. VISUALIZE SAMPLE IMAGE
# ==============================
images, labels = next(iter(train_loader))
print("\nBatch shape:", images.shape)   # [32, 3, 224, 224]




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
# 9. Custom Efficient Net initialized b default weights
# ==============================
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # [cite: 10]
            nn.Conv2d(in_channels, reduced_dim, 1), # 
            nn.SiLU(), # 
            nn.Conv2d(reduced_dim, in_channels, 1), # 
            nn.Sigmoid() # 
        )
    def forward(self, x):
        return x * self.se(x) # [cite: 11]




class CustomEfficientNetB3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
      
        self.stem = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=3, stride=2, padding=1, bias=False), # 
            nn.BatchNorm2d(40), # 
            nn.SiLU() # 
        )


        self.blocks = nn.Sequential(
            # (Inverted Residual) Block
            nn.Conv2d(40, 40 * 6, 1, bias=False), 
            nn.BatchNorm2d(40 * 6), 
            nn.SiLU(), 
            nn.Conv2d(40 * 6, 40 * 6, 3, stride=1, padding=1, groups=40 * 6, bias=False), # Depthwise 
            nn.BatchNorm2d(40 * 6), 
            nn.SiLU(), 
            SqueezeExcitation(40 * 6, 40 // 4),  
            nn.Conv2d(40 * 6, 40, 1, bias=False), 
            nn.BatchNorm2d(40) 
        )


        self.head = nn.Sequential(
            nn.Conv2d(40, 1536, 1, bias=False), 
            nn.BatchNorm2d(1536), 
            nn.SiLU() 
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True), 
            nn.Linear(1536, num_classes) 
        )


    def forward(self, x):
        x = self.stem(x) 
        x = self.blocks(x) 
        x = self.head(x) 
        x = self.avgpool(x) 
        x = torch.flatten(x, 1)
        return self.classifier(x) 
# ==============================
# 9.5 INITIALIZE EL MODELLL
# ==============================
# 1. Loading Pretrained Base
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
pretrained_base = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
pretrained_state_dict = pretrained_base.state_dict()


# 2. Custom Model Initialization
model = CustomEfficientNetB3(num_classes=17)


# 3. Intelligent Weights Transfer (For the first round)
custom_state_dict = model.state_dict()
for name, param in pretrained_state_dict.items():
    if name in custom_state_dict and param.size() == custom_state_dict[name].size():
        custom_state_dict[name].copy_(param)
model.load_state_dict(custom_state_dict)


# 4. Load Best Updated Weights (If they exist from previous runs)
weights_path = "efficientnet_b3_best_weights.pt"
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"✅ Loaded the best updated weights from {weights_path}")
else:
    print("🚀 Starting with default pretrained weights.")


model.to(device)
# Phase 1: Freeze everything except the classifier
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True



# ==============================
# 10. optimizer w loss setup
# ==============================
model.classifier[0] = nn.Dropout(p=0.2, inplace=True)
model = model.to(device)


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


optimizer = optim.AdamW([
    {'params': model.stem.parameters(), 'lr': 1e-4},
    {'params': model.blocks.parameters(), 'lr': 1e-4},
    {'params': model.head.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], weight_decay=0.05)


scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1) #maybe change el cosine? 
# ==============================
# 10. early stopping setup
# ==============================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False


    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



# ==============================
# 11. train & eval
# ==============================
scaler = torch.cuda.amp.GradScaler()


def train_one_epoch_classification(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct_predictions, total_samples = 0.0, 0, 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        predicted_labels = outputs.argmax(dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
    
    return total_loss / total_samples, correct_predictions / total_samples


def evaluate_classification(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            
            predicted_labels = outputs.argmax(dim=1)
            
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    
    return average_loss, accuracy, np.array(all_predictions), np.array(all_labels)


# ==============================
# 11.5 main training loop
# ==============================
EPOCHS = 30
early_stopper = EarlyStopping(patience=5, min_delta=0.001)
best_val_accuracy = 0.0


train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []


for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch_classification(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = evaluate_classification(model, val_loader, criterion, device)
    
    scheduler.step()


    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), "efficientnet_b3_best_weights.pt")
        print(f"Best Accuracy Saved: {val_acc*100:.2f}%")


    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val Loss: {val_loss:.4f}")


    # Early Stopping Check
    early_stopper(val_loss)
    if early_stopper.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break



# ==============================
#  PLOTSS
# ==============================
# Plot Loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure(figsize=(8, 5))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import confusion_matrix


final_val_loss, final_val_acc, final_val_predictions, final_val_labels = evaluate_classification(model, val_loader, criterion, device)


cm = confusion_matrix(final_val_labels, final_val_predictions)
class_names = full_dataset.classes
num_classes = len(class_names)


plt.figure(figsize=(10, 8))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()


plt.xticks(ticks=np.arange(num_classes), labels=class_names, rotation=45, ha="right")
plt.yticks(ticks=np.arange(num_classes), labels=class_names)


for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)


plt.tight_layout()
plt.show()


# ==============================
# 12. process test data
# ==============================
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
                # Check for Grayscale or Palette
                if img.mode in ['L', 'P']:
                    self.test_stats["grayscale_palette"] += 1
                    # Convert to RGB anyway to allow the model to run
                    image = img.convert("RGB")
                else:
                    image = img.convert("RGB")




                # Check Resolution
                w, h = img.size
                if w < 64 or h < 64:
                    self.test_stats["too_small"] += 1
               
                self.test_stats["valid"] += 1
               
        except Exception:
            self.test_stats["corrupted"] += 1
            # black image so the code doesn't crash
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
           
        tensor_image = self.transform(image)
        return tensor_image, img_name






# ==============================
# 13. KAGGLE SUBMISSION
# ==============================

# Test Loader
test_data = KaggleTestDataset(test_data_path, transform=test_transforms)
test_loader_kaggle = DataLoader(test_data, batch_size=batch_size, shuffle=False)




model.load_state_dict(torch.load("efficientnet_b3_best_weights.pt", map_location=device))
model.eval()
submission_data = []


print("Generating predictions for Kaggle...")
with torch.no_grad():
    for imgs, filenames in test_loader_kaggle:
        imgs = imgs.to(device)
        outputs = model(imgs)
       
        # the index of the highest score
        predictions = torch.argmax(outputs, dim=1)
       
        for i in range(len(filenames)):
            submission_data.append({
                "ImageName": filenames[i],
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