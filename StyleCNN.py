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

SCENE_ROOT = "/kaggle/input/competitions/cse-281-spring-26-scene-style-classification/StyleClassificationIndoors/StyleClassificationIndoors"

data_dir = SCENE_ROOT + "/train"
test_data_path = SCENE_ROOT + "/test"
Class_Mapping_Path = SCENE_ROOT + "/class_mapping.txt"

print("\nSelected scene root:", SCENE_ROOT)
print("Train directory:", data_dir)
print("Test directory:", test_data_path)
print("Class mapping file:", Class_Mapping_Path)
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
#image_size = 128
batch_size = 32


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   
                         [0.229, 0.224, 0.225])    
])




test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

stats = {"corrupted": 0, "grayscale_palette": 0, "valid": 0}


def check_and_count_invalid(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if img.mode in ['L','P']:
                stats["grayscale_palette"] += 1
                return False
            img.convert('RGB')
        stats["valid"] += 1
        return True
    except:
        stats["corrupted"] += 1
        return False
train_path = data_dir
full_dataset = datasets.ImageFolder(
    root=train_path,
    transform=train_transforms,
    is_valid_file=check_and_count_invalid
)


print("Dataset stats:", stats)
labels_all = [label for _, label in full_dataset]
print("\nClass distribution:", Counter(labels_all))
train_full_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms, is_valid_file=check_and_count_invalid)
val_full_dataset   = datasets.ImageFolder(root=train_path, transform=test_transforms, is_valid_file=check_and_count_invalid)


indices = np.arange(len(train_full_dataset))
targets = train_full_dataset.targets


train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=SEED,
    stratify=targets
)




train_dataset = Subset(train_full_dataset, train_indices) # Gets the augmented version
val_dataset   = Subset(val_full_dataset, val_indices)     # Gets the clean version
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)  #drop_last dih bt5aly law a5r batch fiha image wa7da bs to drop it
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
class StyleCNN(nn.Module):
    def __init__(self, num_classes=17):
        super(StyleCNN, self).__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2), # 112
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2), # 56
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2), # 28
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((7, 7)) 
        )

     
       
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #  12,544 -> 1024 
            nn.Linear(256 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            
            #  1024 -> 512 
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3), 
            
            #  512 -> num_classes 
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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




model = StyleCNN(num_classes=17).to(device)







def train_one_epoch_classification(model, dataloader, criterion, optimizer, device):
    model.train()
   
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
   
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
       
        optimizer.zero_grad()
       
        outputs = model(images)
        loss = criterion(outputs, labels)
     
        loss.backward()
       
        optimizer.step()
       
        total_loss += loss.item() * images.size(0)
        predicted_labels = outputs.argmax(dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
   
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
   
    return average_loss, accuracy






def validate_model(model, dataloader, criterion, device):
    model.eval()  
   
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
   
    all_preds = []
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
           
            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
           
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
   
    return average_loss, accuracy, all_preds, all_labels
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)


num_epochs = 30
best_val_acc = 0.0
best_cnn_state = None
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        patience: How many epochs to wait after last time validation loss improved.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
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
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
early_stopper = EarlyStopping(patience=5, min_delta=0.001)


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
        torch.save(best_cnn_state, "best_checkpoint.pt")




    early_stopper(val_loss)
    if early_stopper.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}. Stopping training.")
        break




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
    model.load_state_dict(torch.load("best_checkpoint.pt"))




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
plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.legend()
plt.show()
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
cm = confusion_matrix(val_labels, val_predictions)


class_names = [id_to_class[i] for i in range(num_classes)]


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