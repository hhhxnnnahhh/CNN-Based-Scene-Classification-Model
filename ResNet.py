# Scene Style Classification — ResNet101 + StyleCNN (Manual Layers)
import os
import copy
import random
import hashlib
import shutil
from collections import defaultdict, Counter
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
# ==============================
# 0. REPRODUCIBILITY & DEVICE
# ==============================
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
data_dir = r"D:\sem4\ai\cse-281-spring-26-scene-style-classification\StyleClassificationIndoors\StyleClassificationIndoors\train"
Class_Mapping_Path = r"D:\sem4\ai\cse-281-spring-26-scene-style-classification\StyleClassificationIndoors\StyleClassificationIndoors\class_mapping.txt"
test_data_path = r"D:\sem4\ai\cse-281-spring-26-scene-style-classification\StyleClassificationIndoors\StyleClassificationIndoors\test"

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
        class_to_id[class_name.strip()] = int(class_id.strip())

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
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
# ==============================
# 4. LOAD DATASET
# ==============================
stats = {"corrupted": 0, "grayscale_palette": 0, "valid": 0}

def check_and_count_invalid(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if img.mode in ['L', 'P']:
                stats["grayscale_palette"] += 1
                return False
            img.convert('RGB')
        stats["valid"] += 1
        return True
    except:
        stats["corrupted"] += 1
        return False

full_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=train_transforms,
    is_valid_file=check_and_count_invalid
)
print("Dataset stats:", stats)
# ==============================
# 5. CHECK CLASS DISTRIBUTION
# ==============================
labels_all = [label for _, label in full_dataset]
print("\nClass distribution:", Counter(labels_all))
# ==============================
# 6. SPLIT DATASET
# ==============================
train_full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms, is_valid_file=check_and_count_invalid)
val_full_dataset   = datasets.ImageFolder(root=data_dir, transform=test_transforms,  is_valid_file=check_and_count_invalid)

indices = np.arange(len(train_full_dataset))
targets = train_full_dataset.targets

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=SEED,
    stratify=targets
)

train_dataset = Subset(train_full_dataset, train_indices)
val_dataset   = Subset(val_full_dataset,   val_indices)
# ==============================
# 7. CREATE DATALOADERS
# ==============================
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

print("Training images:",   len(train_dataset))
print("Validation images:", len(val_dataset))
# ==============================
# 8. FROM-SCRATCH BUILDING BLOCKS
# ==============================

class ManualConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
        super(ManualConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.out_channels = out_channels

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

class ManualBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(ManualBatchNorm1d, self).__init__()
        self.eps      = eps
        self.momentum = momentum
        self.gamma    = nn.Parameter(torch.ones(num_features))
        self.beta     = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var',  torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var  = x.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean = self.running_mean
            var  = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class ManualBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(ManualBatchNorm2d, self).__init__()
        self.eps      = eps
        self.momentum = momentum
        self.gamma    = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta     = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        # running stats
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var',  torch.ones(1,  num_features, 1, 1))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var  = x.var(dim=(0, 2, 3),  keepdim=True, unbiased=False)
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean = self.running_mean   # use tracked stats at eval
            var  = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class ManualMaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=None, padding=0):
        super(ManualMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride      = stride if stride is not None else kernel_size
        self.padding     = padding

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)


class ManualReLU(nn.Module):
    """
    Rectified Linear Unit (ReLU) implemented from scratch.
    Sets all negative values to zero.
    """
    def __init__(self, inplace=False):
        super(ManualReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            # clamp_ modifies the tensor in-place
            return x.clamp_(min=0)
        else:
            # torch.clamp returns a new tensor
            return torch.clamp(x, min=0)
            
print("[INFO] Manual layers defined: ManualConv2d, ManualBatchNorm2d, ManualMaxPool2d")
# ══════════════════════════════════════════════════════════════════════════════
# 9b. ResNet101 — FROM SCRATCH (Bottleneck + ResNet101 class + build_model)
# ══════════════════════════════════════════════════════════════════════════════
from torchvision.models import resnet101 as tv_resnet101, ResNet101_Weights

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1×1 conv: reduce channels
        self.conv1 = ManualConv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = ManualBatchNorm2d(planes)
        # 3×3 conv: spatial processing 
        self.conv2 = ManualConv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = ManualBatchNorm2d(planes)
        # 1×1 conv: expand channels
        self.conv3 = ManualConv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = ManualBatchNorm2d(planes * self.expansion)
        self.relu  = ManualReLU(inplace=True)
        # Skip-connection projection (when dimensions change)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity          # residual addition
        return self.relu(out)


class ResNet101(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet101, self).__init__()
        self.in_channels = 64

        self.conv1 = ManualConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1     = ManualBatchNorm2d(64)
        self.relu    = ManualReLU(inplace=True)
        self.maxpool = ManualMaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1  = self._make_layer(64,   3, stride=1)   # 256  ch, 56×56
        self.layer2  = self._make_layer(128,  4, stride=2)   # 512  ch, 28×28
        self.layer3  = self._make_layer(256, 23, stride=2)   # 1024 ch, 14×14
        self.layer4  = self._make_layer(512,  3, stride=2)   # 2048 ch,  7×7

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, planes, num_blocks, stride):
        downsample    = None
        out_channels  = planes * Bottleneck.expansion

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                ManualConv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                ManualBatchNorm2d(out_channels),
            )

        layers = [Bottleneck(self.in_channels, planes,
                             stride=stride, downsample=downsample)]
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.in_channels, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, ManualConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, ManualBatchNorm2d):
                nn.init.constant_(m.gamma, 1)
                nn.init.constant_(m.beta,  0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)   
        x = self.layer2(x)  
        x = self.layer3(x)   
        x = self.layer4(x)   
        x = self.avgpool(x)       
        x = torch.flatten(x, 1)  
        x = self.fc(x)          
        return x


def build_model(num_classes, device):
    print("[MODEL] Building ResNet-101 from scratch ...")
    model = ResNet101(num_classes=1000)

    print("[MODEL] Downloading / loading ImageNet pretrained weights ...")
    tv_model = tv_resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    tv_sd    = tv_model.state_dict()
    our_sd   = model.state_dict()

    matched, skipped = 0, 0
    for k in our_sd:
        if k in tv_sd and tv_sd[k].shape == our_sd[k].shape:
            our_sd[k] = tv_sd[k]
            matched += 1
        else:
            skipped += 1
    model.load_state_dict(our_sd)
    print(f"[MODEL] Weights copied: {matched} matched, {skipped} skipped")
    del tv_model, tv_sd   # free memory

    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if any(tag in name for tag in ("layer3", "layer4", "fc")):
            p.requires_grad = True

    in_f = 512 * Bottleneck.expansion   # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_f, 1024),
        ManualBatchNorm1d(1024),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(1024, 512),
        ManualBatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable params: {trainable:,}")
    return model.to(device)


print("[INFO] Bottleneck, ResNet101, build_model defined.")
# ══════════════════════════════════════════════════════════════════════════════
# 9c. ADVANCED DATASET CLASSES
# ══════════════════════════════════════════════════════════════════════════════
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

TTA_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(
            transforms.ToTensor()(c)) for c in crops])),
])

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        return self.transform(Image.open(path).convert('RGB')), label

class FlatImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.transform   = transform
        self.image_paths = sorted([
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ])
        if not self.image_paths:
            raise FileNotFoundError(f'No images found in {directory}')
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        p   = self.image_paths[idx]
        img = Image.open(p).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, os.path.basename(p)

class FlatTTADataset(Dataset):
    def __init__(self, directory):
        self.image_paths = sorted([
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ])
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        p = self.image_paths[idx]
        return TTA_TRANSFORM(Image.open(p).convert('RGB')), os.path.basename(p)

print('[INFO] Dataset classes defined: SubsetWithTransform, FlatImageDataset, FlatTTADataset')
# ══════════════════════════════════════════════════════════════════════════════
# 10. EMA / MIXUP / CUTMIX / TRAIN / EVAL / PREDICT / WARMUP
# ══════════════════════════════════════════════════════════════════════════════
class ModelEMA:
    """
    Exponential Moving Average of model weights.
    shadow_k = decay * shadow_k + (1 - decay) * param_k
    EMA weights are smoother and usually generalize better.
    """
    def __init__(self, model, decay=0.9995):
        self.ema    = copy.deepcopy(model).eval()
        self.decay  = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}
        for p in self.ema.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        sd = model.state_dict()
        for k in self.shadow:
            self.shadow[k] = (self.decay * self.shadow[k]
                              + (1 - self.decay) * sd[k].float())
        self.ema.load_state_dict(
            {k: v.to(self.ema.state_dict()[k].dtype)
             for k, v in self.shadow.items()}
        )


def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, H, W = x.shape
    cw = int(W * np.sqrt(1 - lam)); ch = int(H * np.sqrt(1 - lam))
    cx = np.random.randint(W);      cy = np.random.randint(H)
    x1, x2 = np.clip(cx - cw//2, 0, W), np.clip(cx + cw//2, 0, W)
    y1, y2 = np.clip(cy - ch//2, 0, H), np.clip(cy + ch//2, 0, H)
    xm = x.clone()
    xm[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return xm, y, y[idx], lam

def mixed_loss(criterion, out, ya, yb, lam):
    return lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)


def train_one_epoch(model, ema, loader, criterion,
                    optimizer, device, config, epoch):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in tqdm(loader, desc=f'  Train E{epoch:02d}', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        use_mix = random.random() < config['mixup_prob']
        if use_mix:
            fn = mixup_data if random.random() < 0.5 else cutmix_data
            imgs, ya, yb, lam = fn(
                imgs, labels,
                config['mixup_alpha'] if fn is mixup_data else config['cutmix_alpha']
            )
        optimizer.zero_grad()
        out  = model(imgs)
        loss = (mixed_loss(criterion, out, ya, yb, lam)
                if use_mix else criterion(out, labels))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            config['grad_clip']
        )
        optimizer.step()
        ema.update(model)
        total_loss += loss.item() * imgs.size(0)
        if not use_mix:
            correct += (out.argmax(1) == labels).sum().item()
        total   += imgs.size(0)
        total      += imgs.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='  Eval  ', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def predict_tta(model, loader, device):
    model.eval()
    filenames, preds = [], []
    with torch.no_grad():
        for crops, fnames in tqdm(loader, desc='  TTA   ', leave=False):
            B, nc, C, H, W = crops.shape
            crops = crops.view(B * nc, C, H, W).to(device)
            out   = F.softmax(model(crops), dim=1).view(B, nc, -1).mean(1)
            preds.extend(out.argmax(1).cpu().numpy())
            filenames.extend(fnames)
    return filenames, preds


class WarmupScheduler:
    """
    Linear LR warmup for the first `warmup_epochs`, then hands off
    to a base scheduler (e.g. CosineAnnealingLR or ReduceLROnPlateau).
    """
    def __init__(self, optimizer, warmup_epochs, base_scheduler):
        self.opt      = optimizer
        self.warmup   = warmup_epochs
        self.base     = base_scheduler
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._epoch   = 0

    def step(self, val_loss=None):
        self._epoch += 1
        if self._epoch <= self.warmup:
            s = self._epoch / max(1, self.warmup)
            for pg, lr in zip(self.opt.param_groups, self.base_lrs):
                pg['lr'] = lr * s
        else:
            if isinstance(self.base,
                          optim.lr_scheduler.ReduceLROnPlateau):
                self.base.step(val_loss)
            else:
                self.base.step()

    def get_last_lr(self):
        return [pg['lr'] for pg in self.opt.param_groups]


print('[INFO] ModelEMA, Mixup/CutMix, train_one_epoch, evaluate, predict_tta, WarmupScheduler defined.')
# ==============================
# 11. EARLY STOPPING
# ==============================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
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
            self.counter   = 0


print("[INFO] EarlyStopping defined.")
# ══════════════════════════════════════════════════════════════════════════════
# 12. INITIALIZE MODEL, OPTIMIZER, SCHEDULER & CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # paths
    'train_dir'    : data_dir,
    'test_dir'     : test_data_path,
    # data
    'num_classes'  : num_classes,
    'val_split'    : 0.2,
    'batch_size'   : batch_size,
    'num_workers'  : 0,
    'device'       : str(device),
    # training
    'num_epochs'   : 20,
    'lr'           : 1e-4,
    'weight_decay' : 1e-4,
    'grad_clip'    : 5.0,
    'warmup_epochs': 3,
    # augmentation
    'mixup_prob'   : 0.5,
    'mixup_alpha'  : 0.4,
    'cutmix_alpha' : 1.0,
}

model = build_model(CONFIG['num_classes'], device)

criterion   = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer   = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay']
)
base_sched  = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
scheduler   = WarmupScheduler(optimizer,
                               CONFIG['warmup_epochs'], base_sched)
ema         = ModelEMA(model, decay=0.9995)

CHECKPOINT  = 'best_model_checkpoint.pt'

print(f"[INFO] Model     : {type(model).__name__}")
print(f"[INFO] Device    : {device}")
print(f"[INFO] Max epochs: {CONFIG['num_epochs']}")
# ══════════════════════════════════════════════════════════════════════════════
# 13. MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

best_val_acc     = 0.0
best_model_state = None
history          = {'train_loss': [], 'train_acc': [],
                    'val_loss':   [], 'val_acc':   []}
early_stopper    = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(1, CONFIG['num_epochs'] + 1):
    train_loss, train_acc = train_one_epoch(
        model, ema, train_loader, criterion,
        optimizer, device, CONFIG, epoch
    )
    val_loss, val_acc, _, _ = evaluate(
        ema.ema, val_loader, criterion, device
    )

    scheduler.step(val_loss)
    current_lr = scheduler.get_last_lr()[0]

    if val_acc > best_val_acc:
        best_val_acc     = val_acc
        best_model_state = copy.deepcopy(ema.ema.state_dict())
        torch.save(best_model_state, CHECKPOINT)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch [{epoch:02d}/{CONFIG['num_epochs']}]  lr={current_lr:.2e}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Val   Loss: {val_loss:.4f}   | Val   Acc: {val_acc*100:.2f}%")
    print('-' * 50)

    early_stopper(val_loss)
    if early_stopper.early_stop:
        print(f'Early stopping triggered at epoch {epoch}.')
        break

print('\n' + '='*50)
print('TRAINING FINISHED!')
print(f'Best Validation Accuracy: {best_val_acc*100:.2f}%')
print('='*50)
# ══════════════════════════════════════════════════════════════════════════════
# 14. FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print('\nLoading best EMA weights for final evaluation...')
ema.ema.load_state_dict(torch.load(CHECKPOINT))
ema.ema.to(device)

final_val_loss, final_val_accuracy, val_predictions, val_labels = evaluate(
    ema.ema, val_loader, criterion, device
)

print('\nFinal Validation Results')
print('------------------------')
print(f'Validation Loss:     {final_val_loss:.4f}')
print(f'Validation Accuracy: {final_val_accuracy*100:.2f}%')
# ==============================
# 15. PLOT TRAINING CURVES
# ==============================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'],   label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'],   label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()
# ==============================
# 16. CONFUSION MATRIX
# ==============================
cm = confusion_matrix(val_labels, val_predictions)
class_names = [id_to_class[i] for i in range(num_classes)]

plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues')
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
# ══════════════════════════════════════════════════════════════════════════════
# 17. KAGGLE SUBMISSION (with TTA)
# ══════════════════════════════════════════════════════════════════════════════

tta_dataset = FlatTTADataset(CONFIG['test_dir'])
tta_loader  = DataLoader(tta_dataset, batch_size=8, shuffle=False,
                         num_workers=CONFIG['num_workers'],
                         pin_memory=(CONFIG['device'] == 'cuda'))

print('Generating TTA predictions for Kaggle submission...')
filenames, preds = predict_tta(ema.ema, tta_loader, device)

submission_df = pd.DataFrame({'ImageName': filenames, 'ClassLabel': preds})
submission_df.to_csv('submission.csv', index=False)
print(f'Submission saved as submission.csv  ({len(submission_df)} rows)')
print('Located at:', os.path.abspath('submission.csv'))