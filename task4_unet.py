import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ======================
# Simple UNet definition
# ======================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(1, 16)
        self.down2 = DoubleConv(16, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.out_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = nn.MaxPool2d(2)(x1)
        x3 = self.down2(x2)
        x4 = self.up1(x3)
        return torch.sigmoid(self.out_conv(x4))

# ======================
# Dataset
# ======================
class MRIDataset(Dataset):
    def __init__(self, img_dir, transform=None, limit=10):
        self.img_dir = img_dir
        self.images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")][:limit]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, img  # input = target (dummy seg)

# ======================
# Training (1 epoch)
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

train_dir = "/home/groups/comp3710/OASIS/keras_png_slices_train"
dataset = MRIDataset(train_dir, transform=transform, limit=20)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Using device: {device}")
model.train()
for epoch in range(1):  # just 1 epoch
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# ======================
# Save a segmentation example
# ======================
model.eval()
with torch.no_grad():
    sample_img, _ = dataset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    pred_mask = model(sample_img).squeeze().cpu().numpy()

plt.subplot(1, 2, 1)
plt.title("Input MRI")
plt.imshow(sample_img.cpu().squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Segmentation")
plt.imshow(pred_mask, cmap="gray")
plt.axis("off")

plt.savefig("unet_segmentation.png")
print("✅ Segmentation saved as unet_segmentation.png")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Simple UNet Implementation
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.outc(d1)

# =========================
# Dummy Dataset Loader (OASIS PNGs)
# =========================
class OASISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith(".png")])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, torch.zeros((1, 256, 256))  # dummy mask for now

# =========================
# Setup
# =========================
device = torch.device("cpu")  # force CPU
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dir = "/home/groups/comp3710/OASIS/keras_png_slices_train"
dataset = OASISDataset(train_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# Train for just 2 epochs
# =========================
epochs = 2
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks.squeeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "unet_oasis.pth")
print("Training complete, model saved as unet_oasis.pth")

# =========================
# Save segmentation PNG
# =========================
model.eval()
imgs, _ = next(iter(dataloader))
with torch.no_grad():
    outputs = model(imgs.to(device))
pred = torch.argmax(outputs, dim=1).cpu().numpy()

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for i in range(4):
    axes[i].imshow(imgs[i][0], cmap="gray")
    axes[i].imshow(pred[i], cmap="jet", alpha=0.5)
    axes[i].axis("off")
plt.savefig("unet_segmentation.png")
print("Segmentation results saved as unet_segmentation.png")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ==== Dataset Class ====
class OASISSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("L")  # grayscale
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# ==== UNet Model ====
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(1, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(CBR(128, 256), CBR(256, 256))

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return torch.sigmoid(self.conv_last(d1))

# ==== Dice Loss ====
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# ==== Training Loop ====
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    train_dataset = OASISSegDataset(
        "/home/groups/comp3710/OASIS/keras_png_slices_train",
        "/home/groups/comp3710/OASIS/keras_png_slices_seg_train",
        transform=transform
    )
    val_dataset = OASISSegDataset(
        "/home/groups/comp3710/OASIS/keras_png_slices_validate",
        "/home/groups/comp3710/OASIS/keras_png_slices_seg_validate",
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    for epoch in range(5):  # demo: 5 epochs
        model.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = bce(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
        torch.save(model.state_dict(), "unet_oasis.pth")
    print("Training complete. Model saved as unet_oasis.pth")

if __name__ == "__main__":
    train()

