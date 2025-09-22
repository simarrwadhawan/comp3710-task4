import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ======================
# Dataset Loader
# ======================
class OasisDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = glob.glob(os.path.join(folder, "*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("L")  # grayscale MRI slice
        if self.transform:
            img = self.transform(img)
        return img, 0  # label not needed for VAE

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

data_path = "/home/groups/comp3710/OASIS/keras_png_slices_train"
dataset = OasisDataset(data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ======================
# VAE Model
# ======================
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64*16*16, latent_dim)
        self.fc_logvar = nn.Linear(64*16*16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 64*16*16)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ======================
# Loss Function
# ======================
def vae_loss(recon, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# ======================
# Training Loop
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(imgs)
        loss = vae_loss(recon, imgs, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Average loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "vae_oasis.pth")
print("✅ Training complete, model saved as vae_oasis.pth")

# ======================
# Visualisation
# ======================
def visualize_reconstructions(model, dataloader, device, num_images=8):
    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(dataloader))
        imgs = imgs.to(device)
        recon, _, _ = model(imgs)

        imgs = imgs.cpu().numpy()
        recon = recon.cpu().numpy()

        fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
        for i in range(num_images):
            axes[0, i].imshow(np.squeeze(imgs[i]), cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].imshow(np.squeeze(recon[i]), cmap="gray")
            axes[1, i].axis("off")
        plt.suptitle("Original (top) vs Reconstruction (bottom)")
        plt.savefig("reconstructions.png")
        plt.close()

visualize_reconstructions(model, dataloader, device)
print("🖼️ Reconstructions saved as reconstructions.png")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# ========================
# Dataset Loader for OASIS
# ========================
class OASISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(".png") or f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert("L")  # grayscale MRI
        if self.transform:
            image = self.transform(image)
        return image, 0  # no labels for VAE

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = OASISDataset("/home/groups/comp3710/OASIS", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ========================
# Variational Autoencoder
# ========================
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 64*64),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 64, 64)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ========================
# Training Setup
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch}, Average loss: {train_loss / len(dataloader.dataset):.4f}")

torch.save(model.state_dict(), "vae_oasis.pth")
print("✅ Training complete, model saved as vae_oasis.pth")

# ========================
# Visualisations
# ========================
def save_reconstructions(model, data_loader, device, filename="reconstructions.png", num_images=8):
    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(data_loader))
        imgs = imgs.to(device)
        recon, _, _ = model(imgs)

        imgs = imgs.cpu().numpy()
        recon = recon.cpu().numpy()

        fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
        for i in range(num_images):
            axes[0, i].imshow(np.squeeze(imgs[i]), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(np.squeeze(recon[i]), cmap='gray')
            axes[1, i].axis('off')
        plt.suptitle("Originals (top) vs Reconstructions (bottom)")
        plt.savefig(filename)
        plt.close()

def save_latent_space(model, data_loader, device, filename="latent_space.png", num_points=1000):
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in data_loader:
            imgs = imgs.to(device)
            _, mu, _ = model(imgs)
            latents.append(mu.cpu().numpy())
            labels.append(lbls)
            if len(latents) * imgs.size(0) > num_points:
                break
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)

    latents_2d = TSNE(n_components=2).fit_transform(latents)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter)
    plt.title("Latent Space (t-SNE)")
    plt.savefig(filename)
    plt.close()

save_reconstructions(model, dataloader, device)
save_latent_space(model, dataloader, device)
print("✅ Visualisations saved: reconstructions.png & latent_space.png")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# VAE Definition
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64, 400),
            nn.ReLU(),
        )
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 400)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(400, 64*64),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.decoder_fc(z)
        return self.decoder(h).view(-1, 1, 64, 64)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Dataset
class OASISDataset(Dataset):
    def __init__(self, root, transform=None):
        self.data = datasets.ImageFolder(root=root, transform=transform)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, _ = self.data[idx]
        return img

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = OASISDataset(root="/home/groups/comp3710/OASIS", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch}, Average loss: {train_loss / len(dataloader.dataset):.4f}")

# Save model
torch.save(model.state_dict(), "vae_oasis.pth")
print("Training complete, model saved as vae_oasis.pth")


#  Visualisation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def visualize_reconstructions(model, data_loader, device, num_images=8):
    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(data_loader))
        imgs = imgs.to(device)
        recon, _, _ = model(imgs)

        imgs = imgs.cpu().numpy()
        recon = recon.cpu().numpy()

        fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
        for i in range(num_images):
            axes[0, i].imshow(np.squeeze(imgs[i]), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(np.squeeze(recon[i]), cmap='gray')
            axes[1, i].axis('off')
        plt.suptitle("Originals (top) vs Reconstructions (bottom)")
        plt.savefig("reconstructions.png")
        plt.close()

def visualize_latent_space(model, data_loader, device, num_samples=2000):
    model.eval()
    zs, labels = [], []
    with torch.no_grad():
        for imgs, lbls in data_loader:
            imgs = imgs.to(device)
            _, mu, _ = model(imgs)
            zs.append(mu.cpu().numpy())
            labels.append(lbls.numpy())
            if len(zs) * data_loader.batch_size > num_samples:
                break
    zs = np.concatenate(zs, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(zs)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(z_2d[:,0], z_2d[:,1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter)
    plt.title("Latent Space (t-SNE projection)")
    plt.savefig("latent_space.png")
    plt.close()

# Run visualisations after training
visualize_reconstructions(model, test_loader, device)
visualize_latent_space(model, test_loader, device)

print(" Visualisations saved: reconstructions.png, latent_space.png")

# === Save reconstructions ===
def save_reconstructions(model, data_loader, device, filename="reconstructions.png", num_images=8):
    model.eval()
    with torch.no_grad():
        imgs, _ = next(iter(data_loader))
        imgs = imgs.to(device)
        recon, _, _ = model(imgs)

        imgs = imgs.cpu().numpy()
        recon = recon.cpu().numpy()

        fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
        for i in range(num_images):
            axes[0, i].imshow(np.squeeze(imgs[i]), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(np.squeeze(recon[i]), cmap='gray')
            axes[1, i].axis('off')
        plt.suptitle("Originals (top) vs Reconstructions (bottom)")
        plt.savefig(filename)
        plt.close()

save_reconstructions(model, test_loader, device)

# === Save latent space (t-SNE) ===
def save_latent_space(model, data_loader, device, filename="latent_space.png", num_points=1000):
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in data_loader:
            imgs = imgs.to(device)
            _, mu, _ = model(imgs)
            latents.append(mu.cpu().numpy())
            labels.append(lbls.numpy())
            if len(latents) * imgs.size(0) > num_points:
                break

    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    latents_2d = TSNE(n_components=2).fit_transform(latents)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter)
    plt.title("Latent Space (t-SNE)")
    plt.savefig(filename)
    plt.close()

save_latent_space(model, test_loader, device)
