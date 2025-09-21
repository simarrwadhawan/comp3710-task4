import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
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

