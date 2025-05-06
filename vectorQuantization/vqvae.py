import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
LATENT_DIM = 64
NUM_EMBEDDINGS = 512
BATCH_SIZE = 128
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(datasets.MNIST(root="./data", train=True, download=True, transform=transform),
                          batch_size=BATCH_SIZE, shuffle=True)

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, LATENT_DIM, 1)              # 14x14 -> 14x14 (latent space)
        )

    def forward(self, x):
        return self.net(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 32, 4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        # z: (B, C, H, W)
        B, C, H, W = z.shape
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, C)  # (B*H*W, C)

        # Compute distances
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
            + torch.sum(self.embedding.weight ** 2, dim=1)
        )  # (B*H*W, num_embeddings)

        # Get nearest embedding index
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + 0.25 * e_latent_loss

        # Preserve gradients
        quantized = z + (quantized - z).detach()
        return quantized, loss


# VQ-VAE
class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.vq = VectorQuantizer(NUM_EMBEDDINGS, LATENT_DIM)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.vq(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss

# Model and optimizer
model = VQVAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for x, _ in train_loader:
        x = x.to(DEVICE)
        x_recon, vq_loss = model(x)
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Visualize reconstructions
x, _ = next(iter(train_loader))
x = x.to(DEVICE)
with torch.no_grad():
    x_recon, _ = model(x)
x = x.cpu()
x_recon = x_recon.cpu()

plt.figure(figsize=(10, 2))
for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(x[i][0], cmap="gray")
    plt.axis("off")
    plt.subplot(2, 8, i + 9)
    plt.imshow(x_recon[i][0], cmap="gray")
    plt.axis("off")
plt.suptitle("Top: Original | Bottom: VQ-VAE Reconstruction")
plt.show()
