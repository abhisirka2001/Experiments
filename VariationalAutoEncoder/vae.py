
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load MNIST
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST(root="./data", train=True, download=True, transform=transform), batch_size=128, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # mean
        self.fc22 = nn.Linear(400, 20)  # logvar
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# VAE loss
def vae_loss(x_recon, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Train
for epoch in range(5):
    for x, _ in train_loader:
        x_recon, mu, logvar = vae(x)
        loss = vae_loss(x_recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.2f}")

# Visualize reconstructions
with torch.no_grad():
    x, _ = next(iter(train_loader))
    x_hat, _, _ = vae(x)
    x_hat = x_hat.view(-1, 1, 28, 28)
plt.figure(figsize=(10, 2))
for i in range(8):
    plt.subplot(2, 8, i+1)
    plt.imshow(x[i][0], cmap="gray")
    plt.axis("off")
    plt.subplot(2, 8, i+9)
    plt.imshow(x_hat[i][0], cmap="gray")
    plt.axis("off")
plt.show()
