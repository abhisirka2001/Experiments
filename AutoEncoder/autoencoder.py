import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load MNIST
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST(root="./data", train=True, download=True, transform=transform), batch_size=128, shuffle=True)

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = Autoencoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Train
for epoch in range(5):
    for x, _ in train_loader:
        x_hat = model(x)
        loss = loss_fn(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Visualize results
x, _ = next(iter(train_loader))
with torch.no_grad():
    x_hat = model(x[:8])
plt.figure(figsize=(10, 2))
for i in range(8):
    plt.subplot(2, 8, i+1)
    plt.imshow(x[i][0], cmap="gray")
    plt.axis("off")
    plt.subplot(2, 8, i+9)
    plt.imshow(x_hat[i][0], cmap="gray")
    plt.axis("off")
plt.show()
