import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Velocity field network
class VelocityField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # x (2D) + t (1D)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x, t):
        # Concatenate time to input
        t = t.expand(x.shape[0], 1)
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

# Sample source and target
def sample_gaussian(n):
    return torch.randn(n, 2)

def sample_ring(n):
    theta = torch.rand(n) * 2 * torch.pi
    r = torch.ones_like(theta)  # radius = 1
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)

# Train the velocity field using flow matching loss
def train_flow_field(model, optimizer, steps=2000):
    for step in range(steps):
        x0 = sample_gaussian(128)
        x1 = sample_ring(128)
        t = torch.rand(128, 1)
        xt = (1 - t) * x0 + t * x1

        target_v = (x1 - x0)  # full displacement
        target_vt = target_v * 1.0  # keep full vector (FM uses this directly)

        pred_v = model(xt, t)
        loss = ((pred_v - target_vt)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

# Run integration from x0 using learned v(x, t)
def integrate_flow(model, x0):
    t = torch.linspace(0, 1, 100)

    def odefunc(t, x):
        return model(x, t.expand(x.shape[0], 1))

    x_t = odeint(odefunc, x0, t, method='rk4')  # (time, batch, 2)
    return x_t

# Visualization
def visualize_flow():
    model = VelocityField()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_flow_field(model, optimizer)

    with torch.no_grad():
        x0 = sample_gaussian(256)
        x_t = integrate_flow(model, x0)
        x_final = x_t[-1]

        plt.figure(figsize=(6, 6))
        plt.scatter(x0[:, 0], x0[:, 1], color='blue', alpha=0.3, label='Start')
        plt.scatter(x_final[:, 0], x_final[:, 1], color='red', alpha=0.3, label='After Flow')
        plt.legend()
        plt.title("Flow Matching: Gaussian → Ring")
        plt.axis("equal")
        plt.show()

visualize_flow()
