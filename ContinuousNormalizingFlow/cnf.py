import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint

torch.manual_seed(42)

# Define CNF model: vector field + trace of Jacobian
class CNF(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),  # +1 for time
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, dim)
        )

    def forward(self, t, x):
        t_input = t.expand(x.shape[0], 1)
        inp = torch.cat([x, t_input], dim=1)
        return self.net(inp)

# Wrapper to compute dx/dt and dlogp/dt
class CNFWrapper(nn.Module):
    def __init__(self, cnf_model):
        super().__init__()
        self.cnf = cnf_model

    def forward(self, t, states):
        x, logp = states
        x.requires_grad_(True)

        v = self.cnf(t, x)

        # Compute divergence (trace of Jacobian)
        div = 0.0
        for i in range(x.shape[1]):
            grad = torch.autograd.grad(v[:, i].sum(), x, create_graph=True)[0][:, i]
            div += grad

        dlogp = -div
        return v, dlogp

# Sample from target (ring)
def sample_ring(n):
    theta = torch.rand(n) * 2 * torch.pi
    x = torch.cos(theta)
    y = torch.sin(theta)
    return torch.stack([x, y], dim=1)

# Log probability of standard normal
def standard_normal_logprob(x):
    return -0.5 * (x ** 2).sum(dim=1) - x.shape[1] * 0.5 * torch.log(torch.tensor(2 * torch.pi))

# Training
def train_cnf(model, steps=3000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(steps):
        x1 = sample_ring(128)
        logp1 = torch.zeros(x1.shape[0])

        # Integrate backward from t=1 → t=0
        states = (x1, logp1)
        t = torch.tensor([1.0, 0.0])
        x0, logp0 = odeint(model, states, t, atol=1e-5, rtol=1e-5, method='dopri5')
        x0, logp0 = x0[-1], logp0[-1]

        log_prob = standard_normal_logprob(x0)
        loss = - (log_prob + logp0).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

# Visualize transformation
def visualize_cnf(model):
    with torch.no_grad():
        x = torch.randn(256, 2)
        logp = torch.zeros(x.shape[0])
        t = torch.linspace(0.0, 1.0, 100)

        states = odeint(model, (x, logp), t, atol=1e-5, rtol=1e-5)
        x_t = states[0]
        x_final = x_t[-1]

        plt.figure(figsize=(6, 6))
        plt.scatter(x[:, 0], x[:, 1], color='blue', alpha=0.3, label='Initial Gaussian')
        plt.scatter(x_final[:, 0], x_final[:, 1], color='red', alpha=0.3, label='CNF Output')
        plt.title("CNF: Gaussian → Ring")
        plt.axis("equal")
        plt.legend()
        plt.show()

# Instantiate and train on CPU
cnf_model = CNF(dim=2)
cnf_wrapper = CNFWrapper(cnf_model)

train_cnf(cnf_wrapper)
visualize_cnf(cnf_wrapper)
