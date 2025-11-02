
# train_lstm_burgers_99_1_split_physics.py

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from model_copy import LSTMBurgers, combined_physics_loss
from data_load import BurgersDataset
from torch.optim.lr_scheduler import ExponentialLR

# ---------------- Hyperparameters ----------------
c1           = 3.0
xlen         = 2 * np.pi
mu           = 0.01

seq_length   = 10
batch_size   = 128
num_epochs   = 500
learning_rate = 1e-4

nx           = 200
nt_total     = 2000
hidden_size  = 128
num_layers   = 5
noise_std    = 0.001

dx = xlen / (nx - 1)

# Loss weights
w1 = 0.3   # MSE
w2 = 0.4   # Rankine-Hugoniot
w3 = 0.3   # Shift
w4 = 0.0   # TV
w_phys = 0.3  # Physics (PDE residual)
loss_weights = (w1, w2, w3, w4, w_phys)

# ---------------- CFL & dt ----------------
CFL_conv = 0.5
CFL_diff = 0.1

dt_conv = CFL_conv * dx / c1
ndt_diff = CFL_diff * dx**2 / mu
# choose safe dt
dt = min(0.01, dt_conv, ndt_diff)

print(f"Using dt = {dt:.6e} (CFL-constrained)")
print(f"nx = {nx}, nt_total = {nt_total}")

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Dataset ----------------
full_ds = BurgersDataset(
    c=c1,
    xlen=xlen,
    nx=nx,
    nt=nt_total,
    dt=dt,
    mu=mu,
    seq_length=seq_length,
    noise_std=noise_std,
)

total_len = len(full_ds)
train_len = int(0.99 * total_len)
test_len  = total_len - train_len

train_ds = Subset(full_ds, list(range(train_len)))
test_ds  = Subset(full_ds, list(range(train_len, train_len + test_len)))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Dataset: {total_len} samples → train: {len(train_ds)}, test: {len(test_ds)}")

# ---------------- Model / Optimizer / Scheduler ----------------
model = LSTMBurgers(
    input_size=nx,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=0.1
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
end_lr = 1e-5
gamma = (end_lr / learning_rate) ** (1 / num_epochs)
scheduler = ExponentialLR(optimizer, gamma=gamma)

# ---------------- Training & Eval Functions ----------------
def train_physics(model, train_loader, test_loader, optimizer, scheduler,
                  device, num_epochs, mu, dx, dt, loss_weights, eval_interval=10):
    train_losses, test_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            prev = x[:, -1, :]

            optimizer.zero_grad()
            pred = model(x)
            loss = combined_physics_loss(pred, prev, y, mu, dx, dt, *loss_weights)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % eval_interval == 0:
            val = evaluate_physics(model, test_loader, device, mu, dx, dt, loss_weights)
            test_losses.append(val)
            print(f" → Eval Loss: {val:.6f}")

    return train_losses, test_losses


def evaluate_physics(model, loader, device, mu, dx, dt, loss_weights):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prev = x[:, -1, :]
            pred = model(x)
            loss = combined_physics_loss(pred, prev, y, mu, dx, dt, *loss_weights)
            total += loss.item()
    return total / len(loader)

# ---------------- Run Training ----------------
print(f"\n🚀 Starting physics-informed training...")
train_losses, test_losses = train_physics(
    model, train_loader, test_loader,
    optimizer, scheduler, device, num_epochs,
    mu, dx, dt, loss_weights,
    eval_interval=10
)

# ---------------- Final Evaluation ----------------
final_test = evaluate_physics(model, test_loader, device, mu, dx, dt, loss_weights)
print(f"Final test loss: {final_test:.6f}")

# ---------------- Save ----------------
torch.save(model.state_dict(), "lstm_burgers_physics.pth")
np.save("train_losses.npy", np.array(train_losses))
np.save("test_losses.npy",  np.array(test_losses))
print("✅ Saved model and losses.")


