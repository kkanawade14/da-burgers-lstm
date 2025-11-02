import torch
import torch.nn as nn
from utils import save_losses

class LSTMBurgers(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(LSTMBurgers, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def loc_shift(pred, truth):
    du_dx_pred = pred[:, 1:] - pred[:, :-1]
    du_dx_truth = truth[:, 1:] - truth[:, :-1]
    shock_loc_pred = torch.argmax(torch.abs(du_dx_pred), dim=1)
    shock_loc_truth = torch.argmax(torch.abs(du_dx_truth), dim=1)
    shift_err = (shock_loc_pred.float() - shock_loc_truth.float()) / pred.shape[1]
    return torch.mean(shift_err ** 2)

def rankine_hugoniot_loss(pred, truth):
    du_dx_pred = pred[:, 1:] - pred[:, :-1]
    du_dx_truth = truth[:, 1:] - truth[:, :-1]
    shock_loc_pred = torch.argmax(torch.abs(du_dx_pred), dim=1)
    shock_loc_truth = torch.argmax(torch.abs(du_dx_truth), dim=1)
    u1_pred = torch.gather(pred, 1, (shock_loc_pred.unsqueeze(1) - 1).clamp(min=0))
    u2_pred = torch.gather(pred, 1, shock_loc_pred.unsqueeze(1))
    u1_truth = torch.gather(truth, 1, (shock_loc_truth.unsqueeze(1) - 1).clamp(min=0))
    u2_truth = torch.gather(truth, 1, shock_loc_truth.unsqueeze(1))
    rh_residual_pred = (u2_pred - u1_pred) - (0.5 * (u2_pred**2 - u1_pred**2))
    rh_residual_truth = (u2_truth - u1_truth) - (0.5 * (u2_truth**2 - u1_truth**2))
    return torch.mean((rh_residual_pred - rh_residual_truth) ** 2)

def total_variation_loss(pred):
    return torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1]))

import torch
import torch.nn.functional as F

def physics_informed_loss(pred, prev, mu, dx, dt):
    """
    Finite-difference PINN loss with padding + scaling.
    
    Args:
      pred, prev:        [B, nx] tensors (u^{n+1}, u^n)
      mu:                viscosity
      dx, dt:            grid spacings
    Returns:
      scalar:            mean squared, scaled PDE residual
    """
    # Time derivative at all points
    du_dt = (pred - prev) / dt            # [B, nx]

    # Interior finite differences
    du_dx_int   = (pred[:, 2:] - pred[:, :-2])   / (2 * dx)    # [B, nx-2]
    d2u_dx2_int = (pred[:, 2:] 
                   - 2 * pred[:, 1:-1] 
                   + pred[:, :-2])                 / (dx*dx)   # [B, nx-2]

    # Pad boundaries by replication so shape → [B, nx]
    du_dx   = F.pad(du_dx_int,   (1,1), mode='replicate')  # [B, nx]
    d2u_dx2 = F.pad(d2u_dx2_int, (1,1), mode='replicate')  # [B, nx]

    # Full residual at every grid point
    R = du_dt + pred * du_dx - mu * d2u_dx2               # [B, nx]

    # Nondimensionalize: remove dx⁻² and mu scales
    R_scaled = R * (dx*dx) / mu

    return torch.mean(R_scaled**2)


def combined_physics_loss(pred, prev, truth, mu, dx, dt, w1, w2, w3, w4, w_phys):
    mse_loss = nn.MSELoss()(pred, truth)
    rh_loss = rankine_hugoniot_loss(pred, truth)
    shift_loss = loc_shift(pred, truth)
    tv_loss = total_variation_loss(pred)
    phys_loss = physics_informed_loss(pred, prev, mu, dx, dt)
    return w1 * mse_loss + w2 * rh_loss + w3 * shift_loss + w4 * tv_loss + w_phys * phys_loss

def train(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs, mu, dx, dt, loss_weights, eval_interval=20):
    train_losses, test_losses = [], []
    w1, w2, w3, w4, w_phys = loss_weights
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            prev = x[:, -1, :]
            
            optimizer.zero_grad()
            pred = model(x)
            loss = combined_physics_loss(pred, prev, y, mu, dx, dt, w1, w2, w3, w4, w_phys)
            
            if torch.isnan(loss):
                print("NaN detected in loss! Skipping batch.")
                continue

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.6f}, Learning rate : {current_lr}")

        if (epoch + 1) % eval_interval == 0:
            test_loss = evaluate(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            print(f"Test Loss after {epoch+1} epochs: {test_loss:.6f}")

    print("Training complete.")
    return train_losses, test_losses

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
    return total_loss / len(test_loader)
