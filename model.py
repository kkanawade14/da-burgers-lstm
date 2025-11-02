import torch
import torch.nn as nn
from utils import save_losses

class LSTMBurgers(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        
        super(LSTMBurgers, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Take only the last time step's output
        out = self.fc(lstm_out[:, -1, :])
        
        return out
    
    
def loc_shift(pred, truth):
    du_dx_pred = pred[:, 1:] - pred[:, :-1]  # Approximate spatial gradient (pred)
    du_dx_truth = truth[:, 1:] - truth[:, :-1]  # Approximate spatial gradient (truth)
    
    # Locate the strongest gradient jump (shock location)
    shock_loc_pred = torch.argmax(torch.abs(du_dx_pred), dim=1)
    shock_loc_truth = torch.argmax(torch.abs(du_dx_truth), dim=1)
    shift_err = (shock_loc_pred.float() - shock_loc_truth.float()) / pred.shape[1]  
    return torch.mean(shift_err**2)
    
def rankine_hugoniot_loss(pred, truth):
    """
    Computes the Rankine-Hugoniot loss by comparing the shock jumps in the predicted and ground truth solutions.

    Args:
        pred (torch.Tensor): Predicted values of shape (batch_size, nx)
        truth (torch.Tensor): Ground truth values of shape (batch_size, nx)

    Returns:
        torch.Tensor: Rankine-Hugoniot loss value.
    """
    du_dx_pred = pred[:, 1:] - pred[:, :-1]  # Approximate spatial gradient (pred)
    du_dx_truth = truth[:, 1:] - truth[:, :-1]  # Approximate spatial gradient (truth)
    
    # Locate the strongest gradient jump (shock location)
    shock_loc_pred = torch.argmax(torch.abs(du_dx_pred), dim=1)
    shock_loc_truth = torch.argmax(torch.abs(du_dx_truth), dim=1)

    
    
    # Gather pre-shock and post-shock velocities
    u1_pred = torch.gather(pred, 1, (shock_loc_pred.unsqueeze(1) - 1).clamp(min=0))
    u2_pred = torch.gather(pred, 1, shock_loc_pred.unsqueeze(1))

    # print("shape of u1_pred is ", u1_pred.shape)
    # print("shape of u2_pred is ", u2_pred.shape)

    u1_truth = torch.gather(truth, 1, (shock_loc_truth.unsqueeze(1) - 1).clamp(min=0))
    u2_truth = torch.gather(truth, 1, shock_loc_truth.unsqueeze(1))
    
    # Rankine-Hugoniot residual: Enforcing the correct jump condition
    rh_residual_pred = (u2_pred - u1_pred) - (0.5 * (u2_pred**2 - u1_pred**2))
    rh_residual_truth = (u2_truth - u1_truth) - (0.5 * (u2_truth**2 - u1_truth**2))
    
    # Loss: Ensure predicted Rankine-Hugoniot residual matches the true one
    return torch.mean((rh_residual_pred - rh_residual_truth) ** 2)

def total_variation_loss(pred):
    """
    Computes the total variation loss to preserve sharp shocks.
    """
    return torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1]))

def combined_shock_loss(pred, truth,w1,w2,w3,w4):
    """
    Combined loss function including:
    1. MSE loss for standard accuracy
    2. Rankine-Hugoniot loss for enforcing physical conservation
    3. Total variation loss for maintaining sharpness of shocks

    Args:
        pred (torch.Tensor): Predicted values of shape (batch_size, nx)
        truth (torch.Tensor): Ground truth values of shape (batch_size, nx)
        w1, w2, w3 (float): Weights for each loss component.

    Returns:
        torch.Tensor: Combined loss value.
    """
    
    
    mse_loss = nn.MSELoss()(pred, truth)  # Standard accuracy
    rh_loss = rankine_hugoniot_loss(pred, truth)  # Shock conservation
    tv_loss = total_variation_loss(pred)  # Sharpness preservation
    shift_loss = loc_shift(pred,truth)
    return w1 * mse_loss + w2 * rh_loss + w3 * shift_loss  + w4 * tv_loss# Weighted combination

def train(model, train_loader, test_loader, criterion, optimizer, scheduler,device, num_epochs, learning_rate, eval_interval=20):
    """ Training function with periodic evaluation and loss tracking."""
    train_losses, test_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)  # Forward pass
            loss = criterion(output, y)  # Compute loss
            if torch.isnan(loss):
                print("NaN detected in loss! Skipping batch.")
                continue  # Skip batch if loss is NaN
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            epoch_loss += loss.item()
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.6f}, Learning rate : {current_lr}")
        
        # Evaluate model every eval_interval epochs
        if (epoch + 1) % eval_interval == 0:
            test_loss = evaluate(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            print(f"Test Loss after {epoch+1} epochs: {test_loss:.6f}")
    
    # Save trained model and losses
    # torch.save(model.state_dict(), "lstm_burgers_model.pth")
    print("Training complete.")
    return train_losses, test_losses
    
def evaluate(model, test_loader, criterion, device):
    """Evaluation function to calculate test error."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            output = model(x)  # Forward pass
            loss = criterion(output, y)  # Compute loss
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


