import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn

def generate_sparse_observations(truth_solution, sigma_o, sparsity=0.2):

    nt, nx = truth_solution.shape
    sparse_observations = truth_solution + np.random.normal(0, sigma_o, (nt, nx))
    
    # Create a mask: True means observed, False means missing
    mask = np.random.rand(nt, nx) > sparsity
    
    # Set missing values to NaN
    sparse_observations[~mask] = np.nan

    return sparse_observations, mask



def save_losses(train_losses, test_losses, filename="losses.npz"):
    
    np.savez(filename, train_losses=train_losses, test_losses=test_losses)
    print(f"Losses saved to {filename}")



def plot_losses(filename="losses.npz"):
    
    data = np.load(filename)
    train_losses = data["train_losses"]
    test_losses = data["test_losses"]
    
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Training Loss", marker='o')
    plt.plot(range(20, len(test_losses) * 20 + 1, 20), test_losses, label="Test Loss", marker='s', linestyle='--')
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def total_variation_loss(pred, truth):
    """
    Computes the total variation loss while ensuring the predicted gradient matches the ground truth gradient.

    Args:
        pred (torch.Tensor): Predicted values of shape (batch_size, nx)
        truth (torch.Tensor): Ground truth values of shape (batch_size, nx)

    Returns:
        torch.Tensor: Total Variation loss value.
    """
    pred_diff = torch.abs(pred[:, 1:] - pred[:, :-1])  # Compute spatial gradient of prediction
    truth_diff = torch.abs(truth[:, 1:] - truth[:, :-1])  # Compute spatial gradient of ground truth
    return torch.mean(torch.abs(pred_diff - truth_diff))  # Penalize differences in gradients

def hybrid_loss(pred, truth, alpha=0.1):
    """
    Combines Mean Squared Error (MSE) loss and Total Variation (TV) loss 
    while considering the true gradient structure.

    Args:
        pred (torch.Tensor): Predicted values of shape (batch_size, nx)
        truth (torch.Tensor): Ground truth values of shape (batch_size, nx)
        alpha (float): Weight for TV loss.

    Returns:
        torch.Tensor: Hybrid loss value.
    """
    mse_loss = nn.MSELoss()(pred, truth)  # Standard MSE loss
    tv_loss = total_variation_loss(pred, truth)  # Gradient-aware TV loss
    return (1 - alpha)*mse_loss + alpha * tv_loss  # Combined loss


