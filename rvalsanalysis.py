import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import torch
from model import LSTMBurgers  # Ensure LSTMBurgers model is correctly imported
from data_load import analytical1dburgers, solve1dburgers, solve_one_step
from data_load import solve_multi_step  # Import necessary solvers

# Burgers' equation parameters
xlen = 2 * np.pi  
nx = 1000  
nt_total = 5000  
dt = 0.01  
mu = 0.01  
dx = xlen / (nx - 1)

c1 = 3  # wave speed

# CFL Conditions
CFL_conv = 0.5  
CFL_diff = 0.1  

# Compute CFL-based time step
dt_conv = CFL_conv * dx / c1
dt_diff = CFL_diff * dx**2 / mu
dt_stable = min(dt_conv, dt_diff)

# Adjust dt for stability
if dt > dt_stable:
    print(f"Warning: dt={dt} is too large! Adjusting to dt={dt_stable:.6e} for stability.")
    dt = dt_stable

print(f"Using dt = {dt:.6e} (CFL-constrained)")

# Load trained LSTM model
def load_trained_model(model_path, input_size, device="cpu"):
    model = LSTMBurgers(input_size=input_size, hidden_size=256, num_layers=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def prepare_input(data, idx, seq_length, device="cpu"):
    x = data[:, idx:idx + seq_length].T  # Shape: (seq_length, nx)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_trained_model("lstm_burgers_model.pth", input_size=nx, device=device)

# Define r_values for different experiments
r_values = np.concatenate([np.arange(0.0, 1.1, 0.5), np.arange(2, 500, 80)])

# Initialize error storage
all_errors = np.zeros((r_values.shape[0], nt_total - 2))
all_true_errors = np.zeros((r_values.shape[0], nt_total - 2))

delta = 1e-3
count = 0

# Loop over different r values
for r in r_values:
    errors = []
    true_errors = []
    
    sigma_o = 0.01
    sigma_b = np.sqrt(r) * sigma_o
    X1_b_sol = solve1dburgers(xlen, nx, 1, dt, mu, delta)    
    X1_b = X1_b_sol[:, [1]]

    W_mat = (r / (r + 1)) * np.eye(nx, nx)
    X1_a = X1_b + W_mat @ (observations[:, [1]] - X1_b)

    true_errors.append(np.sqrt(np.mean((X1_a - ground_truth[:, [1]])**2)))
    X1_b_sol[:, [1]] = X1_a

    sol49 = solve_multi_step(X1_a.flatten(), mu, xlen, nx, dt, nt_total - 2)
    sol49_complete = np.hstack((X1_b_sol, sol49))

    errors.append(np.sqrt(np.mean((sol49_complete - ground_truth) ** 2)))
    X_b = X1_a

    # Time-stepping loop
    for step in range(nt_total - 3):
        if step + 3 > 5:
            input_tensor = prepare_input(X1_b_sol, step + 3 - 5, 10, device=device)
            with torch.no_grad():
                u_pred = model(input_tensor)
            X_b = u_pred.cpu().numpy().reshape(nx, 1)

            X_a = X_b + W_mat @ (observations[:, [step + 2]] - X_b)
            true_errors.append(np.sqrt(np.mean((X_a - ground_truth[:, [step + 2]])**2)))

            X1_b_sol = np.hstack((X1_b_sol, X_a))
            sol_remaining = solve_multi_step(X_a.flatten(), mu, xlen, nx, dt, nt_total - (step + 3))
            sol_complete = np.hstack((X1_b_sol, sol_remaining))
            errors.append(np.sqrt(np.mean((sol_complete - ground_truth) ** 2)))

        else:
            X_b = solve_one_step(X_b.flatten(), mu, xlen, nx, dt).reshape(nx, 1)
            X_a = X_b + W_mat @ (observations[:, [step + 2]] - X_b)
            true_errors.append(np.sqrt(np.mean((X_a - ground_truth[:, [step + 2]])**2)))

            X1_b_sol = np.hstack((X1_b_sol, X_a))
            sol_remaining = solve_multi_step(X_a.flatten(), mu, xlen, nx, dt, nt_total - (step + 3))
            sol_complete = np.hstack((X1_b_sol, sol_remaining))
            errors.append(np.sqrt(np.mean((sol_complete - ground_truth) ** 2)))

    all_true_errors[count, :] = true_errors
    all_errors[count, :] = errors
    count += 1

# Plotting RMSE vs Time Step for different r values
plt.figure(figsize=(10, 8))
markers = ['o', 's', '^', 'v', 'p', '*', 'h', 'H', 'D', 'd', '|', '_', 'x', '+', '.']
colors = plt.cm.get_cmap('tab20b', r_values.shape[0])

# Plot all_true_errors (one-step errors)
plt.subplot(2, 1, 1)
for i in range(r_values.shape[0]):
    errors = all_true_errors[i, :]
    r_value = r_values[i]
    marker = markers[i % len(markers)]
    color = colors(i)

    plt.semilogy(errors, marker=marker, linestyle='-', color=color, label=f"r = {r_value}")

plt.xlabel("Time Step")
plt.ylabel("One-Step RMSE")
plt.title("One-Step RMSE vs. Time Step for Different r Values")
plt.legend(loc='best', fontsize='small')
plt.grid(True)

# Plot all_errors (full solution errors)
plt.subplot(2, 1, 2)
for i in range(r_values.shape[0]):
    errors = all_errors[i, :]
    r_value = r_values[i]
    marker = markers[i % len(markers)]
    color = colors(i)

    plt.semilogy(errors, marker=marker, linestyle='-', color=color, label=f"r = {r_value}")

plt.xlabel("Time Step")
plt.ylabel("Full Solution RMSE")
plt.title("Full Solution RMSE vs. Time Step for Different r Values")
plt.legend(loc='best', fontsize='small')
plt.grid(True)

# Save the plot
plt.tight_layout()
plt.savefig("rmse_comparison.png", dpi=300)
plt.show()


# Save all_true_errors (one-step errors) and all_errors (full solution errors)
np.save("all_true_errors.npy", all_true_errors)  # Saves as NumPy array
np.save("all_errors.npy", all_errors)  # Saves as NumPy array

# Alternatively, save as CSV for easier inspection
np.savetxt("all_true_errors.csv", all_true_errors, delimiter=",")
np.savetxt("all_errors.csv", all_errors, delimiter=",")
